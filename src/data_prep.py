"""Pipeline de download, padronização 16kHz mono, retomada idempotente e empacotamento .tar.gz (Spec 01 e Spec 05)."""

from __future__ import annotations

import argparse
import json
import re
import tarfile
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Optional, Set

import numpy as np
import soundfile as sf

from src.mixer import ensure_mono_16khz, prevent_clipping
from src.schemas import (
    DEFAULT_MAX_DURATION_SEC,
    HFDatasetSource,
    MultitalkerManifestEntry,
    REQUIRED_SAMPLE_RATE,
    SingleSpeakerManifestEntry,
)


SUPPORTED_HF_DATASETS = {
    "mls_pt": HFDatasetSource(
        hf_path="facebook/multilingual_librispeech", hf_config="portuguese", text_field="transcript"
    ),
    "fleurs_pt": HFDatasetSource(hf_path="google/fleurs", hf_config="pt_br", text_field="raw_transcription"),
    "coraa": HFDatasetSource(
        hf_path="Racoci/CORAA-v1.1",
        hf_config="default",
        text_field="text",
        split_aliases={"validation": "dev"},
    ),
    "coraa_mupe": HFDatasetSource(
        hf_path="nilc-nlp/CORAA-MUPE-ASR",
        hf_config="default",
        text_field="original_text",
        speaker_fields=["speaker_code", "speaker_type"],
        allowed_values={"audio_quality": ["high"]},
        strip_annotations=True,
    ),
    "common_voice_pt": HFDatasetSource(
        hf_path="mozilla-foundation/common_voice_17_0", hf_config="pt", text_field="sentence"
    ),
}

_ANNOTATION_RE = re.compile(r"\([^)]*\)")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.?!])")


def normalize_transcript(text: str, strip_annotations: bool = False) -> Optional[str]:
    """
    Normaliza uma transcrição para o manifesto (Spec 01 §2.2). Retorna None quando a
    amostra deve ser descartada (segmento truncado ou texto vazio após a limpeza).
    """
    text = unicodedata.normalize("NFC", text)
    if strip_annotations:
        if "<" in text or ">" in text:
            return None
        text = _ANNOTATION_RE.sub(" ", text)
    text = " ".join(text.split())
    if strip_annotations:
        text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return text or None


def _speaker_id_for_item(source: HFDatasetSource, item: Mapping[str, Any], dataset_key: str, idx: int) -> str:
    """Monta o speaker_id a partir de speaker_fields ou das colunas usuais (Spec 01 §2.1)."""
    if source.speaker_fields:
        parts = [str(item.get(f) or "").strip().replace("/", "-") for f in source.speaker_fields]
        if all(parts):
            return f"{dataset_key}_" + "_".join(parts)
    else:
        raw_spk = item.get("speaker_id") or item.get("client_id") or item.get("speaker")
        if raw_spk is not None and str(raw_spk).strip() not in ("", "None"):
            return str(raw_spk).strip()
    return f"{dataset_key}_{idx}"


def rewrite_manifest_audio_paths(
    manifest_path: str,
    output_manifest_path: str,
    new_root_dir: str = "/content/local_dataset",
    source_root_dir: Optional[str] = None,
) -> int:
    """
    Reescreve o campo audio_filepath de um manifesto JSONL para apontar para o SSD local
    da VM do Google Colab (/content/local_dataset/...), validando cada linha (Spec 01).
    """
    in_path = Path(manifest_path)
    out_path = Path(output_manifest_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src_root = Path(source_root_dir).resolve() if source_root_dir else in_path.parent.resolve()
    target_root = Path(new_root_dir)

    lines_out = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            orig_audio = Path(record["audio_filepath"])
            try:
                rel = orig_audio.resolve().relative_to(src_root)
            except ValueError:
                rel = Path("wavs") / orig_audio.name
            record["audio_filepath"] = str(target_root / rel)
            validated = MultitalkerManifestEntry.model_validate(record)
            lines_out.append(validated.model_dump_json())

    with open(out_path, "w", encoding="utf-8") as out_f:
        for item in lines_out:
            out_f.write(item + "\n")

    return len(lines_out)


def _read_existing_manifest_wav_names(manifest_path: Path) -> Set[str]:
    """Lê e valida um manifesto existente para retomada idempotente (resume=True)."""
    seen: Set[str] = set()
    if not manifest_path.exists():
        return seen
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            MultitalkerManifestEntry.model_validate(rec)
            seen.add(Path(rec["audio_filepath"]).name)
    return seen


def export_hf_dataset_to_nemo_manifest(
    dataset_key: str,
    split: str,
    output_dir: str,
    manifest_filename: str = "clean_manifest.json",
    max_samples: Optional[int] = None,
    min_duration: float = 0.5,
    max_duration: float = DEFAULT_MAX_DURATION_SEC,
    target_root_dir: Optional[str] = None,
    resume: bool = True,
    stride: int = 1,
    dataset: Optional[Any] = None,
) -> Path:
    """
    Baixa um dataset PT-BR via Hugging Face Datasets (streaming), força 16kHz mono e gera o
    manifesto validado pelo contrato SingleSpeakerManifestEntry (Spec 01).
    Suporta retomada idempotente (resume=True) caso uma execução anterior tenha sido interrompida.
    `stride` mantém 1 a cada N amostras (Spec 01 §2.3); `dataset` injeta um iterável já carregado (testes).
    """
    if dataset_key not in SUPPORTED_HF_DATASETS:
        raise ValueError(
            f"Dataset '{dataset_key}' não suportado. Opções: {list(SUPPORTED_HF_DATASETS.keys())}"
        )
    if stride < 1:
        raise ValueError(f"stride deve ser >= 1 (recebido: {stride}).")
    source = SUPPORTED_HF_DATASETS[dataset_key]

    out_root = Path(output_dir)
    wav_dir = out_root / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / manifest_filename

    existing_wav_names = _read_existing_manifest_wav_names(manifest_path) if resume else set()
    written = len(existing_wav_names)
    if max_samples is not None and written >= max_samples:
        return manifest_path

    ds = dataset
    if ds is None:
        from datasets import Audio, load_dataset  # Importação tardia para não exigir datasets em testes CPU puros

        ds = load_dataset(source.hf_path, source.hf_config, split=source.hub_split(split), streaming=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=REQUIRED_SAMPLE_RATE))

    mode = "a" if (resume and manifest_path.exists()) else "w"
    with open(manifest_path, mode, encoding="utf-8") as mf:
        for idx, item in enumerate(ds):
            if max_samples is not None and written >= max_samples:
                break
            if idx % stride:
                continue

            wav_filename = f"{dataset_key}_{split}_{idx:07d}.wav"
            if wav_filename in existing_wav_names:
                continue

            if any(str(item.get(field)) not in allowed for field, allowed in source.allowed_values.items()):
                continue

            raw_text = str(item.get(source.text_field) or item.get("transcription") or "")
            text = normalize_transcript(raw_text, strip_annotations=source.strip_annotations)
            if not text:
                continue

            audio_arr = np.asarray(item["audio"]["array"], dtype=np.float32)
            sr = int(item["audio"]["sampling_rate"])
            audio_arr = ensure_mono_16khz(audio_arr, sr)
            duration = round(len(audio_arr) / REQUIRED_SAMPLE_RATE, 4)

            if duration < min_duration or duration > max_duration:
                continue

            audio_arr = prevent_clipping(audio_arr)
            wav_path = wav_dir / wav_filename
            sf.write(str(wav_path), audio_arr, REQUIRED_SAMPLE_RATE, subtype="PCM_16")

            manifest_audio_path = (
                str(Path(target_root_dir) / "wavs" / wav_filename)
                if target_root_dir
                else str(wav_path)
            )

            entry = SingleSpeakerManifestEntry.from_clean_audio(
                audio_filepath=manifest_audio_path,
                duration=duration,
                text=text,
                speaker_id=_speaker_id_for_item(source, item, dataset_key, idx),
            )
            mf.write(entry.model_dump_json() + "\n")
            existing_wav_names.add(wav_filename)
            written += 1

    return manifest_path


def pack_dataset_for_colab_ssd(dataset_dir: str, output_tar_gz: str) -> Path:
    """
    Compacta os arquivos WAV e o manifesto em um único .tar.gz para evitar gargalo
    de I/O no Google Drive durante o treinamento no Colab (Spec 00 e Spec 01).
    """
    src_path = Path(dataset_dir)
    out_path = Path(output_tar_gz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # compresslevel=1: WAV PCM comprime pouco e o nível 9 padrão é lento demais para ~40 GB (Spec 01 §3)
    with tarfile.open(out_path, "w:gz", compresslevel=1) as tar:
        tar.add(src_path, arcname=".")

    return out_path


def prepare_colab_bundle(
    dataset_dir: str,
    output_tar_gz: str,
    colab_root_dir: str = "/content/local_dataset",
) -> Path:
    """
    Reescreve todos os manifestos JSON/JSONL dentro de dataset_dir para apontarem
    para colab_root_dir (/content/local_dataset) e gera o pacote .tar.gz (Spec 01 / Step 04).
    """
    root = Path(dataset_dir)
    for manifest_file in sorted(root.glob("*.json")):
        rewrite_manifest_audio_paths(
            manifest_path=str(manifest_file),
            output_manifest_path=str(manifest_file),
            new_root_dir=colab_root_dir,
            source_root_dir=str(root),
        )
    return pack_dataset_for_colab_ssd(dataset_dir=dataset_dir, output_tar_gz=output_tar_gz)


def ensure_colab_dataset_ready(
    dataset_key: str = "fleurs_pt",
    local_dataset_dir: str = "/content/local_dataset",
    drive_root_dir: str = "/content/drive/MyDrive/parakeet_ptbr",
    max_samples: Optional[int] = None,
    num_train_mixtures: int = 1000,
    num_val_mixtures: int = 200,
    clean_ratio: float = 0.30,
    val_stride: int = 1,
    update_state: bool = False,
) -> Path:
    """
    Orquestrador 100% Colab + Google Drive (Zero GB no computador local — Spec 01 e Spec 05).
    Os artefatos de cada corpus ficam em `drive_root_dir/<dataset_key>/` (Spec 01 §3):
    - Se `ptbr_dataset_16khz.tar.gz` e `tokenizer_ptbr_1024/tokenizer.vocab` já existirem nesse diretório,
      apenas extrai o `.tar.gz` para `local_dataset_dir` (/content/local_dataset) caso ainda não extraído.
    - Caso contrário, executa idempotentemente na VM do Colab:
      1. Step 01: Download e padronização 16kHz mono de `train` e `validation` (1 a cada `val_stride`) em `local_dataset_dir`.
      2. Step 02: Treinamento do tokenizador SentencePiece BPE (1.024 tokens) em `<corpus>/tokenizer_ptbr_1024`.
      3. Step 03: Mixagem sintética sobreposta (70/30 no treino, 100% no val) em `local_dataset_dir`.
      4. Step 04: Empacotamento de `local_dataset_dir` em `<corpus>/ptbr_dataset_16khz.tar.gz`.
    """
    from src.mixer import generate_mixed_dataset_from_manifest
    from src.tokenizer import extract_corpus_from_manifests, train_sentencepiece_bpe_ptbr

    local_root = Path(local_dataset_dir)
    drive_root = Path(drive_root_dir) / dataset_key
    local_root.mkdir(parents=True, exist_ok=True)
    drive_root.mkdir(parents=True, exist_ok=True)

    tar_gz_path = drive_root / "ptbr_dataset_16khz.tar.gz"
    tokenizer_dir = drive_root / "tokenizer_ptbr_1024"
    clean_train_manifest = local_root / "clean_train_manifest.json"
    clean_val_manifest = local_root / "clean_val_manifest.json"
    multi_train_manifest = local_root / "multitalker_train_manifest.json"
    multi_val_manifest = local_root / "multitalker_val_manifest.json"
    warmup_cmd = (
        "python -m src.trainer --stage warmup "
        f"--manifest {clean_train_manifest} --val_manifest {clean_val_manifest} "
        f"--checkpoint_dir /content/drive/MyDrive/parakeet_checkpoints/exp_{dataset_key}_warmup "
        f"--tokenizer_dir {tokenizer_dir} --update_state"
    )

    # Caminho rápido: pacote já construído em sessão anterior no Google Drive
    if tar_gz_path.exists() and (tokenizer_dir / "tokenizer.vocab").exists():
        if not multi_train_manifest.exists():
            with tarfile.open(tar_gz_path, "r:gz") as tar:
                tar.extractall(path=local_root)
        if update_state:
            from src.state import complete_step

            for s_id in (
                "step_01_data_prep_clean",
                "step_02_tokenizer_ptbr",
                "step_03_mixer_multitalker",
            ):
                complete_step(step_id=s_id)
            complete_step(
                step_id="step_04_pack_and_upload_drive",
                next_step_id="step_05_colab_stage1_warmup",
                next_action="Iniciar o treinamento da Fase 1 (warmup) na GPU A100 (Célula 4 do colab_runner.ipynb).",
                next_command=warmup_cmd,
                last_summary=f"Pacote {tar_gz_path} encontrado no Drive e extraído em {local_root}.",
                artifacts={
                    "colab_bundle_tar_gz": str(tar_gz_path),
                    "tokenizer_dir": str(tokenizer_dir),
                },
            )
        return tar_gz_path

    # 1. Step 01: Clean train + val
    if update_state:
        from src.state import complete_step, start_step

        start_step("step_01_data_prep_clean", notes=f"Baixando {dataset_key} (train + validation) no SSD do Colab.")
    if not (clean_train_manifest.exists() and clean_train_manifest.stat().st_size > 0):
        export_hf_dataset_to_nemo_manifest(
            dataset_key=dataset_key,
            split="train",
            output_dir=str(local_root),
            manifest_filename="clean_train_manifest.json",
            max_samples=max_samples,
            resume=True,
        )
    if not (clean_val_manifest.exists() and clean_val_manifest.stat().st_size > 0):
        export_hf_dataset_to_nemo_manifest(
            dataset_key=dataset_key,
            split="validation",
            output_dir=str(local_root),
            manifest_filename="clean_val_manifest.json",
            max_samples=max_samples,
            resume=True,
            stride=val_stride,
        )
    if update_state:
        complete_step(
            step_id="step_01_data_prep_clean",
            next_step_id="step_02_tokenizer_ptbr",
            artifacts={
                "clean_train_manifest": str(clean_train_manifest),
                "clean_val_manifest": str(clean_val_manifest),
            },
        )

    # 2. Step 02: Tokenizer SentencePiece BPE 1024 no Google Drive
    if update_state:
        start_step("step_02_tokenizer_ptbr", notes="Treinando tokenizador BPE 1024 em PT-BR no Colab.")
    if not (tokenizer_dir / "tokenizer.vocab").exists():
        corpus_txt = tokenizer_dir / "corpus_ptbr.txt"
        extract_corpus_from_manifests([str(clean_train_manifest)], str(corpus_txt))
        train_sentencepiece_bpe_ptbr(str(corpus_txt), str(tokenizer_dir))
    if update_state:
        complete_step(
            step_id="step_02_tokenizer_ptbr",
            next_step_id="step_03_mixer_multitalker",
            artifacts={"tokenizer_dir": str(tokenizer_dir)},
        )

    # 3. Step 03: Mixer Multitalker (Train 70/30 + Val)
    if update_state:
        start_step("step_03_mixer_multitalker", notes="Gerando misturas sobrepostas de 80ms no SSD do Colab.")
    generate_mixed_dataset_from_manifest(
        clean_manifest_path=str(clean_train_manifest),
        output_dir=str(local_root),
        output_manifest_path=str(multi_train_manifest),
        num_mixtures=num_train_mixtures,
        clean_ratio=clean_ratio,
        resume=True,
    )
    generate_mixed_dataset_from_manifest(
        clean_manifest_path=str(clean_val_manifest),
        output_dir=str(local_root),
        output_manifest_path=str(multi_val_manifest),
        num_mixtures=num_val_mixtures,
        clean_ratio=0.0,
        resume=True,
    )
    if update_state:
        complete_step(
            step_id="step_03_mixer_multitalker",
            next_step_id="step_04_pack_and_upload_drive",
            artifacts={
                "multitalker_train_manifest": str(multi_train_manifest),
                "multitalker_val_manifest": str(multi_val_manifest),
            },
        )

    # 4. Step 04: Empacotar em .tar.gz direto no Google Drive
    if update_state:
        start_step("step_04_pack_and_upload_drive", notes=f"Compactando {local_root} para {tar_gz_path} no Drive.")
    prepare_colab_bundle(
        dataset_dir=str(local_root),
        output_tar_gz=str(tar_gz_path),
        colab_root_dir=str(local_root),
    )
    if update_state:
        complete_step(
            step_id="step_04_pack_and_upload_drive",
            next_step_id="step_05_colab_stage1_warmup",
            next_action="Iniciar o treinamento da Fase 1 (warmup) na GPU A100 (Célula 4 do colab_runner.ipynb).",
            next_command=warmup_cmd,
            last_summary=f"Etapas 01 a 04 concluídas no Colab. Pacote salvo em {tar_gz_path}.",
            artifacts={"colab_bundle_tar_gz": str(tar_gz_path)},
        )

    return tar_gz_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Preparação de datasets PT-BR em 16kHz para NeMo (Spec 01).")
    parser.add_argument("--dataset", choices=list(SUPPORTED_HF_DATASETS.keys()), default="fleurs_pt")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default="/content/local_dataset")
    parser.add_argument("--manifest_filename", default="clean_manifest.json")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_duration", type=float, default=DEFAULT_MAX_DURATION_SEC)
    parser.add_argument(
        "--target_root_dir",
        default=None,
        help="Prefixo opcional de caminho no manifesto (ex.: /content/local_dataset).",
    )
    parser.add_argument("--pack_tar_gz", default=None, help="Caminho opcional para gerar o arquivo .tar.gz.")
    parser.add_argument(
        "--prepare_colab_bundle",
        action="store_true",
        help="Apenas reescreve manifestos existentes em --output_dir para /content/local_dataset e gera --pack_tar_gz (Step 04).",
    )
    parser.add_argument(
        "--ensure_colab_dataset",
        action="store_true",
        help="Executa/extrai as Etapas 01 a 04 diretamente na VM do Colab + Google Drive (Zero GB no Mac local).",
    )
    parser.add_argument("--local_dataset_dir", default="/content/local_dataset")
    parser.add_argument("--drive_root_dir", default="/content/drive/MyDrive/parakeet_ptbr")
    parser.add_argument("--num_train_mixtures", type=int, default=1000)
    parser.add_argument("--num_val_mixtures", type=int, default=200)
    parser.add_argument("--val_stride", type=int, default=1, help="Mantém 1 a cada N amostras da validação (Spec 01 §2.3).")
    parser.add_argument("--no_resume", action="store_true", help="Desativa retomada incremental e recria o manifesto do zero.")
    parser.add_argument("--update_state", action="store_true", help="Atualiza state/pipeline_state.json e USER.md.")
    args = parser.parse_args()

    if args.ensure_colab_dataset:
        archive = ensure_colab_dataset_ready(
            dataset_key=args.dataset,
            local_dataset_dir=args.local_dataset_dir,
            drive_root_dir=args.drive_root_dir,
            max_samples=args.max_samples,
            num_train_mixtures=args.num_train_mixtures,
            num_val_mixtures=args.num_val_mixtures,
            val_stride=args.val_stride,
            update_state=args.update_state,
        )
        print(f"[Spec 01 / Colab] Dataset e tokenizador prontos em {args.local_dataset_dir} (Drive: {archive})")
        return

    if args.prepare_colab_bundle:
        if not args.pack_tar_gz:
            raise ValueError("--prepare_colab_bundle exige --pack_tar_gz.")
        step_id = "step_04_pack_and_upload_drive"
        if args.update_state:
            from src.state import start_step

            start_step(
                step_id=step_id,
                command=f"python -m src.data_prep --prepare_colab_bundle --output_dir {args.output_dir} --pack_tar_gz {args.pack_tar_gz}",
                notes="Reescrevendo manifestos para /content/local_dataset e gerando pacote .tar.gz.",
            )
        try:
            archive = prepare_colab_bundle(args.output_dir, args.pack_tar_gz)
            print(f"[Spec 01 / Step 04] Bundle Colab SSD gerado em: {archive}")
            if args.update_state:
                from src.state import add_blocker, complete_step

                complete_step(
                    step_id=step_id,
                    next_step_id="step_05_colab_stage1_warmup",
                    next_action="Iniciar a Fase 1 (warmup) no Google Colab A100 após envio do .tar.gz ao Google Drive.",
                    next_command=(
                        "python -m src.trainer --stage warmup "
                        "--manifest /content/local_dataset/clean_train_manifest.json "
                        "--val_manifest /content/local_dataset/clean_val_manifest.json "
                        f"--checkpoint_dir /content/drive/MyDrive/parakeet_checkpoints/exp_{args.dataset}_warmup --update_state"
                    ),
                    last_summary=f"Bundle Colab {archive} empacotado com caminhos /content/local_dataset.",
                    artifacts={"colab_bundle_tar_gz": str(archive)},
                )
                add_blocker(
                    blocker_id="BLK-COLAB-DRIVE",
                    step_id="step_05_colab_stage1_warmup",
                    description="O pacote .tar.gz e o tokenizador precisam estar no Google Drive com runtime A100 ativo",
                    user_action_required=(
                        f"Envie {archive} e tokenizer_artifacts/tokenizer_ptbr_1024/ para "
                        "/content/drive/MyDrive/parakeet_ptbr/ e abra colab_runner.ipynb em uma GPU A100."
                    ),
                )
        except Exception as exc:
            if args.update_state:
                from src.state import fail_step

                fail_step(step_id=step_id, error_message=str(exc))
            raise
        return

    step_id = "step_01_data_prep_clean"
    if args.update_state:
        from src.state import start_step

        start_step(
            step_id=step_id,
            command=f"python -m src.data_prep --dataset {args.dataset} --split {args.split} --output_dir {args.output_dir}",
            notes=f"Exportando {args.dataset} ({args.split}) em 16kHz mono.",
        )

    try:
        manifest = export_hf_dataset_to_nemo_manifest(
            dataset_key=args.dataset,
            split=args.split,
            output_dir=args.output_dir,
            manifest_filename=args.manifest_filename,
            max_samples=args.max_samples,
            max_duration=args.max_duration,
            target_root_dir=args.target_root_dir,
            resume=not args.no_resume,
        )
        print(f"[Spec 01] Manifesto gerado com sucesso: {manifest}")

        archive_path: Optional[Path] = None
        if args.pack_tar_gz:
            archive_path = pack_dataset_for_colab_ssd(args.output_dir, args.pack_tar_gz)
            print(f"[Spec 01] Arquivo compactado pronto para Google Drive/Colab SSD: {archive_path}")

        if args.update_state:
            from src.state import complete_step, load_pipeline_state, update_handover

            st = load_pipeline_state()
            step_01 = next(s for s in st.steps if s.step_id == step_id)
            artifacts = dict(step_01.artifacts)
            artifacts[f"manifest_{args.split}"] = str(manifest)
            if archive_path:
                artifacts["tar_gz"] = str(archive_path)

            has_train = any("train" in k for k in artifacts)
            has_val = any(k in artifacts for k in ("manifest_validation", "manifest_val", "manifest_test"))

            if has_train and not has_val:
                # Mantém step_01 como 'in_progress' até gerar também o manifesto de validação
                update_handover(
                    step_id=step_id,
                    next_action="Exportar o split de validação (clean_val_manifest.json) para concluir os dados limpos.",
                    next_command=(
                        f".venv/bin/python -m src.data_prep --dataset {args.dataset} --split validation "
                        f"--output_dir {args.output_dir} --manifest_filename clean_val_manifest.json --update_state"
                    ),
                    last_summary=f"Split '{args.split}' de {args.dataset} exportado ({manifest}). Falta o split de validação.",
                    artifacts=artifacts,
                )
            else:
                train_m = artifacts.get("manifest_train", str(manifest))
                complete_step(
                    step_id=step_id,
                    next_step_id="step_02_tokenizer_ptbr",
                    next_action="Treinar o tokenizador SentencePiece BPE PT-BR de 1.024 tokens sobre o manifesto limpo.",
                    next_command=(
                        f".venv/bin/python -m src.tokenizer --manifests {train_m} "
                        "--output_dir tokenizer_artifacts/tokenizer_ptbr_1024 --update_state"
                    ),
                    last_summary=f"Splits de treino e validação em PT-BR exportados em {args.output_dir}.",
                    artifacts=artifacts,
                )
    except Exception as exc:
        if args.update_state:
            from src.state import fail_step

            fail_step(step_id=step_id, error_message=str(exc))
        raise


if __name__ == "__main__":
    main()
