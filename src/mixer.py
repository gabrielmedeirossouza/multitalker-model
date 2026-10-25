"""Motor de mistura sintética multitalker, retomada idempotente e extração de máscaras de 80ms (Spec 02)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from src.schemas import (
    FRAME_DURATION_SEC,
    MAX_PEAK_AMPLITUDE,
    MixerConfig,
    MultitalkerCut,
    MultitalkerManifestEntry,
    REQUIRED_SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    SingleSpeakerManifestEntry,
    SpeakerSupervision,
    expected_frames_for_duration,
)


def ensure_mono_16khz(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Garante que o vetor de áudio está em 16.000 Hz, mono, float32 (Spec 01)."""
    if sample_rate != REQUIRED_SAMPLE_RATE:
        raise ValueError(
            f"Spec 01 violada: taxa de amostragem esperada {REQUIRED_SAMPLE_RATE} Hz, recebido {sample_rate} Hz."
        )
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    elif audio.ndim > 2:
        raise ValueError(f"Formato de áudio inválido com {audio.ndim} dimensões.")
    return audio.astype(np.float32, copy=False)


def compute_vad_frame_mask(
    aligned_audio: np.ndarray,
    rms_threshold: float = 0.01,
    samples_per_frame: int = SAMPLES_PER_FRAME,
) -> List[int]:
    """Calcula a máscara binária de atividade (0 ou 1) em quadros de 80ms (1.280 amostras)."""
    total_samples = len(aligned_audio)
    duration_sec = round(total_samples / REQUIRED_SAMPLE_RATE, 4)
    num_frames = expected_frames_for_duration(duration_sec)
    mask: List[int] = []

    for frame_idx in range(num_frames):
        start = frame_idx * samples_per_frame
        end = min(total_samples, start + samples_per_frame)
        segment = aligned_audio[start:end]
        if len(segment) == 0:
            mask.append(0)
            continue
        rms = float(np.sqrt(np.mean(np.square(segment, dtype=np.float64))))
        mask.append(1 if rms >= rms_threshold else 0)

    return mask


def prevent_clipping(audio: np.ndarray, max_peak: float = MAX_PEAK_AMPLITUDE) -> np.ndarray:
    """Reescalona proporcionalmente o sinal caso o pico ultrapasse max_peak (Spec 02)."""
    peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
    if peak > max_peak and peak > 0.0:
        audio = audio * (max_peak / peak)
    return audio.astype(np.float32, copy=False)


def _sample_distinct_speaker_pair(
    records: List[Dict[str, Any]],
    rng: np.random.Generator,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Sorteia 2 registros garantindo locutores distintos sempre que houver speaker_ids diferentes."""
    idx_a = int(rng.integers(0, len(records)))
    rec_a = records[idx_a]
    spk_a = rec_a.get("speaker_id")

    candidates = [
        i for i, r in enumerate(records)
        if i != idx_a and (spk_a is None or r.get("speaker_id") != spk_a)
    ]
    if not candidates:
        candidates = [i for i in range(len(records)) if i != idx_a]

    idx_b = int(rng.choice(candidates))
    return rec_a, records[idx_b]


def _resolve_local_audio_path(raw_filepath: str, manifest_path: str) -> str:
    """
    Resolve o caminho físico do WAV localmente caso o manifesto já contenha um prefixo
    remoto (/content/local_dataset/...) mas os WAVs estejam na subpasta wavs/ ao lado do manifesto.
    """
    p = Path(raw_filepath)
    if p.exists():
        return str(p)
    fallback = Path(manifest_path).resolve().parent / "wavs" / p.name
    if fallback.exists():
        return str(fallback)
    return raw_filepath


def mix_two_speakers(
    audio_a: np.ndarray,
    text_a: str,
    audio_b: np.ndarray,
    text_b: str,
    output_wav_path: str,
    config: Optional[MixerConfig] = None,
    offset_sec: Optional[float] = None,
    sir_db: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    write_wav: bool = True,
    speaker_id_a: Optional[str] = None,
    speaker_id_b: Optional[str] = None,
    manifest_audio_filepath: Optional[str] = None,
) -> Tuple[np.ndarray, List[MultitalkerManifestEntry]]:
    """
    Combina dois áudios limpos em PT-BR com deslocamento temporal e variação de SIR,
    gerando a onda misturada e 2 registros espelhados validados (Alvo A e Alvo B).
    """
    cfg = config or MixerConfig()
    rng = rng or np.random.default_rng()

    audio_a = ensure_mono_16khz(audio_a, cfg.sample_rate)
    audio_b = ensure_mono_16khz(audio_b, cfg.sample_rate)

    max_allowed_samples = int(cfg.max_duration_sec * cfg.sample_rate)
    if len(audio_a) > max_allowed_samples:
        audio_a = audio_a[:max_allowed_samples]
    if len(audio_b) > max_allowed_samples:
        audio_b = audio_b[:max_allowed_samples]

    len_a = len(audio_a)
    len_b = len(audio_b)
    dur_a = len_a / cfg.sample_rate
    dur_b = len_b / cfg.sample_rate

    if dur_a < 0.5 or dur_b < 0.5:
        raise ValueError(
            f"Spec 01 violada: áudios de entrada devem ter duração >= 0.5s (recebido: A={dur_a:.3f}s, B={dur_b:.3f}s)."
        )

    # Determina o offset temporal de B considerando D_A e D_B para atingir a sobreposição desejada (Spec 02)
    if offset_sec is None:
        target_overlap_ratio = float(rng.uniform(cfg.min_overlap_ratio, cfg.max_overlap_ratio))
        raw_offset = (dur_a - target_overlap_ratio * dur_b) / (1.0 + target_overlap_ratio)
        offset_sec = float(np.clip(raw_offset, 0.0, dur_a))
        # Evita truncar o locutor B quando offset_sec é sorteado automaticamente
        if offset_sec + dur_b > cfg.max_duration_sec:
            offset_sec = max(0.0, cfg.max_duration_sec - dur_b)

    offset_samples = int(round(offset_sec * cfg.sample_rate))
    total_samples = min(max_allowed_samples, max(len_a, offset_samples + len_b))
    total_duration = round(total_samples / cfg.sample_rate, 4)

    # Alinha A e B no mesmo eixo temporal antes da soma
    track_a = np.zeros(total_samples, dtype=np.float32)
    track_b = np.zeros(total_samples, dtype=np.float32)
    end_a = min(total_samples, len_a)
    track_a[:end_a] = audio_a[:end_a]

    if offset_samples < total_samples:
        avail_b = min(len_b, total_samples - offset_samples)
        track_b[offset_samples : offset_samples + avail_b] = audio_b[:avail_b]

    # Extrai máscaras de 80ms nas faixas isoladas antes de somar (Ground Truth perfeito)
    mask_a = compute_vad_frame_mask(track_a, rms_threshold=cfg.vad_rms_threshold)
    mask_b = compute_vad_frame_mask(track_b, rms_threshold=cfg.vad_rms_threshold)

    # Calcula SIR (Signal-to-Interference Ratio) e aplica ganho relativo em B
    if sir_db is None:
        sir_db = float(rng.uniform(cfg.min_sir_db, cfg.max_sir_db))
    gain_b = float(10.0 ** (-sir_db / 20.0))

    mixed = track_a + (gain_b * track_b)

    # Ruído gaussiano opcional de fundo (Spec 02 Seção 3.4)
    if cfg.add_gaussian_noise and total_samples > 0:
        snr_db = float(rng.uniform(cfg.noise_snr_db_range[0], cfg.noise_snr_db_range[1]))
        sig_rms = float(np.sqrt(np.mean(np.square(mixed, dtype=np.float64))))
        if sig_rms > 0.0:
            noise_rms = sig_rms * float(10.0 ** (snr_db / 20.0))
            noise = rng.normal(0.0, noise_rms, size=total_samples).astype(np.float32)
            mixed = mixed + noise

    mixed = prevent_clipping(mixed, max_peak=cfg.max_peak_amplitude)

    if write_wav:
        out_path = Path(output_wav_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), mixed, cfg.sample_rate, subtype="PCM_16")

    # Calcula taxa real de sobreposição a partir das máscaras binárias
    active_any = sum(1 for a, b in zip(mask_a, mask_b) if (a == 1 or b == 1))
    active_both = sum(1 for a, b in zip(mask_a, mask_b) if (a == 1 and b == 1))
    actual_overlap = round(active_both / max(1, active_any), 4)

    recorded_filepath = manifest_audio_filepath or str(output_wav_path)

    # Espelhamento A/B (Spec 02): 2 amostras de treino pelo preço de 1 arquivo .wav
    entry_target_a = MultitalkerManifestEntry(
        audio_filepath=recorded_filepath,
        duration=total_duration,
        text=text_a,
        speaker_id=speaker_id_a,
        spk_targets=mask_a,
        bg_spk_targets=mask_b,
        overlap_ratio=actual_overlap,
        sir_db=round(sir_db, 2),
    )
    entry_target_b = MultitalkerManifestEntry(
        audio_filepath=recorded_filepath,
        duration=total_duration,
        text=text_b,
        speaker_id=speaker_id_b,
        spk_targets=mask_b,
        bg_spk_targets=mask_a,
        overlap_ratio=actual_overlap,
        sir_db=round(-sir_db, 2),
    )

    return mixed, [entry_target_a, entry_target_b]


def _inspect_existing_mixed_manifest(manifest_path: Path) -> Tuple[int, int]:
    """
    Para retomada idempotente (resume=True), retorna (num_mixes_concluidos, num_clean_concluidos)
    validando todas as linhas já gravadas no manifesto de saída.
    """
    if not manifest_path.exists():
        return 0, 0
    overlapped_entries = 0
    clean_entries = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            MultitalkerManifestEntry.model_validate(rec)
            if Path(rec["audio_filepath"]).name.startswith("mix_"):
                overlapped_entries += 1
            else:
                clean_entries += 1
    return overlapped_entries // 2, clean_entries


def generate_mixed_dataset_from_manifest(
    clean_manifest_path: str,
    output_dir: str,
    output_manifest_path: str,
    num_mixtures: int,
    seed: int = 42,
    clean_ratio: float = 0.0,
    target_root_dir: Optional[str] = None,
    resume: bool = True,
) -> int:
    """
    Lê um manifesto single-speaker em PT-BR e gera misturas sintéticas validadas (Spec 02).
    Suporta retomada idempotente (resume=True) e proporção híbrida 70/30 (clean_ratio=0.30).
    """
    if not (0.0 <= clean_ratio < 1.0):
        raise ValueError("clean_ratio deve estar no intervalo [0.0, 1.0).")

    rng = np.random.default_rng(seed)
    cfg = MixerConfig(clean_ratio=clean_ratio)

    with open(clean_manifest_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if len(records) < 2:
        raise ValueError("São necessários pelo menos 2 registros no manifesto limpo para criar misturas.")

    out_wav_dir = Path(output_dir) / "wavs"
    out_wav_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = Path(output_manifest_path)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    existing_mixes, existing_clean = (
        _inspect_existing_mixed_manifest(out_manifest) if resume else (0, 0)
    )
    total_overlapped_target = num_mixtures * 2
    total_clean_target = (
        int(round((total_overlapped_target * clean_ratio) / (1.0 - clean_ratio)))
        if clean_ratio > 0.0
        else 0
    )

    # Avança o RNG para manter determinismo caso esteja retomando de existing_mixes > 0
    for _ in range(existing_mixes):
        _sample_distinct_speaker_pair(records, rng)
        rng.uniform(cfg.min_overlap_ratio, cfg.max_overlap_ratio)
        rng.uniform(cfg.min_sir_db, cfg.max_sir_db)

    written_entries = existing_mixes * 2 + existing_clean
    mode = "a" if (resume and out_manifest.exists() and written_entries > 0) else "w"

    with open(out_manifest, mode, encoding="utf-8") as out_f:
        for mix_idx in range(existing_mixes, num_mixtures):
            rec_a, rec_b = _sample_distinct_speaker_pair(records, rng)

            path_a = _resolve_local_audio_path(rec_a["audio_filepath"], clean_manifest_path)
            path_b = _resolve_local_audio_path(rec_b["audio_filepath"], clean_manifest_path)

            audio_a, sr_a = sf.read(path_a, dtype="float32")
            audio_b, sr_b = sf.read(path_b, dtype="float32")
            if sr_a != REQUIRED_SAMPLE_RATE or sr_b != REQUIRED_SAMPLE_RATE:
                raise ValueError("Todos os áudios de entrada devem estar em 16.000 Hz.")

            wav_name = f"mix_{mix_idx:06d}.wav"
            mix_wav_path = str(out_wav_dir / wav_name)
            manifest_wav_path = (
                str(Path(target_root_dir) / "wavs" / wav_name)
                if target_root_dir
                else mix_wav_path
            )
            _, entries = mix_two_speakers(
                audio_a=audio_a,
                text_a=rec_a["text"],
                audio_b=audio_b,
                text_b=rec_b["text"],
                output_wav_path=mix_wav_path,
                config=cfg,
                rng=rng,
                write_wav=True,
                speaker_id_a=rec_a.get("speaker_id"),
                speaker_id_b=rec_b.get("speaker_id"),
                manifest_audio_filepath=manifest_wav_path,
            )
            for entry in entries:
                out_f.write(entry.model_dump_json() + "\n")
                written_entries += 1

        # Se clean_ratio > 0 (Spec 04: 70% sobreposto + 30% limpo), inclui amostras single-speaker
        for _ in range(existing_clean, total_clean_target):
            rec = records[int(rng.integers(0, len(records)))]
            clean_audio_path = (
                str(Path(target_root_dir) / "wavs" / Path(rec["audio_filepath"]).name)
                if target_root_dir
                else rec["audio_filepath"]
            )
            clean_entry = SingleSpeakerManifestEntry.from_clean_audio(
                audio_filepath=clean_audio_path,
                duration=float(rec["duration"]),
                text=rec["text"],
                speaker_id=rec.get("speaker_id"),
            )
            out_f.write(clean_entry.model_dump_json() + "\n")
            written_entries += 1

    return written_entries


def _supervision_from_mask(entry: MultitalkerManifestEntry, speaker: str) -> SpeakerSupervision:
    """Intervalo do primeiro ao último frame ativo da máscara do alvo; sem frame ativo, [0, D] (Spec 02 §3.7)."""
    active = [
        idx for idx, val in enumerate(entry.spk_targets)
        if val == 1 and idx * FRAME_DURATION_SEC < entry.duration
    ]
    if not active:
        return SpeakerSupervision(speaker=speaker, start=0.0, duration=entry.duration, text=entry.text)
    start = round(active[0] * FRAME_DURATION_SEC, 4)
    end = min(round((active[-1] + 1) * FRAME_DURATION_SEC, 4), entry.duration)
    return SpeakerSupervision(speaker=speaker, start=start, duration=round(end - start, 4), text=entry.text)


def manifest_entries_to_cuts(entries: List[MultitalkerManifestEntry]) -> List[MultitalkerCut]:
    """
    Agrupa cada par espelhado (linhas mix_* consecutivas) num cut com 2 supervisions e cada linha
    limpa num cut com 1 supervision, como exige o LhotseSpeechToTextSpkBpeDataset (Spec 02 §3.7).
    """
    cuts: List[MultitalkerCut] = []
    idx = 0
    while idx < len(entries):
        entry = entries[idx]
        cut_id = f"{Path(entry.audio_filepath).stem}-{idx:07d}"
        if not Path(entry.audio_filepath).name.startswith("mix_"):
            supervisions = [
                SpeakerSupervision(
                    speaker=entry.speaker_id or "spk_a", start=0.0, duration=entry.duration, text=entry.text
                )
            ]
            idx += 1
        else:
            pair = entries[idx + 1] if idx + 1 < len(entries) else None
            if (
                pair is None
                or pair.audio_filepath != entry.audio_filepath
                or pair.spk_targets != entry.bg_spk_targets
                or pair.bg_spk_targets != entry.spk_targets
            ):
                raise ValueError(
                    f"Linha {idx + 1} ({entry.audio_filepath}) sem par espelhado válido na linha seguinte (Spec 02 §3.7)."
                )
            spk_a, spk_b = entry.speaker_id, pair.speaker_id
            if not spk_a or not spk_b or spk_a == spk_b:
                spk_a, spk_b = "spk_a", "spk_b"
            supervisions = [_supervision_from_mask(entry, spk_a), _supervision_from_mask(pair, spk_b)]
            idx += 2
        cuts.append(
            MultitalkerCut(
                id=cut_id, audio_filepath=entry.audio_filepath, duration=entry.duration, supervisions=supervisions
            )
        )
    return cuts


def export_lhotse_cuts(manifest_path: str, cuts_path: Optional[str] = None) -> Tuple[str, int]:
    """Converte o manifesto espelhado num CutSet JSONL para o cuts_path da Fase 2 (Spec 02 §3.7, Spec 04 §8.2)."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = [MultitalkerManifestEntry.model_validate_json(line) for line in f if line.strip()]
    cuts = manifest_entries_to_cuts(entries)

    out_path = Path(cuts_path) if cuts_path else Path(manifest_path).with_name(f"{Path(manifest_path).stem}_cuts.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_f:
        for cut in cuts:
            out_f.write(json.dumps(cut.to_lhotse_dict(), ensure_ascii=False) + "\n")
    return str(out_path), len(cuts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulador de sobreposição acústica PT-BR para multitalker-parakeet (Spec 02)."
    )
    parser.add_argument("--clean_manifest", required=True, help="Caminho do manifest.json single-speaker.")
    parser.add_argument("--output_dir", required=True, help="Diretório onde os WAVs misturados serão salvos.")
    parser.add_argument("--output_manifest", required=True, help="Caminho do manifest.json multitalker de saída.")
    parser.add_argument("--num_mixtures", type=int, default=100, help="Quantidade de pares misturados a gerar.")
    parser.add_argument(
        "--clean_ratio",
        type=float,
        default=0.30,
        help="Proporção de fala limpa single-speaker no manifesto final (padrão 0.30 conforme Spec 04).",
    )
    parser.add_argument("--target_root_dir", default=None, help="Prefixo opcional para audio_filepath no manifesto.")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória.")
    parser.add_argument("--no_resume", action="store_true", help="Desativa retomada incremental e recria as misturas do zero.")
    parser.add_argument("--update_state", action="store_true", help="Atualiza state/pipeline_state.json e USER.md.")
    args = parser.parse_args()

    step_id = "step_03_mixer_multitalker"
    if args.update_state:
        from src.state import start_step

        start_step(
            step_id=step_id,
            command=f"python -m src.mixer --clean_manifest {args.clean_manifest} --output_manifest {args.output_manifest}",
            notes=f"Gerando {args.num_mixtures} misturas sobrepostas em {args.output_manifest}.",
        )

    try:
        total = generate_mixed_dataset_from_manifest(
            clean_manifest_path=args.clean_manifest,
            output_dir=args.output_dir,
            output_manifest_path=args.output_manifest,
            num_mixtures=args.num_mixtures,
            seed=args.seed,
            clean_ratio=args.clean_ratio,
            target_root_dir=args.target_root_dir,
            resume=not args.no_resume,
        )
        print(f"[Spec 02] Mixagem concluída: {total} entradas geradas em {args.output_manifest}")

        if args.update_state:
            from src.state import complete_step, load_pipeline_state, update_handover

            st = load_pipeline_state()
            step_03 = next(s for s in st.steps if s.step_id == step_id)
            artifacts = dict(step_03.artifacts)
            out_name = Path(args.output_manifest).name
            key = "multitalker_val_manifest" if "val" in out_name else "multitalker_train_manifest"
            artifacts[key] = args.output_manifest

            has_train = "multitalker_train_manifest" in artifacts
            has_val = "multitalker_val_manifest" in artifacts

            if has_train and not has_val:
                update_handover(
                    step_id=step_id,
                    next_action="Gerar o manifesto sobreposto de validação (multitalker_val_manifest.json).",
                    next_command=(
                        ".venv/bin/python -m src.mixer --clean_manifest data/clean_train/clean_val_manifest.json "
                        "--output_dir data/multitalker_val "
                        "--output_manifest data/multitalker_val/multitalker_val_manifest.json "
                        "--num_mixtures 200 --clean_ratio 0.0 --update_state"
                    ),
                    last_summary=f"Manifesto multitalker de treino gerado ({total} entradas). Falta o manifesto de validação.",
                    artifacts=artifacts,
                    metrics={"train_entries": total, "clean_ratio": args.clean_ratio},
                )
            else:
                complete_step(
                    step_id=step_id,
                    next_step_id="step_04_pack_and_upload_drive",
                    next_action="Reescrever caminhos para /content/local_dataset, empacotar em .tar.gz e enviar ao Google Drive.",
                    next_command=(
                        ".venv/bin/python -m src.data_prep --prepare_colab_bundle "
                        "--output_dir data --pack_tar_gz data/ptbr_dataset_16khz.tar.gz --update_state"
                    ),
                    last_summary=f"Manifestos multitalker (treino e validação) concluídos ({args.output_manifest}).",
                    artifacts=artifacts,
                    metrics={"last_manifest_entries": total, "clean_ratio": args.clean_ratio},
                )
    except Exception as exc:
        if args.update_state:
            from src.state import fail_step

            fail_step(step_id=step_id, error_message=str(exc))
        raise


if __name__ == "__main__":
    main()
