"""Módulo de Avaliação Final (WER Limpo vs. Sobreposto) e Exportação .nemo (Spec 04 e Spec 05 - Step 07)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

from src.mixer import manifest_entries_to_cuts
from src.schemas import EvaluationReport, MultitalkerCut, MultitalkerManifestEntry
from src.state import utc_now_iso
from src.trainer import preflight_validate_manifest


def compute_word_error_rate(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    """
    Calcula o Word Error Rate (WER) exato via distância de edição de Levenshtein em nível de palavras.
    Funciona em CPU pura (para testes de contrato) e após inferência em GPU.
    """
    ref_list = list(references)
    hyp_list = list(hypotheses)
    if len(ref_list) != len(hyp_list):
        raise ValueError(
            f"O número de referências ({len(ref_list)}) deve ser igual ao de hipóteses ({len(hyp_list)})."
        )
    if not ref_list:
        raise ValueError("As listas de referências e hipóteses não podem ser vazias.")

    total_words = 0
    total_edits = 0

    for ref, hyp in zip(ref_list, hyp_list):
        r_words = ref.strip().lower().split()
        h_words = hyp.strip().lower().split()
        total_words += len(r_words)

        # Programação dinâmica O(N*M) para distância de edição de palavras
        n, m = len(r_words), len(h_words)
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, m + 1):
                temp = dp[j]
                cost = 0 if r_words[i - 1] == h_words[j - 1] else 1
                dp[j] = min(
                    dp[j] + 1,       # deleção
                    dp[j - 1] + 1,   # inserção
                    prev + cost,     # substituição
                )
                prev = temp
        total_edits += dp[m]

    if total_words == 0:
        return 0.0 if total_edits == 0 else 1.0
    return round(min(2.0, total_edits / total_words), 4)


def normalize_for_wer(text: str) -> str:
    """Minúsculas e pontuação trocada por espaço: as referências do CORAA-MUPE têm caixa e pontuação (Spec 04 §6)."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def unique_eval_cuts(entries: List[MultitalkerManifestEntry]) -> List[MultitalkerCut]:
    """
    Cuts da Spec 02 §3.7 sem a 2ª cópia de cada par espelhado: cada mistura é decodificada uma vez por
    locutor e cada linha limpa uma vez (Spec 04 §6).
    """
    unique: List[MultitalkerCut] = []
    for cut in manifest_entries_to_cuts(entries):
        prev = unique[-1] if unique else None
        if (
            prev is not None
            and len(cut.supervisions) > 1
            and prev.audio_filepath == cut.audio_filepath
            and prev.supervisions == cut.supervisions
        ):
            continue
        unique.append(cut)
    return unique


def decode_per_target_speaker(
    asr_model: Any,
    cuts: List[MultitalkerCut],
    batch_size: int = 16,
) -> Tuple[List[str], List[str]]:
    """
    Decodifica cada locutor de cada cut condicionado pelas máscaras do speaker_to_target() do NeMo, o mesmo
    caminho do treino; o transcribe() ignora os alvos (inference_mode). Retorna (referências, hipóteses).
    """
    import torch
    from lhotse import CutSet
    from lhotse.dataset.collation import collate_vectors
    from nemo.collections.asr.parts.utils.asr_multispeaker_utils import speaker_to_target

    device = next(asr_model.parameters()).device
    refs: List[str] = []
    hyps: List[str] = []
    batch: List[Tuple[Any, Any, Any, str]] = []

    def flush() -> None:
        audios, spks, bgs, texts = zip(*batch)
        signal = collate_vectors(audios, padding_value=0.0).to(device)
        lengths = torch.tensor([a.shape[0] for a in audios], dtype=torch.long, device=device)
        asr_model.set_speaker_targets(
            collate_vectors(spks, padding_value=0.0).to(device), collate_vectors(bgs, padding_value=0.0).to(device)
        )
        with torch.inference_mode():
            encoded, encoded_len = asr_model.forward(input_signal=signal, input_signal_length=lengths)
            out = asr_model.decoding.rnnt_decoder_predictions_tensor(encoder_output=encoded, encoded_lengths=encoded_len)
        if isinstance(out, tuple):
            out = out[0]
        hyps.extend(h.text if hasattr(h, "text") else str(h) for h in out)
        refs.extend(texts)
        batch.clear()

    for cut in CutSet.from_dicts(c.to_lhotse_dict() for c in cuts):
        audio = torch.as_tensor(cut.load_audio()[0], dtype=torch.float32)
        masks, texts = speaker_to_target(
            cut, num_speakers=4, num_sample_per_mel_frame=160, num_mel_frame_per_asr_frame=8, return_text=True
        )
        masks = masks.transpose(0, 1)[: len(texts)].float()
        for idx, text in enumerate(texts):
            others = [j for j in range(len(texts)) if j != idx]
            bg = (masks[others].sum(dim=0) > 0).float() if others else torch.zeros_like(masks[idx])
            batch.append((audio, masks[idx], bg, text))
            if len(batch) >= batch_size:
                flush()
    if batch:
        flush()
    asr_model.clear_speaker_targets()
    return refs, hyps


def _load_manifest_entries(manifest_path: str, max_lines: Optional[int] = None) -> List[MultitalkerManifestEntry]:
    preflight_validate_manifest(manifest_path, max_lines_to_check=0, check_audio_files_exist=False)
    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = [MultitalkerManifestEntry.model_validate_json(line) for line in f if line.strip()]
    if max_lines and max_lines < len(entries):
        entries = entries[:max_lines]
        last = entries[-1]
        # Não corta um par espelhado ao meio (Spec 02 §3.7)
        if Path(last.audio_filepath).name.startswith("mix_") and (
            len(entries) < 2 or entries[-2].audio_filepath != last.audio_filepath
        ):
            entries = entries[:-1]
    return entries


def evaluate_nemo_checkpoint(
    model_path: str,
    clean_manifest_path: str,
    multitalker_manifest_path: str,
    output_report_path: str,
    batch_size: int = 16,
    export_nemo_path: Optional[str] = None,
    update_state: bool = False,
    max_lines: Optional[int] = None,
) -> EvaluationReport:
    """
    Carrega o modelo .nemo final da Fase 2, avalia o WER em fala limpa e em fala sobreposta
    (multitalker) e salva o EvaluationReport validado (Spec 04 e Spec 05 - Step 07).
    """
    if update_state:
        from src.state import start_step

        start_step(
            step_id="step_07_eval_and_export",
            command=(
                f"python -m src.evaluator --model_path {model_path} "
                f"--clean_manifest {clean_manifest_path} "
                f"--multitalker_manifest {multitalker_manifest_path} "
                f"--output_report {output_report_path} --update_state"
            ),
            notes="Executando avaliação de WER limpo e sobreposto (Step 07).",
        )

    try:
        nemo_file = Path(model_path)
        if not nemo_file.exists():
            raise FileNotFoundError(f"Arquivo de modelo .nemo não encontrado: {nemo_file}")

        clean_records = _load_manifest_entries(clean_manifest_path, max_lines)
        multi_records = _load_manifest_entries(multitalker_manifest_path, max_lines)

        import nemo.collections.asr as nemo_asr

        asr_model = nemo_asr.models.ASRModel.restore_from(str(nemo_file))
        asr_model.eval()

        clean_refs, clean_hyps_str = decode_per_target_speaker(asr_model, unique_eval_cuts(clean_records), batch_size)
        multi_refs, multi_hyps_str = decode_per_target_speaker(asr_model, unique_eval_cuts(multi_records), batch_size)

        clean_wer = compute_word_error_rate(clean_refs, clean_hyps_str)
        multi_wer = compute_word_error_rate(multi_refs, multi_hyps_str)
        clean_wer_norm = compute_word_error_rate(
            [normalize_for_wer(t) for t in clean_refs], [normalize_for_wer(t) for t in clean_hyps_str]
        )
        multi_wer_norm = compute_word_error_rate(
            [normalize_for_wer(t) for t in multi_refs], [normalize_for_wer(t) for t in multi_hyps_str]
        )
        for ref, hyp in list(zip(multi_refs, multi_hyps_str))[:6]:
            print(f"[Step 07] REF: {ref}\n[Step 07] HYP: {hyp}")

        final_model_path = str(nemo_file)
        if export_nemo_path:
            out_nemo = Path(export_nemo_path)
            out_nemo.parent.mkdir(parents=True, exist_ok=True)
            asr_model.save_to(str(out_nemo))
            final_model_path = str(out_nemo)

        report = EvaluationReport(
            model_path=final_model_path,
            clean_manifest_path=clean_manifest_path,
            multitalker_manifest_path=multitalker_manifest_path,
            clean_samples_evaluated=len(clean_refs),
            multitalker_samples_evaluated=len(multi_refs),
            clean_wer=clean_wer,
            multitalker_wer=multi_wer,
            clean_wer_normalized=clean_wer_norm,
            multitalker_wer_normalized=multi_wer_norm,
            evaluated_at=utc_now_iso(),
        )

        out_rep = Path(output_report_path)
        out_rep.parent.mkdir(parents=True, exist_ok=True)
        out_rep.write_text(
            json.dumps(report.model_dump(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        if update_state:
            from src.state import complete_step

            complete_step(
                step_id="step_07_eval_and_export",
                next_step_id="step_07_eval_and_export",
                next_action="Pipeline finalizado. Verifique o relatório de WER e o modelo .nemo exportado.",
                next_command=".venv/bin/python -m src.state status",
                last_summary=(
                    f"Avaliação final concluída: clean_wer={clean_wer:.4f}, "
                    f"multitalker_wer={multi_wer:.4f}. Modelo final em {final_model_path}."
                ),
                artifacts={
                    "final_nemo_model": final_model_path,
                    "eval_report": str(out_rep),
                },
                metrics={
                    "clean_wer": clean_wer,
                    "multitalker_wer": multi_wer,
                    "clean_wer_normalized": clean_wer_norm,
                    "multitalker_wer_normalized": multi_wer_norm,
                    "clean_samples": len(clean_refs),
                    "multitalker_samples": len(multi_refs),
                },
            )

        return report
    except Exception as exc:
        if update_state:
            from src.state import fail_step

            fail_step(step_id="step_07_eval_and_export", error_message=str(exc))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avaliador de WER (Limpo e Multitalker) e Exportador .nemo (Spec 04 / Step 07)."
    )
    parser.add_argument("--model_path", required=True, help="Caminho para o checkpoint .nemo da Fase 2.")
    parser.add_argument("--clean_manifest", required=True, help="Manifesto de teste single-speaker limpo.")
    parser.add_argument("--multitalker_manifest", required=True, help="Manifesto de teste sobreposto.")
    parser.add_argument("--output_report", required=True, help="Caminho de saída para o JSON EvaluationReport.")
    parser.add_argument("--export_nemo_path", default=None, help="Caminho opcional para exportar o .nemo final.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_lines", type=int, default=None, help="Avalia só as primeiras N linhas de cada manifesto.")
    parser.add_argument("--update_state", action="store_true", help="Atualiza state/pipeline_state.json e USER.md.")
    args = parser.parse_args()

    report = evaluate_nemo_checkpoint(
        model_path=args.model_path,
        clean_manifest_path=args.clean_manifest,
        multitalker_manifest_path=args.multitalker_manifest,
        output_report_path=args.output_report,
        batch_size=args.batch_size,
        export_nemo_path=args.export_nemo_path,
        update_state=args.update_state,
        max_lines=args.max_lines,
    )
    print(
        f"[Spec 04 / Step 07] Avaliação concluída: clean_wer={report.clean_wer} ({report.clean_wer_normalized} normalizado), "
        f"multitalker_wer={report.multitalker_wer} ({report.multitalker_wer_normalized} normalizado)"
    )


if __name__ == "__main__":
    main()
