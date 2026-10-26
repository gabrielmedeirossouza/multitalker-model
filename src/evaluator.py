"""Módulo de Avaliação Final (WER Limpo vs. Sobreposto) e Exportação .nemo (Spec 04 e Spec 05 - Step 07)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.mixer import manifest_entries_to_cuts
from src.schemas import FRAME_DURATION_SEC, EvaluationReport, MultitalkerCut, MultitalkerManifestEntry
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


def _raw_and_normalized_wer(refs: List[str], hyps: List[str]) -> Tuple[float, float]:
    return compute_word_error_rate(refs, hyps), compute_word_error_rate(
        [normalize_for_wer(t) for t in refs], [normalize_for_wer(t) for t in hyps]
    )


def latency_ms(att_context_size: Sequence[int]) -> int:
    """Latência do modo cache-aware: (contexto à direita + 1) frames de 80 ms (Spec 04 §6)."""
    return int(round((att_context_size[1] + 1) * FRAME_DURATION_SEC * 1000))


def resolve_eval_contexts(encoder: Any, requested: Optional[Sequence[Sequence[int]]] = None) -> List[List[int]]:
    """
    Contextos [esquerda, direita] a avaliar: os pedidos ou todos os do encoder, do maior lookahead ao menor.
    Fora do treino, o encoder usaria só o 1º de att_context_size_all, que depende da ordem do cfg (Spec 04 §6, §8.7).
    """
    if requested:
        return [[int(v) for v in ctx] for ctx in requested]
    supported = getattr(encoder, "att_context_size_all", None) or [encoder.att_context_size]
    return sorted(([int(v) for v in ctx] for ctx in supported), key=lambda ctx: -ctx[1])


def eval_state_metrics(reports: Sequence[EvaluationReport], prefix: str = "") -> Dict[str, float | int]:
    """Métricas do state por latência; o prefixo (ex.: test_) preserva as da validação (Spec 04 §6)."""
    metrics: Dict[str, float | int] = {}
    for rep in reports:
        lat = latency_ms(rep.att_context_size)
        values = {
            "clean_wer": rep.clean_wer,
            "clean_wer_normalized": rep.clean_wer_normalized,
            "multitalker_wer": rep.multitalker_wer,
            "multitalker_wer_normalized": rep.multitalker_wer_normalized,
        }
        metrics.update({f"{prefix}{key}_{lat}ms": val for key, val in values.items() if val is not None})
    if reports:
        metrics[f"{prefix}clean_samples"] = reports[0].clean_samples_evaluated
        metrics[f"{prefix}multitalker_samples"] = reports[0].multitalker_samples_evaluated
    return metrics


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
    att_context_sizes: Optional[Sequence[Sequence[int]]] = None,
    metrics_prefix: str = "",
) -> List[EvaluationReport]:
    """
    Carrega o modelo .nemo final da Fase 2, avalia o WER em fala limpa e em fala sobreposta (multitalker)
    em cada contexto de atenção (latência) e salva a lista de EvaluationReport validados (Spec 04 §6 / Step 07).
    """
    contexts_arg = "".join(f" --att_context_size {ctx[0]} {ctx[1]}" for ctx in att_context_sizes or [])
    if update_state:
        from src.state import start_step

        start_step(
            step_id="step_07_eval_and_export",
            command=(
                f"python -m src.evaluator --model_path {model_path} "
                f"--clean_manifest {clean_manifest_path} "
                f"--multitalker_manifest {multitalker_manifest_path} "
                f"--output_report {output_report_path}{contexts_arg} "
                f"--metrics_prefix '{metrics_prefix}' --update_state"
            ),
            notes="Executando avaliação de WER limpo e sobreposto por latência (Step 07).",
        )

    try:
        nemo_file = Path(model_path)
        if not nemo_file.exists():
            raise FileNotFoundError(f"Arquivo de modelo .nemo não encontrado: {nemo_file}")

        clean_records = _load_manifest_entries(clean_manifest_path, max_lines)
        multi_records = _load_manifest_entries(multitalker_manifest_path, max_lines)
        # Lotes de durações próximas: menos padding nas 4 passadas por latência (o WER não depende da ordem)
        clean_cuts = sorted(unique_eval_cuts(clean_records), key=lambda cut: cut.duration)
        multi_cuts = sorted(unique_eval_cuts(multi_records), key=lambda cut: cut.duration)

        import nemo.collections.asr as nemo_asr

        asr_model = nemo_asr.models.ASRModel.restore_from(str(nemo_file))
        asr_model.eval()
        print(
            f"[Step 07] Contextos do encoder: {asr_model.encoder.att_context_size_all} "
            f"(padrão ao restaurar: {asr_model.encoder.att_context_size})"
        )

        reports: List[EvaluationReport] = []
        for ctx in resolve_eval_contexts(asr_model.encoder, att_context_sizes):
            asr_model.encoder.set_default_att_context_size(ctx)
            clean_refs, clean_hyps = decode_per_target_speaker(asr_model, clean_cuts, batch_size)
            multi_refs, multi_hyps = decode_per_target_speaker(asr_model, multi_cuts, batch_size)
            clean_wer, clean_wer_norm = _raw_and_normalized_wer(clean_refs, clean_hyps)
            multi_wer, multi_wer_norm = _raw_and_normalized_wer(multi_refs, multi_hyps)
            reports.append(
                EvaluationReport(
                    model_path=str(nemo_file),
                    clean_manifest_path=clean_manifest_path,
                    multitalker_manifest_path=multitalker_manifest_path,
                    clean_samples_evaluated=len(clean_refs),
                    multitalker_samples_evaluated=len(multi_refs),
                    clean_wer=clean_wer,
                    multitalker_wer=multi_wer,
                    clean_wer_normalized=clean_wer_norm,
                    multitalker_wer_normalized=multi_wer_norm,
                    att_context_size=ctx,
                    evaluated_at=utc_now_iso(),
                )
            )
            print(
                f"[Step 07] att_context_size={ctx} ({latency_ms(ctx)} ms): clean_wer={clean_wer} ({clean_wer_norm} "
                f"normalizado), multitalker_wer={multi_wer} ({multi_wer_norm} normalizado)"
            )
            for ref, hyp in list(zip(multi_refs, multi_hyps))[:3]:
                print(f"[Step 07] REF: {ref}\n[Step 07] HYP: {hyp}")

        final_model_path = str(nemo_file)
        if export_nemo_path:
            out_nemo = Path(export_nemo_path)
            out_nemo.parent.mkdir(parents=True, exist_ok=True)
            asr_model.save_to(str(out_nemo))
            final_model_path = str(out_nemo)
            reports = [rep.model_copy(update={"model_path": final_model_path}) for rep in reports]

        out_rep = Path(output_report_path)
        out_rep.parent.mkdir(parents=True, exist_ok=True)
        out_rep.write_text(
            json.dumps([rep.model_dump() for rep in reports], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        if update_state:
            from src.state import complete_step

            per_latency = "; ".join(
                f"{latency_ms(rep.att_context_size)} ms: limpo {rep.clean_wer_normalized:.4f}, "
                f"multitalker {rep.multitalker_wer_normalized:.4f}"
                for rep in reports
            )
            complete_step(
                step_id="step_07_eval_and_export",
                next_step_id="step_07_eval_and_export",
                next_action="Pipeline finalizado. Verifique o relatório de WER e o modelo .nemo exportado.",
                next_command=".venv/bin/python -m src.state status",
                last_summary=(
                    f"Avaliação em {clean_manifest_path} concluída, WER normalizado por latência: {per_latency}. "
                    f"Modelo final em {final_model_path}."
                ),
                artifacts={
                    "final_nemo_model": final_model_path,
                    f"{metrics_prefix}eval_report": str(out_rep),
                },
                metrics=eval_state_metrics(reports, metrics_prefix),
            )

        return reports
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
    parser.add_argument("--output_report", required=True, help="Caminho de saída da lista JSON de EvaluationReport.")
    parser.add_argument("--export_nemo_path", default=None, help="Caminho opcional para exportar o .nemo final.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_lines", type=int, default=None, help="Avalia só as primeiras N linhas de cada manifesto.")
    parser.add_argument(
        "--att_context_size",
        type=int,
        nargs=2,
        action="append",
        metavar=("L", "R"),
        default=None,
        help="Contexto de atenção [esquerda, direita] a avaliar (repetível); padrão: todos os do encoder (Spec 04 §6).",
    )
    parser.add_argument("--metrics_prefix", default="", help="Prefixo das métricas no state (ex.: test_).")
    parser.add_argument("--update_state", action="store_true", help="Atualiza state/pipeline_state.json e USER.md.")
    args = parser.parse_args()

    reports = evaluate_nemo_checkpoint(
        model_path=args.model_path,
        clean_manifest_path=args.clean_manifest,
        multitalker_manifest_path=args.multitalker_manifest,
        output_report_path=args.output_report,
        batch_size=args.batch_size,
        export_nemo_path=args.export_nemo_path,
        update_state=args.update_state,
        max_lines=args.max_lines,
        att_context_sizes=args.att_context_size,
        metrics_prefix=args.metrics_prefix,
    )
    for rep in reports:
        print(
            f"[Spec 04 / Step 07] {latency_ms(rep.att_context_size)} ms: clean_wer={rep.clean_wer} "
            f"({rep.clean_wer_normalized} normalizado), multitalker_wer={rep.multitalker_wer} "
            f"({rep.multitalker_wer_normalized} normalizado)"
        )


if __name__ == "__main__":
    main()
