"""Orquestrador de treinamento em 2 fases e avaliação no NVIDIA NeMo para GPU A100 no Google Colab (Spec 04)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.schemas import MultitalkerManifestEntry, TrainingStageConfig


DEFAULT_CONFIG_PATHS = {
    "warmup": Path(__file__).resolve().parent.parent / "configs" / "a100_warmup.yaml",
    "multitalker": Path(__file__).resolve().parent.parent / "configs" / "a100_multitalker.yaml",
}


def load_and_validate_stage_config(
    stage: str,
    config_path: Optional[str] = None,
) -> Tuple[Dict[str, Any], TrainingStageConfig]:
    """Carrega o arquivo YAML da fase e valida as invariantes da Spec 04 via Pydantic."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATHS[stage]
    with open(path, "r", encoding="utf-8") as f:
        raw_cfg: Dict[str, Any] = yaml.safe_load(f)

    enc_cfg = raw_cfg.get("model", {}).get("encoder", {})
    spk_kernel_cfg = enc_cfg.get("speaker_kernel", {})
    train_ds_cfg = raw_cfg.get("model", {}).get("train_ds", {})

    contract = TrainingStageConfig(
        stage=raw_cfg["stage"],
        freeze_encoder=bool(raw_cfg["freeze_encoder"]),
        precision=raw_cfg["trainer"]["precision"],
        bucketing_strategy=train_ds_cfg["bucketing_strategy"],
        fused_batch_size=int(raw_cfg["model"]["joint"]["fused_batch_size"]),
        resume_if_exists=bool(raw_cfg["exp_manager"]["resume_if_exists"]),
        vocab_size=int(raw_cfg["model"]["tokenizer"]["vocab_size"]),
        return_speaker_targets=bool(train_ds_cfg.get("return_speaker_targets", False)),
        speaker_kernel_enabled=bool(spk_kernel_cfg.get("enabled", True)),
        att_context_size=enc_cfg.get("att_context_size"),
    )
    return raw_cfg, contract


def resolve_nemo_checkpoint_path(configured_path: str) -> Path:
    """
    Localiza o arquivo .nemo gerado pelo exp_manager do NeMo.
    Caso o caminho estático não exista (porque o exp_manager cria subpastas de timestamp
    YYYY-MM-DD_HH-MM-SS/checkpoints/), busca recursivamente o .nemo mais recente no diretório pai.
    """
    direct = Path(configured_path)
    if direct.exists():
        return direct

    # Sobe até encontrar um diretório existente (ex.: .../exp_v1_warmup) e procura *.nemo
    parent = direct.parent
    for _ in range(4):
        if parent.exists():
            candidates = sorted(parent.rglob("*.nemo"), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                return candidates[0]
        parent = parent.parent

    raise FileNotFoundError(
        f"Modelo base (.nemo) não encontrado em '{configured_path}' nem nas subpastas do exp_manager. "
        "Conclua o estágio anterior primeiro ou informe --init_model."
    )


def preflight_validate_manifest(
    manifest_path: str,
    max_lines_to_check: int = 0,
    check_audio_files_exist: bool = False,
) -> int:
    """
    Gatekeeper SDD antes de alocar GPU: valida que as linhas do manifesto obedecem
    ao contrato MultitalkerManifestEntry (Spec 01 / Spec 02) e, quando check_audio_files_exist=True,
    verifica a existência física dos primeiros 20 e dos últimos 20 arquivos WAV (cobrindo tanto
    a porção sobreposta quanto a cauda de 30% de fala limpa).
    """
    checked = 0
    first_entries: List[Tuple[int, str]] = []
    last_entries: List[Tuple[int, str]] = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            data = json.loads(line)
            entry = MultitalkerManifestEntry.model_validate(data)
            checked += 1
            if check_audio_files_exist:
                if len(first_entries) < 20:
                    first_entries.append((line_num, entry.audio_filepath))
                last_entries.append((line_num, entry.audio_filepath))
                if len(last_entries) > 20:
                    last_entries.pop(0)
            if max_lines_to_check > 0 and checked >= max_lines_to_check:
                break

    if checked == 0:
        raise ValueError(f"Manifesto vazio: {manifest_path}")

    if check_audio_files_exist:
        for line_num, audio_fp in first_entries + last_entries:
            wav_p = Path(audio_fp)
            if not wav_p.exists():
                raise FileNotFoundError(
                    f"Áudio referenciado na linha {line_num} de {manifest_path} não encontrado em disco: "
                    f"{wav_p}. Verifique se o pacote .tar.gz foi extraído em /content/local_dataset."
                )

    return checked


def apply_stage_optimizations_to_nemo_model(
    asr_model: Any,
    raw_cfg: Dict[str, Any],
    stage: str,
) -> None:
    """
    Injeta explicitamente no objeto asr_model carregado as otimizações de VRAM/A100
    (fused_batch_size, warprnnt_numba, fastemit_lambda e att_context_size Cache-Aware) (Spec 04).
    """
    try:
        from omegaconf import OmegaConf, open_dict

        is_omega_cfg = hasattr(asr_model, "cfg") and OmegaConf.is_config(asr_model.cfg)
    except ImportError:
        is_omega_cfg = False

    model_cfg = raw_cfg.get("model", {})
    if is_omega_cfg:
        with open_dict(asr_model.cfg):
            if "joint" in model_cfg and "joint" in asr_model.cfg:
                asr_model.cfg.joint.fuse_loss_wer = model_cfg["joint"].get("fuse_loss_wer", True)
                asr_model.cfg.joint.fused_batch_size = model_cfg["joint"].get("fused_batch_size", 16)
            if "loss" in model_cfg and "loss" in asr_model.cfg:
                asr_model.cfg.loss = OmegaConf.merge(asr_model.cfg.loss, OmegaConf.create(model_cfg["loss"]))
            if stage == "multitalker" and "encoder" in model_cfg:
                enc_cfg = model_cfg["encoder"]
                if "att_context_size" in enc_cfg and "encoder" in asr_model.cfg:
                    asr_model.cfg.encoder.att_context_size = enc_cfg["att_context_size"]
    elif hasattr(asr_model, "cfg") and isinstance(asr_model.cfg, dict):
        if "joint" in model_cfg and "joint" in asr_model.cfg:
            asr_model.cfg["joint"]["fuse_loss_wer"] = model_cfg["joint"].get("fuse_loss_wer", True)
            asr_model.cfg["joint"]["fused_batch_size"] = model_cfg["joint"].get("fused_batch_size", 16)
        if "loss" in model_cfg and "loss" in asr_model.cfg:
            asr_model.cfg["loss"].update(model_cfg["loss"])
        if stage == "multitalker" and "encoder" in model_cfg and "encoder" in asr_model.cfg:
            if "att_context_size" in model_cfg["encoder"]:
                asr_model.cfg["encoder"]["att_context_size"] = model_cfg["encoder"]["att_context_size"]

    if hasattr(asr_model, "joint") and "joint" in model_cfg:
        if hasattr(asr_model.joint, "fuse_loss_wer"):
            asr_model.joint.fuse_loss_wer = bool(model_cfg["joint"].get("fuse_loss_wer", True))
        if hasattr(asr_model.joint, "fused_batch_size"):
            asr_model.joint.fused_batch_size = int(model_cfg["joint"].get("fused_batch_size", 16))


def _mirror_state_into_checkpoint_dir(checkpoint_dir: str) -> None:
    """Copia pipeline_state.json e USER.md para dentro de checkpoint_dir no Google Drive."""
    from src.state import DEFAULT_KNOWLEDGE_PATH, DEFAULT_STATE_PATH, DEFAULT_USER_MD_PATH

    ckpt_state_dir = Path(checkpoint_dir) / "state"
    ckpt_state_dir.mkdir(parents=True, exist_ok=True)
    for src_f in (DEFAULT_STATE_PATH, DEFAULT_USER_MD_PATH, DEFAULT_KNOWLEDGE_PATH):
        if src_f.exists():
            shutil.copy2(src_f, ckpt_state_dir / src_f.name)


def run_nemo_training(
    stage: str,
    train_manifest: str,
    val_manifest: Optional[str],
    checkpoint_dir: str,
    tokenizer_dir: Optional[str] = None,
    init_model: Optional[str] = None,
    batch_size: Optional[int] = None,
    config_path: Optional[str] = None,
    check_audio_files_exist: bool = True,
    update_state: bool = False,
) -> None:
    """Executa a Fase 1 (warmup) ou Fase 2 (multitalker) no NVIDIA NeMo + PyTorch Lightning."""
    step_id = "step_05_colab_stage1_warmup" if stage == "warmup" else "step_06_colab_stage2_multitalker"
    if update_state:
        from src.state import start_step

        start_step(
            step_id=step_id,
            command=f"python -m src.trainer --stage {stage} --manifest {train_manifest} --checkpoint_dir {checkpoint_dir}",
            notes=f"Treinamento {stage} iniciado com checkpoints em {checkpoint_dir}.",
        )
        _mirror_state_into_checkpoint_dir(checkpoint_dir)

    try:
        raw_cfg, stage_contract = load_and_validate_stage_config(stage=stage, config_path=config_path)

        checked_train = preflight_validate_manifest(
            train_manifest,
            check_audio_files_exist=check_audio_files_exist,
        )
        print(f"[Gatekeeper SDD] Manifesto de treino validado ({checked_train} amostras verificadas).")

        if val_manifest:
            checked_val = preflight_validate_manifest(
                val_manifest,
                check_audio_files_exist=check_audio_files_exist,
            )
            print(f"[Gatekeeper SDD] Manifesto de validação validado ({checked_val} amostras verificadas).")

        # Sobrescreve parâmetros dinâmicos passados pelo Colab Runner
        raw_cfg["model"]["train_ds"]["manifest_filepath"] = train_manifest
        if val_manifest:
            raw_cfg["model"]["validation_ds"]["manifest_filepath"] = val_manifest
        if tokenizer_dir:
            raw_cfg["model"]["tokenizer"]["dir"] = tokenizer_dir
        if init_model:
            raw_cfg["init_from_nemo_model"] = init_model
        if batch_size:
            raw_cfg["model"]["train_ds"]["batch_size"] = batch_size
        raw_cfg["exp_manager"]["exp_dir"] = checkpoint_dir

        if stage_contract.stage == "warmup" and check_audio_files_exist:
            tok_dir = Path(raw_cfg["model"]["tokenizer"]["dir"])
            if not tok_dir.exists():
                raise FileNotFoundError(
                    f"Diretório do tokenizador BPE PT-BR não encontrado: {tok_dir}. "
                    "Conclua a step_02_tokenizer_ptbr e sincronize o diretório no Google Drive."
                )

        # Importações pesadas de GPU apenas após validação completa dos contratos
        import lightning.pytorch as pl
        from omegaconf import OmegaConf
        import nemo.collections.asr as nemo_asr
        from nemo.utils.exp_manager import exp_manager

        trainer = pl.Trainer(**raw_cfg["trainer"])
        exp_manager(trainer, OmegaConf.create(raw_cfg["exp_manager"]))

        if stage_contract.stage == "warmup":
            asr_model = nemo_asr.models.ASRModel.from_pretrained(
                raw_cfg["pretrained_model"],
                map_location=trainer.strategy.root_device,
            )
            asr_model.change_vocabulary(
                new_tokenizer_dir=raw_cfg["model"]["tokenizer"]["dir"],
                new_tokenizer_type=raw_cfg["model"]["tokenizer"]["type"],
            )
            apply_stage_optimizations_to_nemo_model(asr_model, raw_cfg, stage="warmup")
            asr_model.encoder.freeze()
            print("[Spec 04 - Warmup] FastConformer + Speaker Kernels CONGELADOS. Treinando Decoder RNN-T + Joint.")
        else:
            nemo_path = resolve_nemo_checkpoint_path(raw_cfg["init_from_nemo_model"])
            asr_model = nemo_asr.models.ASRModel.restore_from(
                str(nemo_path),
                map_location=trainer.strategy.root_device,
            )
            apply_stage_optimizations_to_nemo_model(asr_model, raw_cfg, stage="multitalker")
            asr_model.encoder.unfreeze()
            print("[Spec 04 - Multitalker] Modelo completo DESCONGELADO com Speaker Kernels e Cache-Aware ativos.")

        asr_model.setup_training_data(OmegaConf.create(raw_cfg["model"]["train_ds"]))
        if val_manifest:
            asr_model.setup_validation_data(OmegaConf.create(raw_cfg["model"]["validation_ds"]))
        asr_model.setup_optimization(OmegaConf.create(raw_cfg["model"]["optim"]))

        trainer.fit(asr_model)

        if update_state:
            from src.state import complete_step

            next_step = "step_06_colab_stage2_multitalker" if stage == "warmup" else "step_07_eval_and_export"
            next_stage_cmd = (
                "python -m src.trainer --stage multitalker --manifest /content/local_dataset/multitalker_train_manifest.json "
                "--val_manifest /content/local_dataset/multitalker_val_manifest.json "
                "--checkpoint_dir /content/drive/MyDrive/parakeet_checkpoints/exp_v1_multitalker --update_state"
                if stage == "warmup"
                else (
                    f"python -m src.evaluator --model_path {checkpoint_dir}/parakeet_ptbr_multitalker/checkpoints/parakeet_ptbr_multitalker.nemo "
                    "--clean_manifest /content/local_dataset/clean_val_manifest.json "
                    "--multitalker_manifest /content/local_dataset/multitalker_val_manifest.json "
                    f"--output_report {checkpoint_dir}/eval_report.json --update_state"
                )
            )
            complete_step(
                step_id=step_id,
                next_step_id=next_step,
                next_action=f"Avançar para {next_step} após conclusão de {stage}.",
                next_command=next_stage_cmd,
                last_summary=f"Treinamento do estágio '{stage}' concluído com checkpoints salvos em {checkpoint_dir}.",
                artifacts={f"{stage}_checkpoint_dir": checkpoint_dir},
            )
            _mirror_state_into_checkpoint_dir(checkpoint_dir)
    except Exception as exc:
        if update_state:
            from src.state import fail_step

            fail_step(step_id=step_id, error_message=str(exc))
            _mirror_state_into_checkpoint_dir(checkpoint_dir)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executor NeMo para adaptação PT-BR do multitalker-parakeet na A100 (Spec 04)."
    )
    parser.add_argument("--stage", choices=["warmup", "multitalker", "eval"], required=True)
    parser.add_argument("--manifest", required=True, help="Caminho para o train_manifest.json (ou clean_manifest em eval).")
    parser.add_argument("--val_manifest", default=None, help="Caminho opcional para val_manifest.json (ou multitalker_manifest em eval).")
    parser.add_argument("--checkpoint_dir", required=True, help="Diretório no Google Drive para salvar checkpoints/relatórios.")
    parser.add_argument("--tokenizer_dir", default=None, help="Diretório contendo o tokenizador BPE PT-BR (1024).")
    parser.add_argument("--init_model", default=None, help="Caminho opcional do .nemo para iniciar Fase 2 ou avaliar em eval.")
    parser.add_argument("--batch_size", type=int, default=None, help="Sobrescreve o batch_size padrão.")
    parser.add_argument("--config", default=None, help="Caminho opcional para YAML customizado.")
    parser.add_argument(
        "--skip_audio_file_check",
        action="store_true",
        help="Pula a verificação física de existência dos primeiros WAVs no preflight.",
    )
    parser.add_argument("--update_state", action="store_true", help="Atualiza state/pipeline_state.json e USER.md.")
    args = parser.parse_args()

    if args.stage == "eval":
        from src.evaluator import evaluate_nemo_checkpoint

        if not args.init_model or not args.val_manifest:
            raise ValueError("--stage eval exige --init_model (caminho do .nemo) e --val_manifest (manifesto multitalker).")
        resolved_nemo = resolve_nemo_checkpoint_path(args.init_model)
        report_out = str(Path(args.checkpoint_dir) / "eval_report.json")
        evaluate_nemo_checkpoint(
            model_path=str(resolved_nemo),
            clean_manifest_path=args.manifest,
            multitalker_manifest_path=args.val_manifest,
            output_report_path=report_out,
            batch_size=args.batch_size or 16,
            update_state=args.update_state,
        )
        return

    run_nemo_training(
        stage=args.stage,
        train_manifest=args.manifest,
        val_manifest=args.val_manifest,
        checkpoint_dir=args.checkpoint_dir,
        tokenizer_dir=args.tokenizer_dir,
        init_model=args.init_model,
        batch_size=args.batch_size,
        config_path=args.config,
        check_audio_files_exist=not args.skip_audio_file_check,
        update_state=args.update_state,
    )


if __name__ == "__main__":
    main()
