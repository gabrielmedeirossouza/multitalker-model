"""Orquestrador de treinamento em 2 fases e avaliação no NVIDIA NeMo para GPU A100 no Google Colab (Spec 04)."""

from __future__ import annotations

import argparse
import json
import math
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
    val_ds_cfg = raw_cfg.get("model", {}).get("validation_ds", {})
    ckpt_cfg = raw_cfg["exp_manager"].get("checkpoint_callback_params", {})

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
        use_lhotse=bool(train_ds_cfg.get("use_lhotse", False)) and bool(val_ds_cfg.get("use_lhotse", False)),
        trainer_logger=bool(raw_cfg["trainer"].get("logger", True)),
        ckpt_every_n_train_steps=ckpt_cfg.get("every_n_train_steps"),
        ckpt_every_n_epochs=ckpt_cfg.get("every_n_epochs"),
    )
    return raw_cfg, contract


def compute_max_steps(
    num_train_samples: int,
    batch_size: int,
    max_epochs: int,
    accumulate_grad_batches: int = 1,
    num_buckets: int = 0,
) -> int:
    """
    Calcula o teto de trainer.max_steps (passos do otimizador) para o dataloader Lhotse, que não expõe len()
    e impede o NeMo de derivar max_steps a partir de max_epochs. num_buckets soma a margem de lotes parciais
    do bucketing, garantindo que o treino termine por max_epochs (Spec 04 §8.4).
    """
    if num_train_samples <= 0 or batch_size <= 0 or max_epochs <= 0 or accumulate_grad_batches <= 0:
        raise ValueError(
            "compute_max_steps exige num_train_samples, batch_size, max_epochs e accumulate_grad_batches > 0 "
            f"(recebeu {num_train_samples}, {batch_size}, {max_epochs}, {accumulate_grad_batches})."
        )
    if num_buckets < 0:
        raise ValueError(f"num_buckets não pode ser negativo (recebeu {num_buckets}).")
    batches_per_epoch = math.ceil(num_train_samples / batch_size) + num_buckets
    return math.ceil(max_epochs * batches_per_epoch / accumulate_grad_batches)


def align_max_steps_to_val_interval(max_steps: int, val_check_interval: Any) -> int:
    """
    Arredonda max_steps para o próximo múltiplo de um val_check_interval inteiro (validação por passos),
    para que o último passo coincida com uma validação (Spec 04 §8.4). val_check_interval float
    (ex.: 1.0 = fim de época) mantém max_steps inalterado.
    """
    if isinstance(val_check_interval, bool) or not isinstance(val_check_interval, int) or val_check_interval <= 1:
        return max_steps
    return math.ceil(max_steps / val_check_interval) * val_check_interval


def freeze_speaker_kernels(asr_model: Any) -> int:
    """
    Congela spk_kernels e bg_spk_kernels do EncDecMultiTalkerRNNTBPEModel, que ficam fora de
    asr_model.encoder e não são afetados por encoder.freeze() (Spec 04 §5.5). Retorna o nº de tensores congelados.
    """
    frozen = 0
    for attr in ("spk_kernels", "bg_spk_kernels"):
        module = getattr(asr_model, attr, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = False
            frozen += 1
    return frozen


VOCAB_INDEPENDENT_WEIGHT_PREFIXES = {
    "joint": ("enc.", "pred."),
    "decoder": ("prediction.dec_rnn.",),
}


def snapshot_vocab_independent_weights(asr_model: Any) -> Dict[str, Dict[str, Any]]:
    """
    Copia, antes de change_vocabulary(), os pesos pré-treinados do Joint/Decoder que não dependem dos IDs de token
    (allowlist por prefixo; embed/joint_net têm o mesmo shape no vocabulário original de 1024 e não entram) (Spec 04 §5.6).
    """
    snapshot: Dict[str, Dict[str, Any]] = {}
    for module_name, prefixes in VOCAB_INDEPENDENT_WEIGHT_PREFIXES.items():
        module = getattr(asr_model, module_name)
        snapshot[module_name] = {
            name: tensor.detach().clone()
            for name, tensor in module.state_dict().items()
            if name.startswith(prefixes)
        }
    return snapshot


def restore_vocab_independent_weights(asr_model: Any, snapshot: Dict[str, Dict[str, Any]]) -> int:
    """Restaura no Joint/Decoder recriados por change_vocabulary() os pesos da allowlist. Retorna o nº de tensores."""
    restored = 0
    for module_name, weights in snapshot.items():
        module = getattr(asr_model, module_name)
        state = module.state_dict()
        for name, tensor in weights.items():
            if name in state and tuple(state[name].shape) == tuple(tensor.shape):
                state[name] = tensor
                restored += 1
        module.load_state_dict(state)
    return restored


def rebuild_rnnt_loss_from_cfg(asr_model: Any) -> None:
    """
    Recria asr_model.loss a partir de asr_model.cfg.loss e cfg.rnnt_reduction. No NeMo 3.0, change_vocabulary()
    instancia RNNTLoss(num_classes=...) padrão, descartando fastemit_lambda e a redução configurada (Spec 04 §5.3).
    """
    from nemo.collections.asr.losses.rnnt import RNNTLoss

    loss_name, loss_kwargs = asr_model.extract_rnnt_loss_cfg(asr_model.cfg.get("loss", None))
    asr_model.loss = RNNTLoss(
        num_classes=asr_model.joint.num_classes_with_blank - 1,
        loss_name=loss_name,
        loss_kwargs=loss_kwargs,
        reduction=asr_model.cfg.get("rnnt_reduction", "mean_batch"),
    )
    if asr_model.joint.fuse_loss_wer:
        asr_model.joint.set_loss(asr_model.loss)
    print(f"[Spec 04] RNNTLoss reconstruída: loss_name={loss_name}, kwargs={loss_kwargs}.")


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


def configure_multitalker_cuts(
    raw_cfg: Dict[str, Any],
    train_manifest: str,
    val_manifest: Optional[str],
) -> int:
    """
    Fase 2: o LhotseSpeechToTextSpkBpeDataset deriva as máscaras das supervisions, não do manifesto.
    Troca manifest_filepath pelo cuts_path do CutSet derivado do manifesto espelhado (Spec 02 §3.7) e
    retorna o número de cuts de treino, base do max_steps (Spec 04 §8.2).
    """
    from src.mixer import export_lhotse_cuts

    num_train_cuts = 0
    for ds_key, manifest in (("train_ds", train_manifest), ("validation_ds", val_manifest)):
        if not manifest:
            continue
        cuts_path, num_cuts = export_lhotse_cuts(manifest)
        ds_cfg = raw_cfg["model"][ds_key]
        ds_cfg.pop("manifest_filepath", None)
        ds_cfg["cuts_path"] = cuts_path
        print(f"[Spec 04 §8.2] {ds_key}: {num_cuts} cuts Lhotse em {cuts_path}.")
        if ds_key == "train_ds":
            num_train_cuts = num_cuts
    return num_train_cuts


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
        joint = asr_model.joint
        fuse_loss_wer = bool(model_cfg["joint"].get("fuse_loss_wer", True))
        fused_batch_size = int(model_cfg["joint"].get("fused_batch_size", 16))
        if hasattr(joint, "set_fuse_loss_wer"):
            # NeMo 3.0: propriedades somente leitura; set_fuse_loss_wer() sobrescreve loss/WER fundidos (Spec 04 §5.2)
            if joint.fuse_loss_wer != fuse_loss_wer:
                joint.set_fuse_loss_wer(
                    fuse_loss_wer,
                    loss=getattr(asr_model, "loss", None) if fuse_loss_wer else None,
                    metric=getattr(asr_model, "wer", None) if fuse_loss_wer else None,
                )
            joint.set_fused_batch_size(fused_batch_size)
        else:
            if hasattr(joint, "fuse_loss_wer"):
                joint.fuse_loss_wer = fuse_loss_wer
            if hasattr(joint, "fused_batch_size"):
                joint.fused_batch_size = fused_batch_size


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
    max_steps: Optional[int] = None,
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
        if stage_contract.stage == "multitalker":
            checked_train = configure_multitalker_cuts(raw_cfg, train_manifest, val_manifest)
        if tokenizer_dir:
            raw_cfg["model"]["tokenizer"]["dir"] = tokenizer_dir
        if init_model:
            raw_cfg["init_from_nemo_model"] = init_model
        if batch_size:
            raw_cfg["model"]["train_ds"]["batch_size"] = batch_size
        raw_cfg["exp_manager"]["exp_dir"] = checkpoint_dir

        # Spec 04 §8.4: Lhotse não expõe len() — max_steps explícito para o scheduler do NeMo
        trainer_cfg = raw_cfg["trainer"]
        train_ds_cfg = raw_cfg["model"]["train_ds"]
        if max_steps:
            trainer_cfg["max_steps"] = max_steps
        elif int(trainer_cfg.get("max_steps", -1)) <= 0:
            trainer_cfg["max_steps"] = compute_max_steps(
                num_train_samples=checked_train,
                batch_size=int(train_ds_cfg["batch_size"]),
                max_epochs=int(trainer_cfg["max_epochs"]),
                accumulate_grad_batches=int(trainer_cfg.get("accumulate_grad_batches", 1)),
                num_buckets=int(train_ds_cfg.get("num_buckets", 30)) if train_ds_cfg.get("use_bucketing") else 0,
            )
            aligned_steps = align_max_steps_to_val_interval(
                trainer_cfg["max_steps"], trainer_cfg.get("val_check_interval")
            )
            # Validação por passos (check_val_every_n_epoch: null explícito; ausente = 1 no Lightning):
            # max_steps vira o limite efetivo e termina numa validação
            if "check_val_every_n_epoch" in trainer_cfg and trainer_cfg["check_val_every_n_epoch"] is None:
                trainer_cfg["max_steps"] = aligned_steps
                trainer_cfg["max_epochs"] = -1
        print(
            f"[Spec 04 §8.4] trainer.max_steps={trainer_cfg['max_steps']} "
            f"(max_epochs={trainer_cfg.get('max_epochs')}, amostras={checked_train})."
        )

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
            reuse_weights = bool(raw_cfg.get("reuse_vocab_independent_weights", True))
            snapshot = snapshot_vocab_independent_weights(asr_model) if reuse_weights else None
            asr_model.change_vocabulary(
                new_tokenizer_dir=raw_cfg["model"]["tokenizer"]["dir"],
                new_tokenizer_type=raw_cfg["model"]["tokenizer"]["type"],
            )
            if snapshot is not None:
                restored = restore_vocab_independent_weights(asr_model, snapshot)
                print(
                    f"[Spec 04 §5.6] {restored} tensores pré-treinados restaurados no Joint/Decoder "
                    "(joint.enc, joint.pred, decoder LSTM); embed e joint_net reinicializados para o vocabulário PT-BR."
                )
            apply_stage_optimizations_to_nemo_model(asr_model, raw_cfg, stage="warmup")
            rebuild_rnnt_loss_from_cfg(asr_model)
            asr_model.encoder.freeze()
            frozen_kernel_tensors = freeze_speaker_kernels(asr_model)
            trainable = sum(p.numel() for p in asr_model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in asr_model.parameters())
            print(
                "[Spec 04 - Warmup] FastConformer + Speaker Kernels CONGELADOS "
                f"({frozen_kernel_tensors} tensores de kernel). Treinando Decoder RNN-T + Joint: "
                f"{trainable:,} de {total:,} parâmetros ({100.0 * trainable / total:.1f}%)."
            )
        else:
            nemo_path = resolve_nemo_checkpoint_path(raw_cfg["init_from_nemo_model"])
            asr_model = nemo_asr.models.ASRModel.restore_from(
                str(nemo_path),
                map_location=trainer.strategy.root_device,
            )
            apply_stage_optimizations_to_nemo_model(asr_model, raw_cfg, stage="multitalker")
            rebuild_rnnt_loss_from_cfg(asr_model)
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
            # Fase 2 em diretório irmão do warmup (ex.: exp_coraa_mupe_warmup -> exp_coraa_mupe_multitalker)
            warmup_dir = Path(checkpoint_dir)
            multitalker_dir = warmup_dir.with_name(warmup_dir.name.replace("warmup", "multitalker"))
            next_stage_cmd = (
                "python -m src.trainer --stage multitalker --manifest /content/local_dataset/multitalker_train_manifest.json "
                "--val_manifest /content/local_dataset/multitalker_val_manifest.json "
                f"--checkpoint_dir {multitalker_dir} "
                f"--init_model {warmup_dir}/parakeet_ptbr_warmup/checkpoints/parakeet_ptbr_warmup.nemo "
                f"--tokenizer_dir {raw_cfg['model']['tokenizer']['dir']} --update_state"
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
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Sobrescreve trainer.max_steps (padrão: calculado a partir do manifesto e max_epochs, Spec 04 §8.4).",
    )
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
        max_steps=args.max_steps,
        config_path=args.config,
        check_audio_files_exist=not args.skip_audio_file_check,
        update_state=args.update_state,
    )


if __name__ == "__main__":
    main()
