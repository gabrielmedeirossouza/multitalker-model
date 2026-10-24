"""Testes de contrato para manifestos NeMo e configurações de estágio (Spec 01 e Spec 04)."""

import json
import math

import numpy as np
import pytest
from pydantic import ValidationError

from src.schemas import (
    EvaluationReport,
    FRAME_DURATION_SEC,
    HFDatasetSource,
    MultitalkerManifestEntry,
    REQUIRED_ATT_CONTEXT_SIZE,
    SingleSpeakerManifestEntry,
    TrainingStageConfig,
    expected_frames_for_duration,
)
from src.data_prep import (
    SUPPORTED_HF_DATASETS,
    ensure_colab_dataset_ready,
    export_hf_dataset_to_nemo_manifest,
    normalize_transcript,
    pack_dataset_for_colab_ssd,
    rewrite_manifest_audio_paths,
)
from src.evaluator import compute_word_error_rate
from src.trainer import (
    align_max_steps_to_val_interval,
    apply_stage_optimizations_to_nemo_model,
    compute_max_steps,
    freeze_speaker_kernels,
    load_and_validate_stage_config,
    preflight_validate_manifest,
    resolve_nemo_checkpoint_path,
    restore_vocab_independent_weights,
    snapshot_vocab_independent_weights,
)


def test_valid_multitalker_manifest_entry():
    duration = 4.0  # 4.0 / 0.08 = 50 frames exatos
    expected_frames = math.ceil(duration / FRAME_DURATION_SEC)
    entry = MultitalkerManifestEntry(
        audio_filepath="/content/local_dataset/mix_001.wav",
        duration=duration,
        text="Bom dia pessoal, hoje vamos revisar os resultados.",
        spk_targets=[1] * 30 + [0] * 20,
        bg_spk_targets=[0] * 15 + [1] * 25 + [0] * 10,
    )
    assert len(entry.spk_targets) == expected_frames
    assert len(entry.bg_spk_targets) == expected_frames


def test_single_speaker_manifest_factory():
    entry = SingleSpeakerManifestEntry.from_clean_audio(
        audio_filepath="/content/local_dataset/clean_01.wav",
        duration=2.45,
        text="Olá, tudo bem com você?",
        speaker_id="spk_01",
    )
    expected_frames = math.ceil(2.45 / 0.08)
    assert len(entry.spk_targets) == expected_frames
    assert all(v == 1 for v in entry.spk_targets)
    assert all(v == 0 for v in entry.bg_spk_targets)


def test_rejects_misaligned_frame_masks():
    # Para 4.0s são esperados 50 frames; passar 40 deve disparar ValidationError
    with pytest.raises(ValidationError, match="Desalinhamento de máscara"):
        MultitalkerManifestEntry(
            audio_filepath="/content/local_dataset/mix_bad.wav",
            duration=4.0,
            text="Texto de teste",
            spk_targets=[1] * 40,
            bg_spk_targets=[0] * 40,
        )


def test_rejects_unequal_mask_lengths():
    with pytest.raises(ValidationError, match="mesmo comprimento"):
        MultitalkerManifestEntry(
            audio_filepath="/content/local_dataset/mix_bad2.wav",
            duration=4.0,
            text="Texto de teste",
            spk_targets=[1] * 50,
            bg_spk_targets=[0] * 49,
        )


def test_rejects_non_binary_mask_values():
    with pytest.raises(ValidationError, match="valores binários"):
        MultitalkerManifestEntry(
            audio_filepath="/content/local_dataset/mix_bad3.wav",
            duration=1.6,
            text="Texto de teste",
            spk_targets=[1, 2] * 10,
            bg_spk_targets=[0, 1] * 10,
        )


@pytest.mark.parametrize("bad_duration", [0.2, 35.0])
def test_rejects_out_of_bounds_duration(bad_duration: float):
    frames = max(1, math.ceil(bad_duration / 0.08))
    with pytest.raises(ValidationError):
        MultitalkerManifestEntry(
            audio_filepath="/content/local_dataset/mix_dur.wav",
            duration=bad_duration,
            text="Texto de teste",
            spk_targets=[1] * frames,
            bg_spk_targets=[0] * frames,
        )


def test_yaml_configs_comply_with_spec_04():
    _, warmup_contract = load_and_validate_stage_config("warmup")
    assert warmup_contract.stage == "warmup"
    assert warmup_contract.freeze_encoder is True
    assert warmup_contract.precision == "bf16-mixed"
    assert warmup_contract.resume_if_exists is True

    _, multi_contract = load_and_validate_stage_config("multitalker")
    assert multi_contract.stage == "multitalker"
    assert multi_contract.freeze_encoder is False
    assert multi_contract.precision == "bf16-mixed"
    assert multi_contract.resume_if_exists is True


def test_rejects_unfrozen_encoder_in_warmup():
    with pytest.raises(ValidationError, match="freeze_encoder DEVE ser True"):
        TrainingStageConfig(stage="warmup", freeze_encoder=False)


@pytest.mark.parametrize("stage", ["warmup", "multitalker"])
def test_yaml_configs_comply_with_nemo3_lhotse_integration(stage: str):
    raw_cfg, contract = load_and_validate_stage_config(stage)
    assert contract.use_lhotse is True
    assert contract.trainer_logger is False
    assert raw_cfg["model"]["validation_ds"]["use_lhotse"] is True
    assert raw_cfg["model"]["train_ds"]["text_field"] == "text"
    assert raw_cfg["trainer"]["use_distributed_sampler"] is False
    assert raw_cfg["trainer"]["enable_checkpointing"] is False
    assert raw_cfg["exp_manager"]["checkpoint_callback_params"]["save_best_model"] is True


def test_warmup_lr_schedule_scales_with_max_steps():
    raw_cfg, contract = load_and_validate_stage_config("warmup")
    sched = raw_cfg["model"]["optim"]["sched"]
    assert "warmup_steps" not in sched
    assert 0.0 < sched["warmup_ratio"] < 1.0
    # Épocas curtas: checkpoint a cada validação (sem gatilho por passos)
    assert contract.ckpt_every_n_train_steps is None


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"use_lhotse": False}, "use_lhotse DEVE ser True"),
        ({"trainer_logger": True}, "trainer.logger DEVE ser False"),
        ({"ckpt_every_n_train_steps": 1000, "ckpt_every_n_epochs": 1}, "mutuamente exclusivos"),
    ],
)
def test_rejects_nemo3_misconfigurations(kwargs, match):
    with pytest.raises(ValidationError, match=match):
        TrainingStageConfig(stage="warmup", freeze_encoder=True, **kwargs)


def test_compute_max_steps_formula():
    # Sem bucketing: FLEURS pt_br 2596 amostras, batch 32 -> 82 lotes/época; 30 épocas -> 2460 passos
    assert compute_max_steps(num_train_samples=2596, batch_size=32, max_epochs=30) == 2460
    # Com 8 buckets Lhotse: teto de 82 + 8 = 90 lotes/época (medido na A100: 84) -> 2700 passos
    steps = compute_max_steps(num_train_samples=2596, batch_size=32, max_epochs=30, num_buckets=8)
    assert steps == 2700
    assert steps >= 30 * 84  # max_steps nunca corta a última época (e sua validação)
    # accumulate_grad_batches divide o número de passos do otimizador (arredonda para cima)
    assert compute_max_steps(num_train_samples=100, batch_size=16, max_epochs=3, accumulate_grad_batches=2) == 11
    with pytest.raises(ValueError):
        compute_max_steps(num_train_samples=0, batch_size=32, max_epochs=5)
    with pytest.raises(ValueError):
        compute_max_steps(num_train_samples=10, batch_size=2, max_epochs=1, num_buckets=-1)


def test_warmup_checkpointing_respects_drive_budget():
    raw_cfg, _ = load_and_validate_stage_config("warmup")
    ckpt = raw_cfg["exp_manager"]["checkpoint_callback_params"]
    assert ckpt["save_top_k"] == 1
    assert ckpt["always_save_nemo"] is False
    trainer_cfg = raw_cfg["trainer"]
    if trainer_cfg["check_val_every_n_epoch"] is None:
        # Validação por passos (corpus grande): intervalo inteiro de lotes, alinhado a max_steps no src/trainer.py
        assert isinstance(trainer_cfg["val_check_interval"], int) and trainer_cfg["val_check_interval"] > 1
        assert trainer_cfg["accumulate_grad_batches"] == 1
    else:
        # A última época sempre é validada (max_epochs múltiplo de check_val_every_n_epoch)
        assert trainer_cfg["max_epochs"] % trainer_cfg["check_val_every_n_epoch"] == 0


def test_align_max_steps_to_val_interval():
    # MUPE: 3 épocas de ~7,8 mil lotes -> 23.448 passos -> próximo múltiplo de 4000 (termina numa validação)
    assert align_max_steps_to_val_interval(23448, 4000) == 24000
    assert align_max_steps_to_val_interval(24000, 4000) == 24000
    # Validação no fim de época (float) ou intervalos degenerados não alteram max_steps
    assert align_max_steps_to_val_interval(2700, 1.0) == 2700
    assert align_max_steps_to_val_interval(2700, 1) == 2700
    assert align_max_steps_to_val_interval(2700, True) == 2700


def test_vocab_independent_weights_survive_change_vocabulary():
    class FakeTensor:
        def __init__(self, tag, shape):
            self.tag, self.shape = tag, shape

        def detach(self):
            return self

        def clone(self):
            return FakeTensor(self.tag, self.shape)

    class FakeModule:
        def __init__(self, tag, names):
            self._sd = {n: FakeTensor(tag, (1025, 640)) for n in names}

        def state_dict(self):
            return dict(self._sd)

        def load_state_dict(self, sd):
            self._sd = dict(sd)

    joint_names = ["enc.weight", "enc.bias", "pred.weight", "pred.bias", "joint_net.2.weight", "joint_net.2.bias"]
    dec_names = ["prediction.embed.weight", "prediction.dec_rnn.lstm.weight_ih_l0", "prediction.dec_rnn.lstm.bias_hh_l1"]

    class FakeModel:
        joint = FakeModule("pretrained", joint_names)
        decoder = FakeModule("pretrained", dec_names)

    model = FakeModel()
    snapshot = snapshot_vocab_independent_weights(model)
    # change_vocabulary() recria Joint/Decoder com pesos aleatórios (mesmos shapes: vocabulário original também é 1024)
    model.joint, model.decoder = FakeModule("random", joint_names), FakeModule("random", dec_names)
    assert restore_vocab_independent_weights(model, snapshot) == 6

    joint_sd, dec_sd = model.joint.state_dict(), model.decoder.state_dict()
    for name in ("enc.weight", "enc.bias", "pred.weight", "pred.bias"):
        assert joint_sd[name].tag == "pretrained"
    for name in ("prediction.dec_rnn.lstm.weight_ih_l0", "prediction.dec_rnn.lstm.bias_hh_l1"):
        assert dec_sd[name].tag == "pretrained"
    # Camadas dependentes dos IDs de token nunca são copiadas, mesmo com shape idêntico
    assert joint_sd["joint_net.2.weight"].tag == "random"
    assert dec_sd["prediction.embed.weight"].tag == "random"


def test_warmup_config_reuses_vocab_independent_weights():
    raw_cfg, _ = load_and_validate_stage_config("warmup")
    assert raw_cfg["reuse_vocab_independent_weights"] is True


def test_freeze_speaker_kernels_disables_gradients():
    class DummyParam:
        requires_grad = True

    class DummyModule:
        def __init__(self, n):
            self._params = [DummyParam() for _ in range(n)]

        def parameters(self):
            return iter(self._params)

    class DummyModel:
        spk_kernels = DummyModule(4)
        bg_spk_kernels = DummyModule(4)

    model = DummyModel()
    assert freeze_speaker_kernels(model) == 8
    assert not any(p.requires_grad for p in model.spk_kernels.parameters())
    assert not any(p.requires_grad for p in model.bg_spk_kernels.parameters())
    # Modelos sem Speaker Kernels são aceitos sem erro
    assert freeze_speaker_kernels(object()) == 0


def test_rewrite_manifest_audio_paths_and_preflight_check(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir(parents=True)
    wav_file = wav_dir / "sample_01.wav"
    wav_file.write_bytes(b"RIFF_DUMMY")

    local_manifest = tmp_path / "local_manifest.json"
    entry = SingleSpeakerManifestEntry.from_clean_audio(
        audio_filepath=str(wav_file),
        duration=2.0,
        text="Teste de caminho local para Colab SSD.",
        speaker_id="spk_1",
    )
    local_manifest.write_text(entry.model_dump_json() + "\n", encoding="utf-8")

    # Preflight deve passar quando o arquivo existe localmente
    assert preflight_validate_manifest(str(local_manifest), check_audio_files_exist=True) == 1

    # Reescreve para /content/local_dataset
    colab_manifest = tmp_path / "colab_manifest.json"
    count = rewrite_manifest_audio_paths(
        manifest_path=str(local_manifest),
        output_manifest_path=str(colab_manifest),
        new_root_dir="/content/local_dataset",
        source_root_dir=str(tmp_path),
    )
    assert count == 1
    rewritten_line = colab_manifest.read_text(encoding="utf-8").strip()
    assert "/content/local_dataset/wavs/sample_01.wav" in rewritten_line

    # Como /content/local_dataset/wavs/sample_01.wav não existe nesta máquina local,
    # check_audio_files_exist=True deve levantar FileNotFoundError
    with pytest.raises(FileNotFoundError, match="não encontrado em disco"):
        preflight_validate_manifest(str(colab_manifest), check_audio_files_exist=True)


def test_ieee754_floating_point_frame_alignment():
    # Em IEEE-754 puro, 0.56 / 0.08 = 7.000000000000001 (ceil daria 8 sem arredondamento)
    assert expected_frames_for_duration(0.56) == 7
    assert expected_frames_for_duration(1.12) == 14
    assert expected_frames_for_duration(2.24) == 28


def test_apply_stage_optimizations_and_resolve_nemo_checkpoint(tmp_path):
    class DummyJoint:
        fuse_loss_wer = False
        fused_batch_size = 1

    class DummyModel:
        def __init__(self):
            self.cfg = {
                "joint": {"fuse_loss_wer": False, "fused_batch_size": 1},
                "loss": {"loss_name": "default"},
                "encoder": {"att_context_size": [70, 13]},
            }
            self.joint = DummyJoint()

    raw_cfg, _ = load_and_validate_stage_config("multitalker")
    dummy = DummyModel()
    apply_stage_optimizations_to_nemo_model(dummy, raw_cfg, stage="multitalker")

    assert dummy.joint.fuse_loss_wer is True
    assert dummy.joint.fused_batch_size == 16
    assert dummy.cfg["loss"]["loss_name"] == "warprnnt_numba"
    assert dummy.cfg["encoder"]["att_context_size"] == REQUIRED_ATT_CONTEXT_SIZE

    # NeMo 3.0: RNNTJoint expõe fuse_loss_wer/fused_batch_size como propriedades somente leitura + setters
    class Nemo3Joint:
        def __init__(self, fuse):
            self._fuse_loss_wer, self._fused_batch_size = fuse, 4
            self._loss = self._wer = "fused"

        @property
        def fuse_loss_wer(self):
            return self._fuse_loss_wer

        @property
        def fused_batch_size(self):
            return self._fused_batch_size

        def set_fuse_loss_wer(self, fuse_loss_wer, loss=None, metric=None):
            self._fuse_loss_wer, self._loss, self._wer = fuse_loss_wer, loss, metric

        def set_fused_batch_size(self, fused_batch_size):
            self._fused_batch_size = fused_batch_size

    nemo3 = DummyModel()
    nemo3.joint = Nemo3Joint(fuse=True)
    apply_stage_optimizations_to_nemo_model(nemo3, raw_cfg, stage="multitalker")
    assert nemo3.joint.fused_batch_size == 16
    assert nemo3.joint._loss == "fused"  # valor inalterado não religa (nem zera) a perda fundida

    nemo3_off = DummyModel()
    nemo3_off.loss, nemo3_off.wer = "rnnt_loss", "wer_metric"
    nemo3_off.joint = Nemo3Joint(fuse=False)
    apply_stage_optimizations_to_nemo_model(nemo3_off, raw_cfg, stage="multitalker")
    assert nemo3_off.joint.fuse_loss_wer is True
    assert (nemo3_off.joint._loss, nemo3_off.joint._wer) == ("rnnt_loss", "wer_metric")

    # Testa busca recursiva em subpastas de timestamp do exp_manager
    ts_dir = tmp_path / "exp_v1_warmup" / "parakeet_ptbr_warmup" / "2026-10-24_12-00-00" / "checkpoints"
    ts_dir.mkdir(parents=True)
    nemo_file = ts_dir / "parakeet_ptbr_warmup.nemo"
    nemo_file.write_bytes(b"NEMO_DUMMY")

    configured_missing = tmp_path / "exp_v1_warmup" / "parakeet_ptbr_warmup" / "checkpoints" / "parakeet_ptbr_warmup.nemo"
    resolved = resolve_nemo_checkpoint_path(str(configured_missing))
    assert resolved == nemo_file


def test_evaluator_wer_and_report_contract():
    wer_zero = compute_word_error_rate(["olá mundo brasil"], ["Olá mundo Brasil"])
    assert wer_zero == 0.0

    wer_half = compute_word_error_rate(["bom dia pessoal tudo bem"], ["bom dia galera tudo ótimo"])
    assert np.isclose(wer_half, 0.4, atol=1e-4) if "np" in globals() else abs(wer_half - 0.4) < 1e-4

    report = EvaluationReport(
        model_path="/content/drive/MyDrive/model.nemo",
        clean_manifest_path="/content/local_dataset/clean_val.json",
        multitalker_manifest_path="/content/local_dataset/multi_val.json",
        clean_samples_evaluated=100,
        multitalker_samples_evaluated=200,
        clean_wer=0.08,
        multitalker_wer=0.14,
        evaluated_at="2026-10-24T12:00:00Z",
    )
    assert report.clean_wer == 0.08


def test_ensure_colab_dataset_ready_fast_path_extracts_existing_tar(tmp_path):
    # Simula um pacote já existente no Google Drive (/content/drive/MyDrive/parakeet_ptbr)
    staging = tmp_path / "staging"
    staging.mkdir(parents=True)
    entry = SingleSpeakerManifestEntry.from_clean_audio(
        audio_filepath="/content/local_dataset/wavs/a.wav",
        duration=1.6,
        text="Teste no Colab SSD.",
        speaker_id="spk_01",
    )
    (staging / "multitalker_train_manifest.json").write_text(entry.model_dump_json() + "\n", encoding="utf-8")

    # Artefatos por corpus em <drive_root_dir>/<dataset>/ (Spec 01 §3)
    drive_root = tmp_path / "drive_parakeet"
    corpus_root = drive_root / "fleurs_pt"
    tok_dir = corpus_root / "tokenizer_ptbr_1024"
    tok_dir.mkdir(parents=True)
    (tok_dir / "tokenizer.vocab").write_text("a\t0\n", encoding="utf-8")

    tar_path = corpus_root / "ptbr_dataset_16khz.tar.gz"
    pack_dataset_for_colab_ssd(str(staging), str(tar_path))

    colab_ssd = tmp_path / "colab_local_dataset"
    returned_tar = ensure_colab_dataset_ready(
        local_dataset_dir=str(colab_ssd),
        drive_root_dir=str(drive_root),
        update_state=False,
    )
    assert returned_tar == tar_path
    assert (colab_ssd / "multitalker_train_manifest.json").exists()

    # Outro corpus nunca reaproveita o pacote do FLEURS
    assert not (drive_root / "coraa_mupe" / "ptbr_dataset_16khz.tar.gz").exists()


def test_supported_hf_datasets_follow_source_contract():
    assert all(isinstance(src, HFDatasetSource) for src in SUPPORTED_HF_DATASETS.values())
    mupe = SUPPORTED_HF_DATASETS["coraa_mupe"]
    assert mupe.hf_path == "nilc-nlp/CORAA-MUPE-ASR"
    assert mupe.text_field == "original_text"
    assert mupe.speaker_fields == ["speaker_code", "speaker_type"]
    assert mupe.allowed_values == {"audio_quality": ["high"]}
    assert mupe.strip_annotations is True
    assert SUPPORTED_HF_DATASETS["coraa"].hub_split("validation") == "dev"
    assert SUPPORTED_HF_DATASETS["coraa"].hub_split("train") == "train"
    assert mupe.hub_split("validation") == "validation"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Muito pouco dinheiro. (risos) Foi difícil.", "Muito pouco dinheiro. Foi difícil."),
        ("Ele saiu () e voltou ( ) depois (pausa) .", "Ele saiu e voltou depois."),
        ("ãh\xa0 mesas fixas,  onde nós almoçávamos", "ãh mesas fixas, onde nós almoçávamos"),
        ("Então, mata um pouco a sauda>", None),
        ("< em 52. E a senhora saberia dizer?", None),
        ("(risos)", None),
    ],
)
def test_normalize_transcript_strips_mupe_annotations(raw, expected):
    assert normalize_transcript(raw, strip_annotations=True) == expected


def test_normalize_transcript_keeps_text_without_strip_annotations():
    # FLEURS pode ter parênteses legítimos: só NFC e colapso de espaços
    decomposed = "Organizac\u0327a\u0303o (OMS)  mundial"
    assert normalize_transcript(decomposed) == "Organização (OMS) mundial"


def _fake_mupe_item(text, speaker_code="MA_HV273", speaker_type="R", quality="high", seconds=1.0):
    t = np.arange(int(16000 * seconds), dtype=np.float32) / 16000.0
    return {
        "audio": {"array": 0.3 * np.sin(2 * np.pi * 220.0 * t), "sampling_rate": 16000},
        "original_text": text,
        "speaker_code": speaker_code,
        "speaker_type": speaker_type,
        "audio_quality": quality,
    }


def test_export_fake_mupe_dataset_filters_and_composes_speaker(tmp_path):
    items = [
        _fake_mupe_item("Durante 15 anos, mais ou menos. (risos)"),
        _fake_mupe_item("Áudio ruim.", quality="low"),
        _fake_mupe_item("E com dezess>"),
        _fake_mupe_item("E a senhora nasceu onde?", speaker_type="P/1"),
        _fake_mupe_item("Curto demais.", seconds=0.3),
    ]
    manifest = export_hf_dataset_to_nemo_manifest(
        dataset_key="coraa_mupe",
        split="train",
        output_dir=str(tmp_path),
        manifest_filename="clean_train_manifest.json",
        dataset=items,
    )
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert [r["text"] for r in records] == ["Durante 15 anos, mais ou menos.", "E a senhora nasceu onde?"]
    assert [r["speaker_id"] for r in records] == ["coraa_mupe_MA_HV273_R", "coraa_mupe_MA_HV273_P-1"]
    assert records[0]["audio_filepath"].endswith("coraa_mupe_train_0000000.wav")
    for r in records:
        MultitalkerManifestEntry.model_validate(r)


def test_export_val_stride_keeps_one_every_k(tmp_path):
    items = [_fake_mupe_item(f"Frase número {i}.") for i in range(10)]
    manifest = export_hf_dataset_to_nemo_manifest(
        dataset_key="coraa_mupe",
        split="validation",
        output_dir=str(tmp_path),
        manifest_filename="clean_val_manifest.json",
        stride=4,
        dataset=items,
    )
    texts = [json.loads(line)["text"] for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert texts == ["Frase número 0.", "Frase número 4.", "Frase número 8."]
