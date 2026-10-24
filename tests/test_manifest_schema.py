"""Testes de contrato para manifestos NeMo e configurações de estágio (Spec 01 e Spec 04)."""

import math
import pytest
from pydantic import ValidationError

from src.schemas import (
    EvaluationReport,
    FRAME_DURATION_SEC,
    MultitalkerManifestEntry,
    REQUIRED_ATT_CONTEXT_SIZE,
    SingleSpeakerManifestEntry,
    TrainingStageConfig,
    expected_frames_for_duration,
)
from src.data_prep import ensure_colab_dataset_ready, pack_dataset_for_colab_ssd, rewrite_manifest_audio_paths
from src.evaluator import compute_word_error_rate
from src.trainer import (
    apply_stage_optimizations_to_nemo_model,
    load_and_validate_stage_config,
    preflight_validate_manifest,
    resolve_nemo_checkpoint_path,
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

    drive_root = tmp_path / "drive_parakeet"
    tok_dir = drive_root / "tokenizer_ptbr_1024"
    tok_dir.mkdir(parents=True)
    (tok_dir / "tokenizer.vocab").write_text("a\t0\n", encoding="utf-8")

    tar_path = drive_root / "ptbr_dataset_16khz.tar.gz"
    pack_dataset_for_colab_ssd(str(staging), str(tar_path))

    colab_ssd = tmp_path / "colab_local_dataset"
    returned_tar = ensure_colab_dataset_ready(
        local_dataset_dir=str(colab_ssd),
        drive_root_dir=str(drive_root),
        update_state=False,
    )
    assert returned_tar == tar_path
    assert (colab_ssd / "multitalker_train_manifest.json").exists()
