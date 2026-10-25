"""Testes de contrato para o simulador de mistura sintética e máscaras de 80ms (Spec 02)."""

import json

import numpy as np
import pytest

import soundfile as sf
from src.mixer import (
    ensure_mono_16khz,
    export_lhotse_cuts,
    generate_mixed_dataset_from_manifest,
    mix_two_speakers,
    prevent_clipping,
)
from src.schemas import (
    MAX_PEAK_AMPLITUDE,
    MixerConfig,
    REQUIRED_SAMPLE_RATE,
    SingleSpeakerManifestEntry,
)


def _generate_sine_voice(duration_sec: float, freq_hz: float = 220.0, amp: float = 0.8) -> np.ndarray:
    t = np.linspace(0.0, duration_sec, int(REQUIRED_SAMPLE_RATE * duration_sec), endpoint=False)
    return (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def test_mixer_prevents_clipping_and_mirrors_masks(tmp_path):
    # Ambos com amplitude máxima 1.0 para forçar colisão acima de 1.0 antes do anti-clipping
    audio_a = _generate_sine_voice(2.4, freq_hz=200.0, amp=1.0)
    audio_b = _generate_sine_voice(2.0, freq_hz=350.0, amp=1.0)

    out_wav = tmp_path / "mixed_test.wav"
    mixed, entries = mix_two_speakers(
        audio_a=audio_a,
        text_a="Bom dia pessoal, hoje vamos revisar os resultados.",
        audio_b=audio_b,
        text_b="Desculpa interromper, mas o relatório já saiu?",
        output_wav_path=str(out_wav),
        offset_sec=1.0,
        sir_db=0.0,
        write_wav=True,
    )

    # 1. Verifica ausência de clipping acústico (pico <= 0.99)
    assert float(np.max(np.abs(mixed))) <= MAX_PEAK_AMPLITUDE + 1e-5
    assert out_wav.exists()

    # 2. Verifica espelhamento A/B (2 amostras geradas para 1 arquivo WAV)
    assert len(entries) == 2
    entry_a, entry_b = entries
    assert entry_a.audio_filepath == entry_b.audio_filepath
    assert entry_a.text == "Bom dia pessoal, hoje vamos revisar os resultados."
    assert entry_b.text == "Desculpa interromper, mas o relatório já saiu?"

    # 3. As máscaras do Alvo A e do Alvo B devem ser exatamente inversas entre si
    assert entry_a.spk_targets == entry_b.bg_spk_targets
    assert entry_a.bg_spk_targets == entry_b.spk_targets

    # 4. Verifica sobreposição real entre 1.0s e 2.4s
    assert entry_a.overlap_ratio is not None and entry_a.overlap_ratio > 0.2


def test_ensure_mono_16khz_rejects_wrong_sample_rate():
    dummy = np.zeros(8000, dtype=np.float32)
    with pytest.raises(ValueError, match="16000 Hz"):
        ensure_mono_16khz(dummy, sample_rate=8000)


def test_prevent_clipping_scales_down_loud_signal():
    loud = np.array([-2.5, 0.0, 2.5], dtype=np.float32)
    scaled = prevent_clipping(loud, max_peak=0.99)
    assert np.isclose(np.max(np.abs(scaled)), 0.99, atol=1e-5)


def test_mixer_caps_duration_and_supports_gaussian_noise(tmp_path):
    audio_a = _generate_sine_voice(15.0, freq_hz=200.0, amp=0.5)
    audio_b = _generate_sine_voice(15.0, freq_hz=300.0, amp=0.5)
    cfg = MixerConfig(max_duration_sec=20.0, add_gaussian_noise=True)

    mixed, entries = mix_two_speakers(
        audio_a=audio_a,
        text_a="Locutor A longo.",
        audio_b=audio_b,
        text_b="Locutor B longo.",
        output_wav_path=str(tmp_path / "capped.wav"),
        config=cfg,
        offset_sec=12.0,
        write_wav=False,
        speaker_id_a="spk_A",
        speaker_id_b="spk_B",
    )
    assert entries[0].duration <= 20.0
    assert entries[0].speaker_id == "spk_A"
    assert entries[1].speaker_id == "spk_B"
    assert float(np.max(np.abs(mixed))) <= MAX_PEAK_AMPLITUDE + 1e-5


def test_generate_mixed_dataset_enforces_distinct_speakers_and_clean_ratio(tmp_path):
    wav_dir = tmp_path / "clean_wavs"
    wav_dir.mkdir(parents=True)
    clean_manifest = tmp_path / "clean_manifest.json"

    lines = []
    for idx, spk in enumerate(["spk_1", "spk_1", "spk_2", "spk_2"]):
        wav_path = wav_dir / f"c_{idx}.wav"
        tone = _generate_sine_voice(1.6, freq_hz=200.0 + idx * 40.0, amp=0.5)
        sf.write(str(wav_path), tone, REQUIRED_SAMPLE_RATE, subtype="PCM_16")
        entry = SingleSpeakerManifestEntry.from_clean_audio(
            audio_filepath=str(wav_path),
            duration=1.6,
            text=f"Frase número {idx} do {spk}.",
            speaker_id=spk,
        )
        lines.append(entry.model_dump_json())

    clean_manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out_dir = tmp_path / "mixed_out"
    out_manifest = out_dir / "multitalker_manifest.json"
    # 7 misturas = 14 entradas sobrepostas; com clean_ratio=0.30, adiciona 6 entradas limpas -> total 20
    total = generate_mixed_dataset_from_manifest(
        clean_manifest_path=str(clean_manifest),
        output_dir=str(out_dir),
        output_manifest_path=str(out_manifest),
        num_mixtures=7,
        clean_ratio=0.30,
        seed=123,
    )
    assert total == 20
    out_lines = [l for l in out_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(out_lines) == 20

    # Testa retomada idempotente (resume=True): pedir 7 misturas novamente não deve duplicar
    total_resumed = generate_mixed_dataset_from_manifest(
        clean_manifest_path=str(clean_manifest),
        output_dir=str(out_dir),
        output_manifest_path=str(out_manifest),
        num_mixtures=7,
        clean_ratio=0.30,
        seed=123,
        resume=True,
    )
    assert total_resumed == 20
    assert len([l for l in out_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]) == 20


def _mixed_pair_entries(tmp_path, speaker_id_a="spk_A", speaker_id_b="spk_B"):
    # A ocupa 0-2.4s; B entra em 1.0s (frame 12 = 0.96-1.04s) e termina em 3.0s
    _, entries = mix_two_speakers(
        audio_a=_generate_sine_voice(2.4, freq_hz=200.0, amp=0.5),
        text_a="Bom dia pessoal.",
        audio_b=_generate_sine_voice(2.0, freq_hz=350.0, amp=0.5),
        text_b="O relatório já saiu?",
        output_wav_path=str(tmp_path / "wavs" / "mix_000000.wav"),
        offset_sec=1.0,
        sir_db=0.0,
        write_wav=False,
        speaker_id_a=speaker_id_a,
        speaker_id_b=speaker_id_b,
    )
    return entries


def _write_manifest(path, entries):
    path.write_text("".join(e.model_dump_json() + "\n" for e in entries), encoding="utf-8")


def test_export_lhotse_cuts_merges_mirrored_pairs_and_keeps_clean_lines(tmp_path):
    clean = SingleSpeakerManifestEntry.from_clean_audio(
        audio_filepath=str(tmp_path / "wavs" / "coraa_mupe_train_0000001.wav"),
        duration=1.6,
        text="Fala limpa.",
        speaker_id="spk_C",
    )
    manifest = tmp_path / "multitalker_train_manifest.json"
    # A mesma linha limpa repetida (sorteio com reposição) deve virar 2 cuts distintos
    _write_manifest(manifest, _mixed_pair_entries(tmp_path) + [clean, clean])

    cuts_path, num_cuts = export_lhotse_cuts(str(manifest))
    cuts = [json.loads(line) for line in open(cuts_path, encoding="utf-8")]

    assert cuts_path.endswith("multitalker_train_manifest_cuts.jsonl")
    # 1 cut por linha do manifesto: o par espelhado vira 2 cuts (composição 70/30 por amostra, Spec 02 §3.7)
    assert num_cuts == len(cuts) == 4
    assert len({c["id"] for c in cuts}) == 4

    for mix_cut in cuts[:2]:
        assert mix_cut["type"] == "MonoCut"
        assert mix_cut["custom"] == {}
        assert mix_cut["duration"] == 3.0
        assert mix_cut["recording"]["sources"][0]["source"].endswith("mix_000000.wav")
        assert mix_cut["recording"]["num_samples"] == 48000
        sup_a, sup_b = mix_cut["supervisions"]
        assert (sup_a["speaker"], sup_a["text"], sup_a["start"], sup_a["duration"]) == ("spk_A", "Bom dia pessoal.", 0.0, 2.4)
        assert (sup_b["speaker"], sup_b["text"], sup_b["start"], sup_b["duration"]) == ("spk_B", "O relatório já saiu?", 0.96, 2.04)
        assert all(s["recording_id"] == mix_cut["recording"]["id"] and s["language"] == "pt" for s in (sup_a, sup_b))

    for clean_cut in cuts[2:]:
        assert len(clean_cut["supervisions"]) == 1
        sup = clean_cut["supervisions"][0]
        assert (sup["speaker"], sup["start"], sup["duration"]) == ("spk_C", 0.0, 1.6)


def test_export_lhotse_cuts_relabels_missing_or_equal_speakers(tmp_path):
    manifest = tmp_path / "multitalker_val_manifest.json"
    _write_manifest(manifest, _mixed_pair_entries(tmp_path, speaker_id_a="spk_X", speaker_id_b="spk_X"))

    cuts_path, _ = export_lhotse_cuts(str(manifest), cuts_path=str(tmp_path / "val_cuts.jsonl"))
    cut = json.loads(open(cuts_path, encoding="utf-8").readline())
    assert [s["speaker"] for s in cut["supervisions"]] == ["spk_a", "spk_b"]


def test_export_lhotse_cuts_rejects_unpaired_mix_line(tmp_path):
    manifest = tmp_path / "multitalker_train_manifest.json"
    _write_manifest(manifest, _mixed_pair_entries(tmp_path)[:1])
    with pytest.raises(ValueError, match="par espelhado"):
        export_lhotse_cuts(str(manifest))


def test_train_and_val_mixtures_in_same_dir_do_not_overwrite(tmp_path):
    clean_manifest = tmp_path / "clean_manifest.json"
    lines = []
    for idx, spk in enumerate(["spk_1", "spk_2", "spk_3"]):
        wav_path = tmp_path / f"c_{idx}.wav"
        sf.write(str(wav_path), _generate_sine_voice(1.0 + 0.4 * idx, freq_hz=200.0 + 50.0 * idx, amp=0.5), REQUIRED_SAMPLE_RATE)
        lines.append(
            SingleSpeakerManifestEntry.from_clean_audio(
                audio_filepath=str(wav_path), duration=1.0 + 0.4 * idx, text=f"Frase {idx}.", speaker_id=spk
            ).model_dump_json()
        )
    clean_manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out_dir = tmp_path / "local_dataset"
    manifests = {}
    for split, seed in (("train", 1), ("val", 2)):
        manifests[split] = out_dir / f"multitalker_{split}_manifest.json"
        generate_mixed_dataset_from_manifest(
            clean_manifest_path=str(clean_manifest),
            output_dir=str(out_dir),
            output_manifest_path=str(manifests[split]),
            num_mixtures=3,
            seed=seed,
        )

    paths = {
        split: [json.loads(l)["audio_filepath"] for l in m.read_text(encoding="utf-8").splitlines()]
        for split, m in manifests.items()
    }
    assert set(paths["train"]).isdisjoint(paths["val"])
    assert all(p.split("/")[-1].startswith("mix_multitalker_train_manifest_") for p in paths["train"])
    assert len(list((out_dir / "wavs").glob("mix_*.wav"))) == 6
    # A duração gravada no manifesto de treino continua batendo com o WAV em disco
    for line in manifests["train"].read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        assert abs(sf.info(rec["audio_filepath"]).duration - rec["duration"]) < 1e-3
