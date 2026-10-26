"""Contratos Pydantic v2 derivados das Especificações (Specs 01 a 04)."""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

REQUIRED_SAMPLE_RATE: int = 16000
FRAME_DURATION_SEC: float = 0.08  # 80 ms por frame no codificador FastConformer
SAMPLES_PER_FRAME: int = 1280     # 16000 Hz * 0.08 s
MAX_PEAK_AMPLITUDE: float = 0.99
REQUIRED_VOCAB_SIZE: int = 1024
DEFAULT_MAX_DURATION_SEC: float = 20.0
HARD_MAX_DURATION_SEC: float = 30.0
REQUIRED_ATT_CONTEXT_SIZE: List[List[int]] = [[70, 0], [70, 1], [70, 6], [70, 13]]

CANONICAL_STEP_IDS: tuple[str, ...] = (
    "step_00_audit_and_contracts",
    "step_01_data_prep_clean",
    "step_02_tokenizer_ptbr",
    "step_03_mixer_multitalker",
    "step_04_pack_and_upload_drive",
    "step_05_colab_stage1_warmup",
    "step_06_colab_stage2_multitalker",
    "step_07_eval_and_export",
)

StepStatus = Literal["pending", "in_progress", "blocked", "completed", "failed"]
ExecutionEnv = Literal["local_cpu", "colab_cpu", "colab_a100", "user_action"]
OverallStatus = Literal["ready", "in_progress", "blocked_by_user", "failed", "completed"]


def expected_frames_for_duration(duration: float) -> int:
    """Calcula ceil(duration / 0.08) eliminando artefatos binários IEEE-754 (ex.: 0.56 / 0.08)."""
    return math.ceil(round(duration / FRAME_DURATION_SEC, 6))


class MultitalkerManifestEntry(BaseModel):
    """Contrato de uma linha do manifesto JSONL para o multitalker-parakeet (Spec 01 e Spec 02)."""

    audio_filepath: str = Field(..., min_length=1)
    duration: float = Field(..., ge=0.5, le=30.0)
    text: str = Field(..., min_length=1)
    spk_targets: List[int] = Field(..., min_length=1)
    bg_spk_targets: List[int] = Field(..., min_length=1)
    speaker_id: Optional[str] = None
    overlap_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    sir_db: Optional[float] = Field(default=None, ge=-20.0, le=20.0)

    @field_validator("spk_targets", "bg_spk_targets")
    @classmethod
    def validate_binary_mask_values(cls, v: List[int]) -> List[int]:
        if any(val not in (0, 1) for val in v):
            raise ValueError("As máscaras spk_targets e bg_spk_targets devem conter apenas valores binários (0 ou 1).")
        return v

    @model_validator(mode="after")
    def check_frame_alignment(self) -> "MultitalkerManifestEntry":
        expected_frames = expected_frames_for_duration(self.duration)
        if abs(len(self.spk_targets) - expected_frames) > 1:
            raise ValueError(
                f"Desalinhamento de máscara (Spec 01): esperado ~{expected_frames} frames "
                f"(80ms) para duração de {self.duration}s, mas recebeu {len(self.spk_targets)}."
            )
        if len(self.spk_targets) != len(self.bg_spk_targets):
            raise ValueError(
                f"spk_targets ({len(self.spk_targets)}) e bg_spk_targets ({len(self.bg_spk_targets)}) "
                "devem ter exatamente o mesmo comprimento."
            )
        return self


class SingleSpeakerManifestEntry(MultitalkerManifestEntry):
    """Contrato para amostra de locutor único em PT-BR no estágio de Warm-up (Spec 01)."""

    @classmethod
    def from_clean_audio(
        cls,
        audio_filepath: str,
        duration: float,
        text: str,
        speaker_id: Optional[str] = None,
    ) -> "SingleSpeakerManifestEntry":
        num_frames = expected_frames_for_duration(duration)
        return cls(
            audio_filepath=audio_filepath,
            duration=duration,
            text=text,
            speaker_id=speaker_id,
            spk_targets=[1] * num_frames,
            bg_spk_targets=[0] * num_frames,
            overlap_ratio=0.0,
        )


class SpeakerSupervision(BaseModel):
    """Intervalo de fala de um locutor dentro de um cut Lhotse da Fase 2 (Spec 02 §3.7)."""

    speaker: str = Field(..., min_length=1)
    start: float = Field(..., ge=0.0)
    duration: float = Field(..., gt=0.0)
    text: str = Field(..., min_length=1)


class MultitalkerCut(BaseModel):
    """MonoCut Lhotse com uma supervision por locutor, lido pelo LhotseSpeechToTextSpkBpeDataset (Spec 04 §8.2)."""

    id: str = Field(..., min_length=1)
    audio_filepath: str = Field(..., min_length=1)
    duration: float = Field(..., ge=0.5, le=30.0)
    supervisions: List[SpeakerSupervision] = Field(..., min_length=1, max_length=2)

    @model_validator(mode="after")
    def check_supervisions(self) -> "MultitalkerCut":
        speakers = [sup.speaker for sup in self.supervisions]
        if len(set(speakers)) != len(speakers):
            raise ValueError(f"Rótulos de locutor repetidos no cut {self.id}: {speakers} (Spec 02 §3.7).")
        for sup in self.supervisions:
            # Mesma tolerância do SupervisionSet.find() do Lhotse, que descarta supervisions além do cut
            if sup.start + sup.duration > self.duration + 1e-3:
                raise ValueError(
                    f"Supervision de {sup.speaker} termina em {sup.start + sup.duration:.4f}s, "
                    f"após a duração do cut {self.id} ({self.duration}s)."
                )
        return self

    def to_lhotse_dict(self, language: str = "pt") -> Dict[str, object]:
        """Serializa no formato de CutSet.from_file() (uma linha JSONL por cut)."""
        return {
            "id": self.id,
            "start": 0.0,
            "duration": self.duration,
            "channel": 0,
            "supervisions": [
                {
                    "id": f"{self.id}-{idx}",
                    "recording_id": self.id,
                    "start": sup.start,
                    "duration": sup.duration,
                    "channel": 0,
                    "text": sup.text,
                    "language": language,
                    "speaker": sup.speaker,
                }
                for idx, sup in enumerate(self.supervisions)
            ],
            "recording": {
                "id": self.id,
                "sources": [{"type": "file", "channels": [0], "source": self.audio_filepath}],
                "sampling_rate": REQUIRED_SAMPLE_RATE,
                "num_samples": int(round(self.duration * REQUIRED_SAMPLE_RATE)),
                "duration": self.duration,
                "channel_ids": [0],
            },
            # speaker_to_target() do NeMo chama cut.custom.get('rttm_filepath'): custom não pode ser None
            "custom": {},
            "type": "MonoCut",
        }


class HFDatasetSource(BaseModel):
    """Contrato de uma fonte de fala single-speaker PT-BR no Hugging Face (Spec 01 §2.1)."""

    hf_path: str
    hf_config: Optional[str] = None
    text_field: str
    split_aliases: Dict[str, str] = Field(default_factory=dict)
    speaker_fields: List[str] = Field(default_factory=list)
    allowed_values: Dict[str, List[str]] = Field(default_factory=dict)
    strip_annotations: bool = False

    def hub_split(self, split: str) -> str:
        """Nome do split no Hub para um split canônico do pipeline (ex.: validation -> dev)."""
        return self.split_aliases.get(split, split)


class MixerConfig(BaseModel):
    """Contrato de configuração do simulador de sobreposição sintética (Spec 02)."""

    sample_rate: Literal[16000] = REQUIRED_SAMPLE_RATE
    frame_duration_sec: float = Field(default=FRAME_DURATION_SEC)
    min_overlap_ratio: float = Field(default=0.15, ge=0.05, le=0.90)
    max_overlap_ratio: float = Field(default=0.50, ge=0.10, le=0.95)
    min_sir_db: float = Field(default=-5.0, ge=-15.0, le=0.0)
    max_sir_db: float = Field(default=5.0, ge=0.0, le=15.0)
    vad_rms_threshold: float = Field(default=0.01, gt=0.0, lt=0.5)
    max_peak_amplitude: float = Field(default=MAX_PEAK_AMPLITUDE, gt=0.5, le=0.99)
    max_duration_sec: float = Field(default=DEFAULT_MAX_DURATION_SEC, ge=1.0, le=HARD_MAX_DURATION_SEC)
    add_gaussian_noise: bool = False
    noise_snr_db_range: tuple[float, float] = (-40.0, -20.0)
    clean_ratio: float = Field(default=0.30, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_ranges(self) -> "MixerConfig":
        if self.min_overlap_ratio >= self.max_overlap_ratio:
            raise ValueError("min_overlap_ratio deve ser menor que max_overlap_ratio.")
        if self.min_sir_db > self.max_sir_db:
            raise ValueError("min_sir_db deve ser menor ou igual a max_sir_db.")
        if self.noise_snr_db_range[0] > self.noise_snr_db_range[1]:
            raise ValueError("noise_snr_db_range[0] deve ser menor ou igual a noise_snr_db_range[1].")
        return self


class TokenizerConfig(BaseModel):
    """Contrato de configuração do tokenizador SentencePiece BPE em PT-BR (Spec 03)."""

    vocab_size: int = Field(default=REQUIRED_VOCAB_SIZE)
    model_type: Literal["bpe"] = "bpe"
    character_coverage: float = Field(default=0.9999, ge=0.999, le=1.0)
    preserve_case_and_punctuation: bool = True

    @field_validator("vocab_size")
    @classmethod
    def enforce_compact_vocab(cls, v: int) -> int:
        if v != REQUIRED_VOCAB_SIZE:
            raise ValueError(
                f"Spec 03 exige vocab_size == {REQUIRED_VOCAB_SIZE} para não explodir a VRAM "
                f"da Joint Network 4D do RNN-T na GPU A100 (recebido: {v})."
            )
        return v


class TrainingStageConfig(BaseModel):
    """Contrato de configuração de estágio de treinamento NeMo na A100 (Spec 04)."""

    stage: Literal["warmup", "multitalker"]
    freeze_encoder: bool
    precision: Literal["bf16-mixed", "16-mixed"] = "bf16-mixed"
    bucketing_strategy: Literal["fully_randomized", "synced_randomized"] = "fully_randomized"
    fused_batch_size: int = Field(default=16, ge=4, le=64)
    resume_if_exists: bool = True
    vocab_size: int = REQUIRED_VOCAB_SIZE
    return_speaker_targets: bool = False
    speaker_kernel_enabled: bool = True
    att_context_size: Optional[List[List[int]]] = None
    use_lhotse: bool = True
    trainer_logger: bool = False
    ckpt_every_n_train_steps: Optional[int] = None
    ckpt_every_n_epochs: Optional[int] = None

    @model_validator(mode="after")
    def check_stage_invariants(self) -> "TrainingStageConfig":
        if not self.use_lhotse:
            raise ValueError(
                "Spec 04 §8.2: use_lhotse DEVE ser True; EncDecMultiTalkerRNNTBPEModel só cria dataloader via Lhotse."
            )
        if self.trainer_logger:
            raise ValueError(
                "Spec 04 §8.3: trainer.logger DEVE ser False; o exp_manager do NeMo cria o logger TensorBoard."
            )
        if self.ckpt_every_n_train_steps and self.ckpt_every_n_epochs:
            raise ValueError(
                "Spec 04 §4: every_n_train_steps e every_n_epochs são mutuamente exclusivos no ModelCheckpoint."
            )
        if self.stage == "warmup" and not self.freeze_encoder:
            raise ValueError(
                "Spec 04: No estágio 'warmup', freeze_encoder DEVE ser True para evitar "
                "esquecimento catastrófico do FastConformer e dos Speaker Kernels."
            )
        if self.stage == "multitalker":
            if self.freeze_encoder:
                raise ValueError(
                    "Spec 04: No estágio 'multitalker', freeze_encoder DEVE ser False para permitir "
                    "a adaptação acústica ao PT-BR com máscaras spk_targets."
                )
            if not self.return_speaker_targets:
                raise ValueError(
                    "Spec 04: No estágio 'multitalker', return_speaker_targets DEVE ser True."
                )
            if not self.speaker_kernel_enabled:
                raise ValueError(
                    "Spec 04: No estágio 'multitalker', speaker_kernel_enabled DEVE ser True."
                )
            if self.att_context_size is not None and self.att_context_size != REQUIRED_ATT_CONTEXT_SIZE:
                raise ValueError(
                    f"Spec 04: att_context_size na fase 'multitalker' deve ser {REQUIRED_ATT_CONTEXT_SIZE}, "
                    f"mas recebeu {self.att_context_size}."
                )
        if not self.resume_if_exists:
            raise ValueError("Spec 04: resume_if_exists DEVE ser True para tolerar quedas de sessão no Colab.")
        return self


class EvaluationReport(BaseModel):
    """Contrato do relatório final de avaliação WER limpo vs. sobreposto (Spec 04 / Spec 05 - Step 07)."""

    model_path: str = Field(..., min_length=1)
    clean_manifest_path: str = Field(..., min_length=1)
    multitalker_manifest_path: str = Field(..., min_length=1)
    clean_samples_evaluated: int = Field(..., ge=1)
    multitalker_samples_evaluated: int = Field(..., ge=1)
    clean_wer: float = Field(..., ge=0.0, le=2.0)
    multitalker_wer: float = Field(..., ge=0.0, le=2.0)
    # Sem caixa e pontuação (Spec 04 §6): as referências do CORAA-MUPE têm ambas
    clean_wer_normalized: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    multitalker_wer_normalized: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    # [esquerda, direita] em frames de 80 ms usado na decodificação (Spec 04 §6); None nos relatórios antigos
    att_context_size: Optional[List[int]] = Field(default=None, min_length=2, max_length=2)
    evaluated_at: str = Field(..., min_length=1)


class UserBlocker(BaseModel):
    """Impedimento que exige ação do usuário humano (Spec 05)."""

    id: str = Field(..., min_length=1)
    step_id: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    user_action_required: str = Field(..., min_length=1)
    created_at: str = Field(..., min_length=1)
    resolved: bool = False
    resolved_at: Optional[str] = None


class PipelineStepState(BaseModel):
    """Estado de uma etapa individual do pipeline SDD operado por agentes (Spec 05)."""

    step_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    environment: ExecutionEnv
    status: StepStatus = "pending"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    last_command: Optional[str] = None
    artifacts: Dict[str, str] = Field(default_factory=dict)
    metrics: Dict[str, str | float | int] = Field(default_factory=dict)
    notes: Optional[str] = None


class AgentHandoverContext(BaseModel):
    """Contexto determinístico de handover para o próximo agente sem contexto (Spec 05)."""

    last_agent_summary: str = Field(..., min_length=1)
    next_agent_action: str = Field(..., min_length=1)
    next_exact_command: str = Field(..., min_length=1)
    critical_warnings: List[str] = Field(default_factory=list)


class AgentDiscoveryEntry(BaseModel):
    """Registro de conhecimento incremental descoberto por um agente (Spec 05)."""

    id: str = Field(..., min_length=1)
    timestamp: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    problem: str = Field(..., min_length=1)
    solution: str = Field(..., min_length=1)


class PipelineState(BaseModel):
    """Contrato completo da máquina de estados do pipeline entre sessões de agentes (Spec 05)."""

    project_name: str = "parakeet-ptbr-multitalker"
    model_target: str = "nvidia/multitalker-parakeet-streaming-0.6b-v1"
    updated_at: str = Field(..., min_length=1)
    overall_status: OverallStatus = "ready"
    current_step_id: str = Field(..., min_length=1)
    steps: List[PipelineStepState]
    blockers: List[UserBlocker] = Field(default_factory=list)
    agent_handover: AgentHandoverContext
    discoveries: List[AgentDiscoveryEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_pipeline_invariants(self) -> "PipelineState":
        step_ids = tuple(s.step_id for s in self.steps)
        if step_ids != CANONICAL_STEP_IDS:
            raise ValueError(
                f"Spec 05 violada: a lista de etapas deve corresponder exatamente a {CANONICAL_STEP_IDS}, "
                f"mas recebeu {step_ids}."
            )
        if self.current_step_id not in CANONICAL_STEP_IDS:
            raise ValueError(
                f"Spec 05 violada: current_step_id '{self.current_step_id}' não pertence a {CANONICAL_STEP_IDS}."
            )
        active_blockers = [b for b in self.blockers if not b.resolved]
        if self.overall_status == "blocked_by_user" and not active_blockers:
            raise ValueError(
                "Spec 05 violada: overall_status == 'blocked_by_user' exige pelo menos 1 UserBlocker ativo (resolved=False)."
            )
        if self.overall_status == "failed" and not any(s.status == "failed" for s in self.steps):
            raise ValueError(
                "Spec 05 violada: overall_status == 'failed' exige pelo menos 1 etapa com status == 'failed'."
            )
        return self


# ---------------------------------------------------------------------------
# Spec 06: Operação remota não-bloqueante do Google Colab (src.colab_guard)
# ---------------------------------------------------------------------------

ColabGuardStatus = Literal[
    "ok",
    "remote_error",
    "needs_user",
    "timeout",
    "idle_timeout",
    "session_lost",
    "cli_error",
    "job_running",
    "job_not_found",
]

# Códigos de saída determinísticos (devem coincidir com src/colab_cli_bridge.py).
COLAB_GUARD_EXIT_CODES: Dict[str, int] = {
    "ok": 0,
    "remote_error": 1,
    "needs_user": 10,
    "timeout": 11,
    "idle_timeout": 12,
    "session_lost": 13,
    "cli_error": 14,
    "job_running": 20,
    "job_not_found": 21,
}

DEFAULT_SYNC_EXEC_TIMEOUT_SEC: float = 120.0
MAX_SYNC_EXEC_TIMEOUT_SEC: float = 600.0
DEFAULT_CLI_PASSTHROUGH_TIMEOUT_SEC: float = 300.0


class ColabGuardOutcome(BaseModel):
    """Resultado determinístico de qualquer chamada remota ao Colab feita por agentes (Spec 06)."""

    status: ColabGuardStatus
    exit_code: int
    command: str = Field(..., min_length=1)
    elapsed_sec: float = Field(..., ge=0.0)
    reason: Optional[str] = None
    user_action: Optional[str] = None
    auth_url: Optional[str] = None
    blocker_id: Optional[str] = None
    output_tail: str = ""
    data: Dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_exit_code(self) -> "ColabGuardOutcome":
        expected = COLAB_GUARD_EXIT_CODES[self.status]
        if self.exit_code != expected:
            raise ValueError(
                f"Spec 06 violada: status '{self.status}' exige exit_code={expected}, mas recebeu {self.exit_code}."
            )
        if self.status == "needs_user" and not (self.user_action or self.auth_url):
            raise ValueError("Spec 06 violada: status 'needs_user' exige user_action ou auth_url para o humano.")
        return self
