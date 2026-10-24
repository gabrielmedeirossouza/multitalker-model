"""Gerenciador Stateful de Progresso, Handover entre Agentes, USER.md e Conhecimento Incremental (Spec 05)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.schemas import (
    AgentDiscoveryEntry,
    AgentHandoverContext,
    CANONICAL_STEP_IDS,
    PipelineState,
    PipelineStepState,
    UserBlocker,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATE_PATH = REPO_ROOT / "state" / "pipeline_state.json"
DEFAULT_KNOWLEDGE_PATH = REPO_ROOT / "state" / "AGENT_KNOWLEDGE.md"
DEFAULT_USER_MD_PATH = REPO_ROOT / "USER.md"
DEFAULT_COLAB_DRIVE_STATE_DIR = Path("/content/drive/MyDrive/parakeet_ptbr/state")

STATUS_BADGE = {
    "pending": "⏳ Pendente",
    "in_progress": "🔄 Em Execução",
    "blocked": "🛑 Bloqueado (Ação do Usuário)",
    "completed": "✅ Concluído",
    "failed": "❌ Falhou",
}

OVERALL_STATUS_BADGE = {
    "ready": "🟢 Pronto para Iniciar / Continuar",
    "in_progress": "🔄 Em Execução por Agente",
    "blocked_by_user": "🛑 Aguardando Ação do Usuário",
    "failed": "❌ Falha Detectada (Requer Correção)",
    "completed": "✅ Pipeline Finalizado",
}


def utc_now_iso() -> str:
    """Retorna timestamp ISO-8601 UTC."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_drive_mirror_dir() -> Optional[Path]:
    """
    Retorna o diretório de espelhamento no Google Drive quando definido via
    PARAKEET_DRIVE_STATE_DIR ou quando /content/drive/MyDrive está montado no Colab (Spec 05).
    """
    env_dir = os.environ.get("PARAKEET_DRIVE_STATE_DIR")
    if env_dir:
        return Path(env_dir)
    if Path("/content/drive/MyDrive").exists():
        return DEFAULT_COLAB_DRIVE_STATE_DIR
    return None


def _mirror_state_to_drive_if_mounted(
    state_path: Path,
    user_md_path: Path,
    knowledge_path: Path = DEFAULT_KNOWLEDGE_PATH,
) -> Optional[Path]:
    """Espelha os 3 arquivos de estado no Google Drive para sobreviver a desconexões do Colab."""
    mirror_dir = _get_drive_mirror_dir()
    if mirror_dir is None:
        return None
    mirror_dir.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        shutil.copy2(state_path, mirror_dir / "pipeline_state.json")
    if user_md_path.exists():
        shutil.copy2(user_md_path, mirror_dir / "USER.md")
    if knowledge_path.exists():
        shutil.copy2(knowledge_path, mirror_dir / "AGENT_KNOWLEDGE.md")
    return mirror_dir


def sync_state_from_drive(
    drive_state_dir: Optional[Path] = None,
    state_path: Path = DEFAULT_STATE_PATH,
    user_md_path: Path = DEFAULT_USER_MD_PATH,
    knowledge_path: Path = DEFAULT_KNOWLEDGE_PATH,
) -> bool:
    """Restaura o estado persistido no Google Drive para a VM efêmera do Colab no início da sessão."""
    src_dir = drive_state_dir or _get_drive_mirror_dir() or DEFAULT_COLAB_DRIVE_STATE_DIR
    drive_state_file = src_dir / "pipeline_state.json"
    if not drive_state_file.exists():
        return False

    raw = json.loads(drive_state_file.read_text(encoding="utf-8"))
    validated = PipelineState.model_validate(raw)

    if (src_dir / "AGENT_KNOWLEDGE.md").exists():
        knowledge_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_dir / "AGENT_KNOWLEDGE.md", knowledge_path)

    save_pipeline_state(validated, state_path=state_path, user_md_path=user_md_path)
    return True


def create_initial_pipeline_state() -> PipelineState:
    """Cria o estado inicial canônico do pipeline SDD (Spec 05)."""
    now = utc_now_iso()
    steps = [
        PipelineStepState(
            step_id="step_00_audit_and_contracts",
            title="Auditoria SDD, Specs (00-05), Contratos Pydantic e Gatekeeper CPU (< 0.3 GB local)",
            environment="local_cpu",
            status="completed",
            started_at=now,
            completed_at=now,
            last_command=".venv/bin/pytest tests/ -v",
            artifacts={
                "specs": "specs/00..05_*.md",
                "schemas": "src/schemas.py",
                "state_file": "state/pipeline_state.json",
                "knowledge_base": "state/AGENT_KNOWLEDGE.md",
            },
            metrics={"contract_tests_passed": 28},
            notes="Repositório auditado (incluindo LLM-as-a-Judge); fluxo 100% Colab+Drive sem download de áudio local.",
        ),
        PipelineStepState(
            step_id="step_01_data_prep_clean",
            title="Download e Padronização 16kHz Mono dos Datasets PT-BR direto no Colab SSD",
            environment="colab_cpu",
            status="pending",
        ),
        PipelineStepState(
            step_id="step_02_tokenizer_ptbr",
            title="Treinamento e Validação do Tokenizador SentencePiece BPE PT-BR (1.024 tokens) no Colab/Drive",
            environment="colab_cpu",
            status="pending",
        ),
        PipelineStepState(
            step_id="step_03_mixer_multitalker",
            title="Geração no Colab SSD do Dataset Sintético Sobreposto (70% Multitalker + 30% Limpo, 80ms)",
            environment="colab_cpu",
            status="pending",
        ),
        PipelineStepState(
            step_id="step_04_pack_and_upload_drive",
            title="Empacotamento de /content/local_dataset em ptbr_dataset_16khz.tar.gz no Google Drive",
            environment="colab_cpu",
            status="pending",
        ),
        PipelineStepState(
            step_id="step_05_colab_stage1_warmup",
            title="Treinamento Fase 1 (Warm-up: Encoder Congelado, Decoder/Joint PT-BR) na GPU A100",
            environment="colab_a100",
            status="pending",
        ),
        PipelineStepState(
            step_id="step_06_colab_stage2_multitalker",
            title="Treinamento Fase 2 (Multitalker Cache-Aware + Speaker Kernels) na GPU A100",
            environment="colab_a100",
            status="pending",
        ),
        PipelineStepState(
            step_id="step_07_eval_and_export",
            title="Avaliação Final (WER Limpo e Sobreposto) e Exportação do Checkpoint .nemo",
            environment="colab_a100",
            status="pending",
        ),
    ]
    return PipelineState(
        updated_at=now,
        overall_status="ready",
        current_step_id="step_01_data_prep_clean",
        steps=steps,
        blockers=[],
        agent_handover=AgentHandoverContext(
            last_agent_summary=(
                "Auditoria completa e configuração 100% Colab+Drive concluídas (step_00). "
                "Zero GB de áudio será baixado no Mac local; Steps 01 a 04 rodam direto na VM do Colab via --ensure_colab_dataset."
            ),
            next_agent_action=(
                "No Google Colab (Célula 3 do colab_runner.ipynb), executar --ensure_colab_dataset para "
                "baixar/normalizar em /content/local_dataset, treinar o tokenizador, gerar misturas 70/30 e salvar o .tar.gz no Google Drive."
            ),
            next_exact_command=(
                "python -m src.data_prep --ensure_colab_dataset --dataset fleurs_pt "
                "--local_dataset_dir /content/local_dataset "
                "--drive_root_dir /content/drive/MyDrive/parakeet_ptbr --update_state"
            ),
            critical_warnings=[
                "NUNCA baixe datasets de áudio reais no Mac local; use sempre a VM do Colab (/content/local_dataset) + Google Drive.",
                "No ambiente local (Mac), use sempre os binários `.venv/bin/python` e `.venv/bin/pytest`.",
                "Mantenha `max_duration <= 20.0` segundos para evitar descarte ou OOM na Joint Network 4D na A100.",
                "Nunca passe vírgula em string separada por vírgulas para `user_defined_symbols` do SentencePiece.",
            ],
        ),
        discoveries=[],
    )


def render_user_markdown(state: PipelineState) -> str:
    """Renderiza deterministicamente o arquivo USER.md sem prosa supérflua (Spec 05)."""
    active_blockers = [b for b in state.blockers if not b.resolved]
    current_step = next(s for s in state.steps if s.step_id == state.current_step_id)

    lines: List[str] = [
        "# USER.md — Painel de Status e Ações do Usuário",
        "",
        "> Arquivo gerado e sincronizado automaticamente via `python -m src.state`. Sem prosa desnecessária.",
        "",
        "## 1. Resumo Executivo",
        "",
        f"- **Projeto:** `{state.project_name}` (`{state.model_target}`)",
        f"- **Status Geral:** {OVERALL_STATUS_BADGE.get(state.overall_status, state.overall_status)}",
        f"- **Etapa Atual:** `{current_step.step_id}` — {current_step.title} (`{current_step.environment}`)",
        f"- **Última Atualização (UTC):** `{state.updated_at}`",
        "",
        "## 2. O Que o Usuário Precisa Fazer Agora (Impedimentos / Ações)",
        "",
    ]

    if active_blockers:
        lines.append("> [!IMPORTANT]")
        lines.append("> Existem impedimentos ativos que exigem ação humana antes do próximo passo de GPU/Rede:")
        lines.append("")
        lines.append("| ID | Etapa | Impedimento | Ação Necessária do Usuário |")
        lines.append("|---|---|---|---|")
        for b in active_blockers:
            lines.append(f"| `{b.id}` | `{b.step_id}` | {b.description} | **{b.user_action_required}** |")
    else:
        lines.append("- **Nenhum impedimento bloqueante no momento.** Os agentes de IA podem prosseguir com a etapa atual.")
        lines.append(
            "- **Pré-requisitos futuros (Etapas 04 a 07):** Ter conta Google Colab Pro/Pro+ com GPU **NVIDIA A100** e espaço no Google Drive (`/content/drive/MyDrive/parakeet_ptbr`)."
        )

    lines.extend(
        [
            "",
            "## 3. Progresso das Etapas (Pipeline Stateful)",
            "",
            "| # | ID da Etapa | Ambiente | Status | Artefatos / Métricas |",
            "|---|---|---|---|---|",
        ]
    )

    for idx, step in enumerate(state.steps):
        badge = STATUS_BADGE.get(step.status, step.status)
        summary_items: List[str] = []
        for k, v in step.metrics.items():
            summary_items.append(f"`{k}={v}`")
        for k, v in step.artifacts.items():
            summary_items.append(f"{k}: `{v}`")
        if step.status == "failed" and step.notes:
            summary_items.append(f"erro: `{step.notes}`")
        summary_str = ", ".join(summary_items) if summary_items else "—"
        lines.append(f"| {idx} | `{step.step_id}` | `{step.environment}` | {badge} | {summary_str} |")

    lines.extend(
        [
            "",
            "## 4. Handover para o Próximo Agente de IA",
            "",
            f"- **Resumo da Última Sessão:** {state.agent_handover.last_agent_summary}",
            f"- **Próxima Ação do Agente:** {state.agent_handover.next_agent_action}",
            "- **Comando Exato para o Próximo Agente:**",
            "  ```bash",
            f"  {state.agent_handover.next_exact_command}",
            "  ```",
        ]
    )

    if state.agent_handover.critical_warnings:
        lines.append("- **Alertas Técnicos Críticos:**")
        for warn in state.agent_handover.critical_warnings:
            lines.append(f"  - {warn}")

    lines.append("")
    return "\n".join(lines)


def load_pipeline_state(state_path: Path = DEFAULT_STATE_PATH) -> PipelineState:
    """Carrega e valida o state/pipeline_state.json."""
    with open(state_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return PipelineState.model_validate(data)


def save_pipeline_state(
    state: PipelineState,
    state_path: Path = DEFAULT_STATE_PATH,
    user_md_path: Path = DEFAULT_USER_MD_PATH,
) -> None:
    """Salva o estado validado, regenera USER.md e espelha no Google Drive quando montado."""
    state.updated_at = utc_now_iso()
    state = PipelineState.model_validate(state.model_dump())

    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(state.model_dump(), indent=2, ensure_ascii=False) + "\n")

    user_md_content = render_user_markdown(state)
    user_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(user_md_path, "w", encoding="utf-8") as uf:
        uf.write(user_md_content)

    _mirror_state_to_drive_if_mounted(state_path=state_path, user_md_path=user_md_path)


def start_step(
    step_id: str,
    command: Optional[str] = None,
    notes: Optional[str] = None,
    state_path: Path = DEFAULT_STATE_PATH,
    user_md_path: Path = DEFAULT_USER_MD_PATH,
) -> PipelineState:
    """Marca uma etapa como 'in_progress' e atualiza o estado e USER.md."""
    state = load_pipeline_state(state_path)
    if step_id not in CANONICAL_STEP_IDS:
        raise ValueError(f"Etapa inválida: {step_id}")

    now = utc_now_iso()
    for step in state.steps:
        if step.step_id == step_id:
            step.status = "in_progress"
            step.started_at = step.started_at or now
            if command:
                step.last_command = command
            if notes:
                step.notes = notes
            break

    state.current_step_id = step_id
    active_blockers = [b for b in state.blockers if not b.resolved]
    state.overall_status = "blocked_by_user" if active_blockers else "in_progress"
    save_pipeline_state(state, state_path, user_md_path)
    return state


def update_handover(
    step_id: str,
    next_action: str,
    next_command: str,
    last_summary: Optional[str] = None,
    artifacts: Optional[Dict[str, str]] = None,
    metrics: Optional[Dict[str, str | float | int]] = None,
    notes: Optional[str] = None,
    state_path: Path = DEFAULT_STATE_PATH,
    user_md_path: Path = DEFAULT_USER_MD_PATH,
) -> PipelineState:
    """
    Atualiza o handover e artefatos de uma etapa parcialmente concluída (mantendo-a 'in_progress')
    para que o próximo agente continue exatamente do ponto pendente (ex.: train feito, falta val).
    """
    state = load_pipeline_state(state_path)
    if step_id not in CANONICAL_STEP_IDS:
        raise ValueError(f"Etapa inválida: {step_id}")

    now = utc_now_iso()
    for step in state.steps:
        if step.step_id == step_id:
            step.status = "in_progress"
            step.started_at = step.started_at or now
            if artifacts:
                step.artifacts.update(artifacts)
            if metrics:
                step.metrics.update(metrics)
            if notes:
                step.notes = notes
            break

    state.current_step_id = step_id
    if last_summary:
        state.agent_handover.last_agent_summary = last_summary
    state.agent_handover.next_agent_action = next_action
    state.agent_handover.next_exact_command = next_command

    active_blockers = [b for b in state.blockers if not b.resolved]
    state.overall_status = "blocked_by_user" if active_blockers else "in_progress"
    save_pipeline_state(state, state_path, user_md_path)
    return state


def fail_step(
    step_id: str,
    error_message: str,
    command: Optional[str] = None,
    next_action: Optional[str] = None,
    state_path: Path = DEFAULT_STATE_PATH,
    user_md_path: Path = DEFAULT_USER_MD_PATH,
) -> PipelineState:
    """Marca uma etapa como 'failed', registra o diagnóstico de erro e atualiza USER.md (Spec 05)."""
    state = load_pipeline_state(state_path)
    if step_id not in CANONICAL_STEP_IDS:
        raise ValueError(f"Etapa inválida: {step_id}")

    for step in state.steps:
        if step.step_id == step_id:
            step.status = "failed"
            step.notes = error_message
            if command:
                step.last_command = command
            break

    state.current_step_id = step_id
    state.overall_status = "failed"
    state.agent_handover.last_agent_summary = f"Falha na etapa {step_id}: {error_message}"
    if next_action:
        state.agent_handover.next_agent_action = next_action
    else:
        state.agent_handover.next_agent_action = (
            f"Diagnosticar e corrigir a falha em {step_id} ({error_message}) e reexecutar o comando."
        )
    if command:
        state.agent_handover.next_exact_command = command

    save_pipeline_state(state, state_path, user_md_path)
    return state


def complete_step(
    step_id: str,
    next_step_id: Optional[str] = None,
    next_action: Optional[str] = None,
    next_command: Optional[str] = None,
    last_summary: Optional[str] = None,
    artifacts: Optional[Dict[str, str]] = None,
    metrics: Optional[Dict[str, str | float | int]] = None,
    notes: Optional[str] = None,
    state_path: Path = DEFAULT_STATE_PATH,
    user_md_path: Path = DEFAULT_USER_MD_PATH,
) -> PipelineState:
    """Marca uma etapa como 'completed', avança current_step_id e sincroniza USER.md."""
    state = load_pipeline_state(state_path)
    if step_id not in CANONICAL_STEP_IDS:
        raise ValueError(f"Etapa inválida: {step_id}")

    now = utc_now_iso()
    for step in state.steps:
        if step.step_id == step_id:
            step.status = "completed"
            step.started_at = step.started_at or now
            step.completed_at = now
            if artifacts:
                step.artifacts.update(artifacts)
            if metrics:
                step.metrics.update(metrics)
            if notes:
                step.notes = notes
            break

    if next_step_id:
        state.current_step_id = next_step_id
    else:
        idx = CANONICAL_STEP_IDS.index(step_id)
        if idx + 1 < len(CANONICAL_STEP_IDS):
            state.current_step_id = CANONICAL_STEP_IDS[idx + 1]

    if last_summary:
        state.agent_handover.last_agent_summary = last_summary
    if next_action:
        state.agent_handover.next_agent_action = next_action
    if next_command:
        state.agent_handover.next_exact_command = next_command

    all_done = all(s.status == "completed" for s in state.steps)
    active_blockers = [b for b in state.blockers if not b.resolved]
    if all_done:
        state.overall_status = "completed"
    elif active_blockers:
        state.overall_status = "blocked_by_user"
    else:
        state.overall_status = "ready"

    save_pipeline_state(state, state_path, user_md_path)
    return state


def add_blocker(
    blocker_id: str,
    step_id: str,
    description: str,
    user_action_required: str,
    state_path: Path = DEFAULT_STATE_PATH,
    user_md_path: Path = DEFAULT_USER_MD_PATH,
) -> PipelineState:
    """Registra um impedimento que exige ação do usuário e atualiza USER.md imediatamente."""
    state = load_pipeline_state(state_path)
    now = utc_now_iso()
    state.blockers = [b for b in state.blockers if b.id != blocker_id]
    state.blockers.append(
        UserBlocker(
            id=blocker_id,
            step_id=step_id,
            description=description,
            user_action_required=user_action_required,
            created_at=now,
            resolved=False,
        )
    )
    for step in state.steps:
        if step.step_id == step_id and step.status != "completed":
            step.status = "blocked"
    state.overall_status = "blocked_by_user"
    save_pipeline_state(state, state_path, user_md_path)
    return state


def resolve_blocker(
    blocker_id: str,
    state_path: Path = DEFAULT_STATE_PATH,
    user_md_path: Path = DEFAULT_USER_MD_PATH,
) -> PipelineState:
    """Marca um impedimento como resolvido e atualiza USER.md."""
    state = load_pipeline_state(state_path)
    now = utc_now_iso()
    affected_steps = set()
    for b in state.blockers:
        if b.id == blocker_id:
            b.resolved = True
            b.resolved_at = now
            affected_steps.add(b.step_id)

    active_blockers = [b for b in state.blockers if not b.resolved]
    still_blocked_steps = {b.step_id for b in active_blockers}
    for step in state.steps:
        if step.step_id in affected_steps and step.step_id not in still_blocked_steps and step.status == "blocked":
            step.status = "pending"

    state.overall_status = "blocked_by_user" if active_blockers else "ready"
    save_pipeline_state(state, state_path, user_md_path)
    return state


def log_discovery(
    category: str,
    title: str,
    problem: str,
    solution: str,
    discovery_id: Optional[str] = None,
    state_path: Path = DEFAULT_STATE_PATH,
    user_md_path: Path = DEFAULT_USER_MD_PATH,
    knowledge_path: Path = DEFAULT_KNOWLEDGE_PATH,
) -> PipelineState:
    """
    Registra conhecimento incremental descoberto por um agente em state/pipeline_state.json
    e adiciona a entrada estruturada em state/AGENT_KNOWLEDGE.md (Spec 05).
    """
    state = load_pipeline_state(state_path)
    now = utc_now_iso()
    disc_id = discovery_id or f"DISC-{len(state.discoveries) + 1:03d}"

    entry = AgentDiscoveryEntry(
        id=disc_id,
        timestamp=now,
        category=category,
        title=title,
        problem=problem,
        solution=solution,
    )
    state.discoveries = [d for d in state.discoveries if d.id != disc_id] + [entry]

    knowledge_path.parent.mkdir(parents=True, exist_ok=True)
    if not knowledge_path.exists():
        knowledge_path.write_text(
            "# AGENT_KNOWLEDGE.md — Base de Conhecimento Incremental dos Agentes\n\n",
            encoding="utf-8",
        )

    existing_text = knowledge_path.read_text(encoding="utf-8")
    if f"### [{entry.id}]" not in existing_text:
        block = (
            f"\n### [{entry.id}] {entry.title} (`{entry.category}` — `{entry.timestamp}`)\n"
            f"- **Problema / Armadilha:** {entry.problem}\n"
            f"- **Solução / Regra Obrigatória:** {entry.solution}\n"
        )
        with open(knowledge_path, "a", encoding="utf-8") as kf:
            kf.write(block)

    save_pipeline_state(state, state_path, user_md_path)
    _mirror_state_to_drive_if_mounted(state_path=state_path, user_md_path=user_md_path, knowledge_path=knowledge_path)
    return state


def verify_state_files(
    state_path: Path = DEFAULT_STATE_PATH,
    user_md_path: Path = DEFAULT_USER_MD_PATH,
    knowledge_path: Path = DEFAULT_KNOWLEDGE_PATH,
) -> bool:
    """Valida a integridade e sincronia dos 3 arquivos de estado da Spec 05."""
    if not state_path.exists():
        raise FileNotFoundError(f"Arquivo de estado ausente: {state_path}")
    if not user_md_path.exists():
        raise FileNotFoundError(f"Arquivo USER.md ausente: {user_md_path}")
    if not knowledge_path.exists():
        raise FileNotFoundError(f"Arquivo AGENT_KNOWLEDGE.md ausente: {knowledge_path}")

    state = load_pipeline_state(state_path)
    expected_user_md = render_user_markdown(state)
    actual_user_md = user_md_path.read_text(encoding="utf-8")
    if actual_user_md != expected_user_md:
        raise ValueError(
            "USER.md está dessincronizado com state/pipeline_state.json. "
            "Execute `python -m src.state render-user-md`."
        )

    knowledge_text = knowledge_path.read_text(encoding="utf-8")
    for d in state.discoveries:
        if f"[{d.id}]" not in knowledge_text:
            raise ValueError(f"Descoberta {d.id} consta no JSON mas está ausente em {knowledge_path}.")

    return True


def _parse_kv_pairs(pairs: Optional[List[str]]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not pairs:
        return result
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Formato inválido '{item}', esperado chave=valor.")
        k, v = item.split("=", 1)
        result[k.strip()] = v.strip()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI Stateful para Handover de Agentes e USER.md (Spec 05).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Inicializa state/pipeline_state.json e USER.md se não existirem.")
    sub.add_parser("status", help="Exibe o resumo do estado atual e instruções para o agente.")
    sub.add_parser("render-user-md", help="Regenera USER.md a partir de state/pipeline_state.json.")
    sub.add_parser("verify", help="Verifica integridade e sincronia de pipeline_state.json, USER.md e AGENT_KNOWLEDGE.md.")

    p_sync = sub.add_parser("sync-from-drive", help="Sincroniza estado persistido no Google Drive para a VM Colab.")
    p_sync.add_argument("--drive_state_dir", default=None)

    p_start = sub.add_parser("start-step", help="Marca início de uma etapa.")
    p_start.add_argument("--step", required=True)
    p_start.add_argument("--cmd", default=None)
    p_start.add_argument("--note", default=None)

    p_fail = sub.add_parser("fail-step", help="Marca falha em uma etapa e registra diagnóstico.")
    p_fail.add_argument("--step", required=True)
    p_fail.add_argument("--error", required=True)
    p_fail.add_argument("--cmd", default=None)
    p_fail.add_argument("--next-action", default=None)

    p_comp = sub.add_parser("complete-step", help="Conclui uma etapa e define handover.")
    p_comp.add_argument("--step", required=True)
    p_comp.add_argument("--next-step", default=None)
    p_comp.add_argument("--next-action", default=None)
    p_comp.add_argument("--next-cmd", default=None)
    p_comp.add_argument("--summary", default=None)
    p_comp.add_argument("--artifact", action="append", default=None, help="chave=valor")
    p_comp.add_argument("--metric", action="append", default=None, help="chave=valor")
    p_comp.add_argument("--note", default=None)

    p_hand = sub.add_parser("update-handover", help="Atualiza handover parcial mantendo a etapa in_progress.")
    p_hand.add_argument("--step", required=True)
    p_hand.add_argument("--next-action", required=True)
    p_hand.add_argument("--next-cmd", required=True)
    p_hand.add_argument("--summary", default=None)
    p_hand.add_argument("--artifact", action="append", default=None, help="chave=valor")
    p_hand.add_argument("--metric", action="append", default=None, help="chave=valor")
    p_hand.add_argument("--note", default=None)

    p_block = sub.add_parser("add-blocker", help="Adiciona um impedimento que requer ação do usuário.")
    p_block.add_argument("--id", required=True)
    p_block.add_argument("--step", required=True)
    p_block.add_argument("--desc", required=True)
    p_block.add_argument("--action", required=True)

    p_unblock = sub.add_parser("resolve-blocker", help="Resolve um impedimento do usuário.")
    p_unblock.add_argument("--id", required=True)

    p_disc = sub.add_parser("log-discovery", help="Registra descoberta incremental na base de conhecimento.")
    p_disc.add_argument("--id", default=None)
    p_disc.add_argument("--category", required=True)
    p_disc.add_argument("--title", required=True)
    p_disc.add_argument("--problem", required=True)
    p_disc.add_argument("--solution", required=True)

    args = parser.parse_args()

    if args.cmd == "init":
        state = create_initial_pipeline_state()
        save_pipeline_state(state)
        print("[Spec 05] Estado inicial e USER.md criados com sucesso.")
    elif args.cmd == "status":
        state = load_pipeline_state()
        print(
            json.dumps(
                {
                    "overall_status": state.overall_status,
                    "current_step_id": state.current_step_id,
                    "active_blockers": [b.model_dump() for b in state.blockers if not b.resolved],
                    "agent_handover": state.agent_handover.model_dump(),
                    "discoveries_count": len(state.discoveries),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.cmd == "render-user-md":
        state = load_pipeline_state()
        save_pipeline_state(state)
        print("[Spec 05] USER.md sincronizado com sucesso.")
    elif args.cmd == "verify":
        verify_state_files()
        print("[Spec 05] Verificação concluída: pipeline_state.json, USER.md e AGENT_KNOWLEDGE.md estão íntegros e sincronizados.")
    elif args.cmd == "sync-from-drive":
        d_dir = Path(args.drive_state_dir) if args.drive_state_dir else None
        ok = sync_state_from_drive(drive_state_dir=d_dir)
        print(f"[Spec 05] Sincronização a partir do Google Drive: {'restaurado' if ok else 'nenhum estado prévio no Drive'}.")
    elif args.cmd == "start-step":
        start_step(step_id=args.step, command=args.cmd, notes=args.note)
        print(f"[Spec 05] Etapa '{args.step}' marcada como 'in_progress' e USER.md atualizado.")
    elif args.cmd == "fail-step":
        fail_step(
            step_id=args.step,
            error_message=args.error,
            command=args.cmd,
            next_action=args.next_action,
        )
        print(f"[Spec 05] Etapa '{args.step}' marcada como 'failed' e USER.md atualizado.")
    elif args.cmd == "complete-step":
        complete_step(
            step_id=args.step,
            next_step_id=args.next_step,
            next_action=args.next_action,
            next_command=args.next_cmd,
            last_summary=args.summary,
            artifacts=_parse_kv_pairs(args.artifact),
            metrics=_parse_kv_pairs(args.metric),
            notes=args.note,
        )
        print(f"[Spec 05] Etapa '{args.step}' concluída e USER.md atualizado.")
    elif args.cmd == "update-handover":
        update_handover(
            step_id=args.step,
            next_action=args.next_action,
            next_command=args.next_cmd,
            last_summary=args.summary,
            artifacts=_parse_kv_pairs(args.artifact),
            metrics=_parse_kv_pairs(args.metric),
            notes=args.note,
        )
        print(f"[Spec 05] Handover parcial de '{args.step}' atualizado (status=in_progress) e USER.md sincronizado.")
    elif args.cmd == "add-blocker":
        add_blocker(
            blocker_id=args.id,
            step_id=args.step,
            description=args.desc,
            user_action_required=args.action,
        )
        print(f"[Spec 05] Bloqueio '{args.id}' registrado e destacado em USER.md.")
    elif args.cmd == "resolve-blocker":
        resolve_blocker(blocker_id=args.id)
        print(f"[Spec 05] Bloqueio '{args.id}' resolvido e USER.md atualizado.")
    elif args.cmd == "log-discovery":
        log_discovery(
            category=args.category,
            title=args.title,
            problem=args.problem,
            solution=args.solution,
            discovery_id=args.id,
        )
        print("[Spec 05] Descoberta incremental registrada em pipeline_state.json e AGENT_KNOWLEDGE.md.")


if __name__ == "__main__":
    main()
