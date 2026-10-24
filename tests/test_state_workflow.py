"""Testes de contrato para a Orquestração Stateful de Agentes, USER.md e Conhecimento Incremental (Spec 05)."""

import pytest

from src.state import (
    DEFAULT_KNOWLEDGE_PATH,
    DEFAULT_STATE_PATH,
    DEFAULT_USER_MD_PATH,
    add_blocker,
    complete_step,
    create_initial_pipeline_state,
    fail_step,
    log_discovery,
    resolve_blocker,
    save_pipeline_state,
    start_step,
    sync_state_from_drive,
    update_handover,
    verify_state_files,
)


def test_repository_state_files_exist_and_are_in_sync():
    """Gatekeeper permanente: garante que pipeline_state.json, USER.md e AGENT_KNOWLEDGE.md estão sincronizados."""
    assert verify_state_files(
        state_path=DEFAULT_STATE_PATH,
        user_md_path=DEFAULT_USER_MD_PATH,
        knowledge_path=DEFAULT_KNOWLEDGE_PATH,
    ) is True


def test_stateful_lifecycle_transitions_and_user_md_sync(tmp_path):
    state_file = tmp_path / "state" / "pipeline_state.json"
    knowledge_file = tmp_path / "state" / "AGENT_KNOWLEDGE.md"
    user_md_file = tmp_path / "USER.md"

    knowledge_file.parent.mkdir(parents=True, exist_ok=True)
    knowledge_file.write_text("# AGENT_KNOWLEDGE.md\n\n", encoding="utf-8")

    initial = create_initial_pipeline_state()
    save_pipeline_state(initial, state_path=state_file, user_md_path=user_md_file)
    assert verify_state_files(state_file, user_md_file, knowledge_file) is True

    # 1. Inicia step_01_data_prep_clean
    st = start_step(
        step_id="step_01_data_prep_clean",
        command="python -m src.data_prep",
        notes="Baixando FLEURS PT-BR",
        state_path=state_file,
        user_md_path=user_md_file,
    )
    assert st.overall_status == "in_progress"
    assert st.current_step_id == "step_01_data_prep_clean"

    # 2. Registra bloqueio do usuário e verifica que aparece em USER.md
    st = add_blocker(
        blocker_id="BLK-TEST-01",
        step_id="step_01_data_prep_clean",
        description="Falta token do Hugging Face para Common Voice",
        user_action_required="Defina HF_TOKEN no ambiente",
        state_path=state_file,
        user_md_path=user_md_file,
    )
    assert st.overall_status == "blocked_by_user"
    assert "BLK-TEST-01" in user_md_file.read_text(encoding="utf-8")
    assert "Defina HF_TOKEN no ambiente" in user_md_file.read_text(encoding="utf-8")

    # 3. Resolve bloqueio
    st = resolve_blocker(
        blocker_id="BLK-TEST-01",
        state_path=state_file,
        user_md_path=user_md_file,
    )
    assert st.overall_status == "ready"
    assert "BLK-TEST-01" not in user_md_file.read_text(encoding="utf-8")

    # 4. Conclui etapa e avança para step_02_tokenizer_ptbr
    st = complete_step(
        step_id="step_01_data_prep_clean",
        next_step_id="step_02_tokenizer_ptbr",
        next_action="Treinar tokenizador BPE 1024",
        next_command="python -m src.tokenizer",
        artifacts={"clean_manifest": "data/clean_manifest.json"},
        metrics={"samples": 1200},
        state_path=state_file,
        user_md_path=user_md_file,
    )
    assert st.current_step_id == "step_02_tokenizer_ptbr"

    # 5. Registra descoberta incremental
    log_discovery(
        category="teste",
        title="Descoberta de Teste",
        problem="Problema X",
        solution="Solução Y",
        discovery_id="DISC-999",
        state_path=state_file,
        user_md_path=user_md_file,
        knowledge_path=knowledge_file,
    )
    assert "[DISC-999]" in knowledge_file.read_text(encoding="utf-8")
    assert verify_state_files(state_file, user_md_file, knowledge_file) is True

    # 6. Corromper manualmente o USER.md deve ser detectado pelo verify_state_files
    user_md_file.write_text("# USER.md adulterado manualmente\n", encoding="utf-8")
    with pytest.raises(ValueError, match="USER.md está dessincronizado"):
        verify_state_files(state_file, user_md_file, knowledge_file)


def test_update_handover_fail_step_and_drive_mirror(tmp_path, monkeypatch):
    drive_dir = tmp_path / "drive_state"
    monkeypatch.setenv("PARAKEET_DRIVE_STATE_DIR", str(drive_dir))

    state_file = tmp_path / "local" / "pipeline_state.json"
    knowledge_file = tmp_path / "local" / "AGENT_KNOWLEDGE.md"
    user_md_file = tmp_path / "local" / "USER.md"
    knowledge_file.parent.mkdir(parents=True, exist_ok=True)
    knowledge_file.write_text("# AGENT_KNOWLEDGE.md\n\n", encoding="utf-8")

    initial = create_initial_pipeline_state()
    save_pipeline_state(initial, state_path=state_file, user_md_path=user_md_file)

    # Verifica que o espelhamento para o Drive ocorreu automaticamente
    assert (drive_dir / "pipeline_state.json").exists()
    assert (drive_dir / "USER.md").exists()

    # Handover parcial mantendo in_progress
    st = update_handover(
        step_id="step_01_data_prep_clean",
        next_action="Baixar split val",
        next_command="python -m src.data_prep --split validation",
        artifacts={"manifest_train": "data/clean_train.json"},
        state_path=state_file,
        user_md_path=user_md_file,
    )
    assert st.overall_status == "in_progress"
    assert st.steps[1].status == "in_progress"

    # Marca falha na etapa
    st_failed = fail_step(
        step_id="step_01_data_prep_clean",
        error_message="Erro de rede no download",
        state_path=state_file,
        user_md_path=user_md_file,
    )
    assert st_failed.overall_status == "failed"
    assert st_failed.steps[1].status == "failed"
    assert "Erro de rede no download" in user_md_file.read_text(encoding="utf-8")

    # Testa restauração a partir do Drive em um novo diretório efêmero
    new_local_state = tmp_path / "new_vm" / "pipeline_state.json"
    new_local_user = tmp_path / "new_vm" / "USER.md"
    new_local_know = tmp_path / "new_vm" / "AGENT_KNOWLEDGE.md"
    assert sync_state_from_drive(
        drive_state_dir=drive_dir,
        state_path=new_local_state,
        user_md_path=new_local_user,
        knowledge_path=new_local_know,
    ) is True
    assert new_local_state.exists() and new_local_user.exists()
