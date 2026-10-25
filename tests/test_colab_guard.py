"""Testes de contrato para a operação remota não-bloqueante do Google Colab (Spec 06)."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.colab_cli_bridge import BRIDGE_MODES, EXIT_CODES as BRIDGE_EXIT_CODES
from src.colab_guard import (
    DONE_SENTINEL,
    JOB_RESULT_PREFIX,
    ColabGuard,
    build_job_kill_code,
    build_job_status_code,
    build_launch_code,
    build_parser,
    classify_run,
    detect_interaction_request,
    detect_session_lost,
    extract_auth_url,
    parse_prefixed_json,
    run_guarded,
    validate_sync_timeout,
    wrap_remote_code,
)
from src.schemas import (
    COLAB_GUARD_EXIT_CODES,
    MAX_SYNC_EXEC_TIMEOUT_SEC,
    ColabGuardOutcome,
)
from src.state import (
    create_initial_pipeline_state,
    load_pipeline_state,
    save_pipeline_state,
    verify_state_files,
)


def test_exit_codes_sync_and_outcome_schema():
    """Spec 06: códigos de saída de schemas.py e colab_cli_bridge.py devem coincidir."""
    assert COLAB_GUARD_EXIT_CODES == BRIDGE_EXIT_CODES

    ok = ColabGuardOutcome(status="ok", exit_code=0, command="preflight", elapsed_sec=1.2)
    assert ok.exit_code == 0

    with pytest.raises(ValidationError):
        ColabGuardOutcome(status="needs_user", exit_code=0, command="drive", elapsed_sec=0.5, user_action="Approve")

    with pytest.raises(ValidationError):
        ColabGuardOutcome(status="needs_user", exit_code=10, command="drive", elapsed_sec=0.5)


def test_sync_timeout_limits():
    """Spec 06: timeouts síncronos > 600s ou <= 0 devem ser rejeitados."""
    assert validate_sync_timeout(120.0) == 120.0
    assert validate_sync_timeout(MAX_SYNC_EXEC_TIMEOUT_SEC) == MAX_SYNC_EXEC_TIMEOUT_SEC
    with pytest.raises(ValueError):
        validate_sync_timeout(0)
    with pytest.raises(ValueError):
        validate_sync_timeout(MAX_SYNC_EXEC_TIMEOUT_SEC + 1)


def test_detect_interaction_and_url_extraction():
    drive_msg = (
        "\n[colab] REQUIRED: Google Drive Authorization needed.\nPlease visit:\n\n"
        "https://accounts.google.com/o/oauth2/v2/auth?client_id=123&scope=drive\n"
        "Press Enter after you have granted access... "
    )
    found = detect_interaction_request(drive_msg)
    assert found is not None
    reason, action, url = found
    assert reason == "drive_auth"
    assert "Google Drive" in action
    assert url == "https://accounts.google.com/o/oauth2/v2/auth?client_id=123&scope=drive"
    assert extract_auth_url(drive_msg) == url

    assert detect_session_lost("[colab] Session 'parakeet' appears to be lost (404/401). Cleaning up.")
    assert not detect_session_lost(" Tudo certo no kernel ")


def test_run_guarded_aborts_unterminated_drive_prompt_fast():
    """Spec 06: prompt sem '\\n' que espera /dev/tty ou sleep deve ser abortado em poucos segundos."""
    code = (
        "import sys, time\n"
        "sys.stdout.write('REQUIRED: Google Drive Authorization needed.\\n')\n"
        "sys.stdout.write('https://accounts.google.com/o/oauth2/v2/auth?client_id=abc\\n')\n"
        "sys.stdout.write('Press Enter after you have granted access... ')\n"
        "sys.stdout.flush()\n"
        "time.sleep(120)\n"
    )
    t0 = time.monotonic()
    run = run_guarded([sys.executable, "-c", code], hard_timeout=30.0, echo=False)
    elapsed = time.monotonic() - t0

    assert elapsed < 8.0
    assert run.killed_reason == "needs_user"
    assert run.interaction is not None
    assert run.interaction[0] == "drive_auth"
    assert run.interaction[2] == "https://accounts.google.com/o/oauth2/v2/auth?client_id=abc"

    status, reason = classify_run(run, expect_sentinel=True)
    assert status == "needs_user"
    assert reason == "drive_auth"


def test_run_guarded_hard_and_idle_timeouts():
    code = "import sys, time\nsys.stdout.write('iniciando...\\n'); sys.stdout.flush()\ntime.sleep(60)\n"

    run_hard = run_guarded([sys.executable, "-c", code], hard_timeout=1.2, echo=False)
    assert run_hard.killed_reason == "timeout"
    assert classify_run(run_hard, expect_sentinel=True) == ("timeout", "guard_timeout")

    run_idle = run_guarded([sys.executable, "-c", code], hard_timeout=30.0, idle_timeout=1.0, echo=False)
    assert run_idle.killed_reason == "idle_timeout"
    assert classify_run(run_idle, expect_sentinel=True) == ("idle_timeout", "guard_idle_timeout")


def test_sentinel_detection_distinguishes_ok_from_remote_exception():
    wrapped_ok = wrap_remote_code("print('hello world')")
    run_ok = run_guarded([sys.executable, "-c", wrapped_ok], hard_timeout=10.0, echo=False)
    assert classify_run(run_ok, expect_sentinel=True) == ("ok", None)

    # Simula o comportamento do `colab exec`: sai com exit code 0 mesmo quando o código remoto lança exceção
    # antes de chegar ao sentinela.
    fake_colab_exec = (
        "import sys\n"
        "src = sys.stdin.read()\n"
        "try:\n"
        "    exec(compile(src, '<cell>', 'exec'), {})\n"
        "except Exception as e:\n"
        "    sys.stderr.write(f'RuntimeError: {e}\\n')\n"
        "sys.exit(0)\n"
    )
    run_err = run_guarded(
        [sys.executable, "-c", fake_colab_exec],
        stdin_text=wrap_remote_code("raise RuntimeError('boom')"),
        hard_timeout=10.0,
        echo=False,
    )
    assert DONE_SENTINEL not in run_err.output
    assert classify_run(run_err, expect_sentinel=True) == ("remote_error", "remote_exception_or_no_sentinel")


def test_needs_user_registers_blocker_and_forbidden_cli_subcommands(tmp_path: Path):
    """Spec 06 + Spec 05: needs_user registra bloqueio em pipeline_state.json e sincroniza USER.md."""
    state_file = tmp_path / "state" / "pipeline_state.json"
    knowledge_file = tmp_path / "state" / "AGENT_KNOWLEDGE.md"
    user_md = tmp_path / "USER.md"
    knowledge_file.parent.mkdir(parents=True, exist_ok=True)
    knowledge_file.write_text("# AGENT_KNOWLEDGE.md\n\n", encoding="utf-8")
    save_pipeline_state(create_initial_pipeline_state(), state_path=state_file, user_md_path=user_md)

    fake_colab = tmp_path / "colab"
    fake_colab.write_text(
        f"#!{sys.executable}\n"
        "import sys, time\n"
        "sys.stdout.write('To authorize colab-cli, visit this URL in any browser:\\n')\n"
        "sys.stdout.write('  https://accounts.google.com/o/oauth2/v2/auth?client_id=xyz\\n')\n"
        "sys.stdout.write('Enter the authorization code: ')\n"
        "sys.stdout.flush()\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    fake_colab.chmod(fake_colab.stat().st_mode | stat.S_IEXEC)

    guard = ColabGuard(
        session="parakeet",
        colab_bin=str(fake_colab),
        step_id="step_05_colab_stage1_warmup",
        register_blockers=True,
        open_browser=False,
        echo=False,
        state_path=state_file,
        user_md_path=user_md,
    )
    outcome = guard.exec_code("print(1)", timeout=10.0, interrupt_on_timeout=False)
    assert outcome.status == "needs_user"
    assert outcome.exit_code == COLAB_GUARD_EXIT_CODES["needs_user"] == 10
    assert outcome.blocker_id == "blocker_colab_colab_cli_oauth"
    assert outcome.auth_url == "https://accounts.google.com/o/oauth2/v2/auth?client_id=xyz"

    st = load_pipeline_state(state_file)
    assert st.overall_status == "blocked_by_user"
    assert any(b.id == "blocker_colab_colab_cli_oauth" and not b.resolved for b in st.blockers)
    assert verify_state_files(state_path=state_file, user_md_path=user_md, knowledge_path=knowledge_file)

    for forbidden in ("exec", "drivemount", "auth", "repl", "console", "ssh"):
        forb = guard.cli([forbidden])
        assert forb.status == "cli_error"
        assert forb.reason == "interactive_subcommand_forbidden"


def test_remote_background_job_lifecycle(tmp_path: Path):
    """Spec 06: launch, job-status e job-kill compilam e gerenciam processos em background."""
    job_dir = str(tmp_path / "jobs")
    cwd = str(tmp_path)

    for snippet in (
        build_launch_code("tjob", "echo hi", cwd=cwd, job_dir=job_dir),
        build_job_status_code("tjob", job_dir=job_dir, tail_lines=10),
        build_job_kill_code("tjob", job_dir=job_dir),
    ):
        compile(snippet, "<remote_job>", "exec")

    def run_snippet(code: str) -> dict:
        out = subprocess.check_output([sys.executable, "-c", code], text=True, timeout=15)
        payload = parse_prefixed_json(out, JOB_RESULT_PREFIX)
        assert payload is not None, out
        return payload

    assert run_snippet(build_job_status_code("short", job_dir=job_dir))["job_state"] == "not_found"

    launched = run_snippet(
        build_launch_code("short", f"{sys.executable} -c \"print('linha_final_ok')\"", cwd=cwd, job_dir=job_dir)
    )
    assert launched["job_state"] == "launched"

    for _ in range(30):
        st = run_snippet(build_job_status_code("short", job_dir=job_dir, tail_lines=10))
        if st["job_state"] != "running":
            break
        time.sleep(0.1)
    assert st["job_state"] == "succeeded"
    assert st["exit_code"] == 0
    assert "linha_final_ok" in st["tail"]

    # Job longo: deve aparecer running, recusar duplicata e terminar com job-kill.
    run_snippet(build_launch_code("long", f"{sys.executable} -c 'import time; time.sleep(60)'", cwd=cwd, job_dir=job_dir))
    dup = run_snippet(build_launch_code("long", "echo noop", cwd=cwd, job_dir=job_dir))
    assert dup["job_state"] == "already_running"

    killed = run_snippet(build_job_kill_code("long", job_dir=job_dir))
    assert killed["job_state"] in ("killed", "not_running")
    st_after = run_snippet(build_job_status_code("long", job_dir=job_dir))
    assert st_after["job_state"] == "failed"
    assert st_after["exit_code"] == 143


def test_restart_subcommand_and_bridge_mode():
    """Spec 06 §5.6: restart libera kernel preso em syscall que ignora SIGINT (ex.: Drive FUSE)."""
    assert build_parser().parse_args(["restart"]).cmd == "restart"
    assert {"interrupt", "restart"} <= set(BRIDGE_MODES)
