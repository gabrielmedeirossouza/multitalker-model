"""Executor guardado e não-bloqueante para operar o Google Colab via CLI (Spec 06).

Uso (sempre a partir do Mac, com o `.venv`):

    .venv/bin/python -m src.colab_guard preflight
    .venv/bin/python -m src.colab_guard drive --open-browser
    .venv/bin/python -m src.colab_guard exec -f script.py --timeout 120
    .venv/bin/python -m src.colab_guard exec --code "import os; print(os.listdir('/content'))"
    .venv/bin/python -m src.colab_guard launch --job warmup --cmd "python -m src.trainer ..."
    .venv/bin/python -m src.colab_guard job-status --job warmup --tail 40
    .venv/bin/python -m src.colab_guard job-kill --job warmup
    .venv/bin/python -m src.colab_guard interrupt
    .venv/bin/python -m src.colab_guard cli -- sessions

Garantias:
  * Todo subcomando termina em tempo limitado (limite total + inatividade opcional).
  * Prompts de autorização humana (Drive, OAuth do CLI) são detectados no fluxo de saída;
    o processo é morto imediatamente e o resultado é `needs_user` (exit code 10), com o
    bloqueio registrado em state/pipeline_state.json + USER.md (Spec 05).
  * A última linha impressa é sempre `COLAB_GUARD_RESULT {json}` (ColabGuardOutcome).
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.schemas import (
    COLAB_GUARD_EXIT_CODES,
    DEFAULT_CLI_PASSTHROUGH_TIMEOUT_SEC,
    DEFAULT_SYNC_EXEC_TIMEOUT_SEC,
    MAX_SYNC_EXEC_TIMEOUT_SEC,
    ColabGuardOutcome,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BRIDGE_SCRIPT = REPO_ROOT / "src" / "colab_cli_bridge.py"

DEFAULT_SESSION = os.environ.get("COLAB_SESSION", "parakeet")
DEFAULT_COLAB_BIN = os.environ.get("COLAB_BIN", str(Path.home() / ".local" / "bin" / "colab"))
DEFAULT_JOB_DIR = "/content/jobs"
DEFAULT_REMOTE_CWD = "/content/multitalker-model"

RESULT_PREFIX = "COLAB_GUARD_RESULT "
BRIDGE_RESULT_PREFIX = "COLAB_BRIDGE_RESULT "
JOB_RESULT_PREFIX = "COLAB_GUARD_JOB "
DONE_SENTINEL = "__COLAB_GUARD_DONE__"

NEEDS_USER_GRACE_SEC = 2.0
KILL_GRACE_SEC = 3.0
TAIL_CHARS = 4000

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07")
URL_RE = re.compile(r"https://[^\s'\"<>]+")

# (regex, reason, user_action) — detectados no fluxo contínuo (prompts podem não terminar em "\n").
INTERACTION_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    (
        re.compile(r"REQUIRED: Google Drive Authorization|Press Enter after you have granted access", re.I),
        "drive_auth",
        "Abra a URL de autorização do Google Drive, aprove o acesso para esta VM do Colab e responda 'ok' ao agente.",
    ),
    (
        re.compile(r"To authorize colab-cli, visit this URL|Enter the authorization code", re.I),
        "colab_cli_oauth",
        "O google-colab-cli perdeu o login OAuth. Rode `~/.local/bin/colab sessions` no seu terminal, conclua o login e responda 'ok'.",
    ),
    (
        re.compile(r"No valid default credentials found", re.I),
        "colab_cli_adc",
        "Credenciais ADC ausentes para o colab-cli. Rode o `gcloud auth application-default login` indicado e responda 'ok'.",
    ),
    (
        re.compile(r"Go to this URL in a browser|Enter (your )?verification code|Enter your authorization code", re.I),
        "google_auth_prompt",
        "Um fluxo de autenticação Google pediu interação humana. Abra a URL indicada, autorize e responda 'ok'.",
    ),
]

SESSION_LOST_RE = re.compile(
    r"appears to be lost \(404/401\)|Session '[^']*' not found|No active sessions|session not found",
    re.I,
)
REMOTE_TIMEOUT_RE = re.compile(r"\bTimeoutError\b")
SENTINEL_LINE_RE = re.compile(rf"^{DONE_SENTINEL}\s*$", re.M)


# ---------------------------------------------------------------------------
# Funções puras (testáveis)
# ---------------------------------------------------------------------------


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def detect_interaction_request(text: str) -> Optional[Tuple[str, str, Optional[str]]]:
    """Retorna (reason, user_action, url) se o texto contém um pedido de interação humana."""
    clean = strip_ansi(text)
    for pattern, reason, action in INTERACTION_PATTERNS:
        if pattern.search(clean):
            return reason, action, extract_auth_url(clean)
    return None


def extract_auth_url(text: str) -> Optional[str]:
    """Extrai a URL de autorização mais provável (prioriza accounts.google.com)."""
    urls = [u.rstrip(".,);") for u in URL_RE.findall(strip_ansi(text))]
    if not urls:
        return None
    for u in urls:
        if "accounts.google.com" in u or "oauth" in u.lower():
            return u
    return urls[-1]


def detect_session_lost(text: str) -> bool:
    return bool(SESSION_LOST_RE.search(strip_ansi(text)))


def validate_sync_timeout(timeout: float) -> float:
    if timeout <= 0:
        raise ValueError("Spec 06: --timeout deve ser > 0.")
    if timeout > MAX_SYNC_EXEC_TIMEOUT_SEC:
        raise ValueError(
            f"Spec 06: timeout síncrono {timeout:.0f}s excede MAX_SYNC_EXEC_TIMEOUT_SEC="
            f"{MAX_SYNC_EXEC_TIMEOUT_SEC:.0f}s. Use `launch` + `job-status` para tarefas longas."
        )
    return timeout


def wrap_remote_code(code: str) -> str:
    """Anexa o sentinela de conclusão; sem ele, o processo encerrado é tratado como remote_error."""
    return f"{code.rstrip()}\n\nprint({DONE_SENTINEL!r}, flush=True)\n"


def parse_prefixed_json(text: str, prefix: str) -> Optional[Dict[str, object]]:
    """Retorna o último payload JSON impresso após `prefix`."""
    found = None
    for line in strip_ansi(text).splitlines():
        idx = line.find(prefix)
        if idx >= 0:
            try:
                found = json.loads(line[idx + len(prefix):])
            except ValueError:
                continue
    return found


def resolve_colab_cli_python(colab_bin: str) -> str:
    """Descobre o interpretador do google-colab-cli (env COLAB_CLI_PYTHON ou shebang do binário)."""
    env = os.environ.get("COLAB_CLI_PYTHON")
    if env:
        return env
    try:
        with open(colab_bin, "r", encoding="utf-8", errors="ignore") as f:
            first = f.readline().strip()
        if first.startswith("#!"):
            parts = first[2:].strip().split()
            if parts and parts[0].endswith("env") and len(parts) > 1:
                return shutil.which(parts[-1]) or parts[-1]
            return parts[0]
    except OSError:
        pass
    return str(Path.home() / ".local" / "share" / "uv" / "tools" / "google-colab-cli" / "bin" / "python")


# ---------------------------------------------------------------------------
# Executor guardado
# ---------------------------------------------------------------------------


@dataclass
class GuardedRun:
    argv: List[str]
    returncode: Optional[int]
    output: str
    elapsed_sec: float
    killed_reason: Optional[str] = None  # needs_user | timeout | idle_timeout
    interaction: Optional[Tuple[str, str, Optional[str]]] = None
    extra: Dict[str, object] = field(default_factory=dict)


def _kill_process_group(proc: subprocess.Popen) -> None:
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        return
    for sig, wait in ((signal.SIGTERM, KILL_GRACE_SEC), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(pgid, sig)
        except OSError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def run_guarded(
    argv: Sequence[str],
    *,
    stdin_text: Optional[str] = None,
    hard_timeout: float,
    idle_timeout: Optional[float] = None,
    echo: bool = True,
    env: Optional[Dict[str, str]] = None,
) -> GuardedRun:
    """Executa um comando com limites de tempo e detecção de prompts humanos.

    - Nova sessão (sem TTY controlador): `open('/dev/tty')` falha em vez de bloquear.
    - stdin fechado após enviar `stdin_text` (EOF imediato para qualquer `input()`).
    - Saída lida em blocos (prompts sem "\\n" também são detectados).
    """
    full_env = dict(os.environ)
    full_env.update({"PYTHONUNBUFFERED": "1", "NO_COLOR": "1", "TERM": "dumb"})
    if env:
        full_env.update(env)

    start = time.monotonic()
    try:
        proc = subprocess.Popen(
            list(argv),
            stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=full_env,
            bufsize=0,
        )
    except OSError as exc:
        return GuardedRun(list(argv), 127, f"{type(exc).__name__}: {exc}\n", 0.0, killed_reason=None)

    if stdin_text is not None and proc.stdin is not None:
        try:
            proc.stdin.write(stdin_text.encode("utf-8"))
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass

    chunks: "queue.Queue[Optional[bytes]]" = queue.Queue()

    def reader() -> None:
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        while True:
            try:
                data = os.read(fd, 4096)
            except OSError:
                data = b""
            if not data:
                chunks.put(None)
                return
            chunks.put(data)

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    buf: List[str] = []
    scan_window = ""
    last_output = time.monotonic()
    killed_reason: Optional[str] = None
    interaction: Optional[Tuple[str, str, Optional[str]]] = None
    interaction_deadline: Optional[float] = None
    eof = False

    while True:
        now = time.monotonic()
        if not eof:
            try:
                item = chunks.get(timeout=0.25)
            except queue.Empty:
                item = b""
            if item is None:
                eof = True
            elif item:
                text = item.decode("utf-8", errors="replace")
                buf.append(text)
                last_output = time.monotonic()
                if echo:
                    sys.stdout.write(text)
                    sys.stdout.flush()
                scan_window = (scan_window + text)[-16000:]
                if interaction is None:
                    found = detect_interaction_request(scan_window)
                    if found:
                        interaction = found
                        interaction_deadline = time.monotonic() + NEEDS_USER_GRACE_SEC
                elif interaction[2] is None:
                    url = extract_auth_url(scan_window)
                    if url:
                        interaction = (interaction[0], interaction[1], url)
        else:
            if proc.poll() is not None:
                break
            time.sleep(0.05)

        now = time.monotonic()
        if interaction is not None and (
            (interaction_deadline is not None and now >= interaction_deadline) or eof
        ):
            if proc.poll() is None:
                _kill_process_group(proc)
            killed_reason = "needs_user"
            break
        if now - start > hard_timeout:
            _kill_process_group(proc)
            killed_reason = "timeout"
            break
        if idle_timeout and now - last_output > idle_timeout:
            _kill_process_group(proc)
            killed_reason = "idle_timeout"
            break
        if eof and proc.poll() is not None:
            break

    # Drena o restante já lido (sem bloquear).
    t.join(timeout=1.0)
    while True:
        try:
            item = chunks.get_nowait()
        except queue.Empty:
            break
        if item:
            text = item.decode("utf-8", errors="replace")
            buf.append(text)
            if echo:
                sys.stdout.write(text)
                sys.stdout.flush()
    try:
        rc = proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        rc = proc.poll()

    output = "".join(buf)
    if interaction is not None and interaction[2] is None:
        url = extract_auth_url(output)
        interaction = (interaction[0], interaction[1], url)
    return GuardedRun(
        argv=list(argv),
        returncode=rc,
        output=output,
        elapsed_sec=round(time.monotonic() - start, 3),
        killed_reason=killed_reason,
        interaction=interaction,
    )


def classify_run(run: GuardedRun, *, expect_sentinel: bool) -> Tuple[str, Optional[str]]:
    """Mapeia um GuardedRun para (status, reason) do ColabGuardOutcome."""
    if run.killed_reason == "needs_user" or run.interaction is not None:
        return "needs_user", run.interaction[0] if run.interaction else "interaction_required"
    if run.killed_reason in ("timeout", "idle_timeout"):
        return run.killed_reason, f"guard_{run.killed_reason}"
    clean = strip_ansi(run.output)
    if detect_session_lost(clean):
        return "session_lost", "session_not_found"
    if run.returncode == 127:
        return "cli_error", "command_not_found"
    if expect_sentinel:
        if SENTINEL_LINE_RE.search(clean):
            return "ok", None
        if REMOTE_TIMEOUT_RE.search(clean):
            return "timeout", "colab_exec_timeout"
        return "remote_error", "remote_exception_or_no_sentinel"
    if run.returncode == 0:
        return "ok", None
    return "remote_error", f"exit_code_{run.returncode}"


# ---------------------------------------------------------------------------
# Construção de outcome + integração com o estado (Spec 05)
# ---------------------------------------------------------------------------


def _tail(text: str, n: int = TAIL_CHARS) -> str:
    clean = strip_ansi(text).replace(DONE_SENTINEL, "").rstrip()
    return clean[-n:]


def build_outcome(
    status: str,
    command: str,
    elapsed_sec: float,
    *,
    reason: Optional[str] = None,
    user_action: Optional[str] = None,
    auth_url: Optional[str] = None,
    output: str = "",
    data: Optional[Dict[str, object]] = None,
) -> ColabGuardOutcome:
    if status == "needs_user" and not (user_action or auth_url):
        user_action = "O Colab pediu interação humana. Verifique a saída do comando e responda 'ok' ao agente."
    return ColabGuardOutcome(
        status=status,  # type: ignore[arg-type]
        exit_code=COLAB_GUARD_EXIT_CODES[status],
        command=command,
        elapsed_sec=max(0.0, float(elapsed_sec)),
        reason=reason,
        user_action=user_action,
        auth_url=auth_url,
        output_tail=_tail(output),
        data=data or {},
    )


def register_needs_user_blocker(
    outcome: ColabGuardOutcome,
    step_id: Optional[str] = None,
    state_path: Optional[Path] = None,
    user_md_path: Optional[Path] = None,
) -> ColabGuardOutcome:
    """Registra o bloqueio humano em pipeline_state.json + USER.md e devolve o outcome atualizado."""
    from src import state as st

    sp = state_path or st.DEFAULT_STATE_PATH
    up = user_md_path or st.DEFAULT_USER_MD_PATH
    current = st.load_pipeline_state(sp)
    step = step_id or current.current_step_id
    blocker_id = f"blocker_colab_{outcome.reason or 'interaction'}"
    action = outcome.user_action or "Ação humana necessária no Colab."
    if outcome.auth_url:
        action = f"{action} URL: {outcome.auth_url}"
    st.add_blocker(
        blocker_id=blocker_id,
        step_id=step,
        description=f"Colab pediu interação humana ({outcome.reason}) durante `{outcome.command}`.",
        user_action_required=action,
        state_path=sp,
        user_md_path=up,
    )
    return outcome.model_copy(update={"blocker_id": blocker_id})


def resolve_blockers_with_prefix(
    prefix: str,
    state_path: Optional[Path] = None,
    user_md_path: Optional[Path] = None,
) -> List[str]:
    from src import state as st

    sp = state_path or st.DEFAULT_STATE_PATH
    up = user_md_path or st.DEFAULT_USER_MD_PATH
    current = st.load_pipeline_state(sp)
    ids = [
        b.id
        for b in current.blockers
        if not b.resolved and (b.id.startswith(prefix) or ("drive" in prefix and "drive" in b.id))
    ]
    for bid in ids:
        st.resolve_blocker(bid, state_path=sp, user_md_path=up)
    return ids


def print_user_action_banner(outcome: ColabGuardOutcome) -> None:
    lines = [
        "",
        "=" * 72,
        "🛑 AÇÃO DO USUÁRIO NECESSÁRIA (Colab) — o agente deve PARAR e comunicar o usuário",
        f"Motivo: {outcome.reason}",
        f"Ação:   {outcome.user_action}",
    ]
    if outcome.auth_url:
        lines.append(f"URL:    {outcome.auth_url}")
    if outcome.blocker_id:
        lines.append(f"Bloqueio registrado em USER.md: {outcome.blocker_id}")
    lines.append("=" * 72)
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Código remoto (executado no kernel do Colab)
# ---------------------------------------------------------------------------

_REMOTE_JOB_HELPERS = r'''
import json as _j, os as _o, signal as _sig, subprocess as _sp, time as _t, shlex as _sh
def _cg_alive(pid):
    try:
        _o.kill(pid, 0)
    except OSError:
        return False
    try:
        with open(f"/proc/{pid}/stat") as _f:
            return _f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True
def _cg_tail(path, n):
    if not _o.path.exists(path):
        return ""
    with open(path, "rb") as _f:
        _f.seek(0, 2); _size = _f.tell(); _f.seek(max(0, _size - 65536))
        _data = _f.read().decode("utf-8", "replace")
    return "\n".join(_data.splitlines()[-n:])
def _cg_paths(d, job):
    return {k: _o.path.join(d, job + ext) for k, ext in (("meta", ".json"), ("log", ".log"), ("rc", ".exitcode"))}
'''


def build_launch_code(job: str, command: str, cwd: str = DEFAULT_REMOTE_CWD, job_dir: str = DEFAULT_JOB_DIR) -> str:
    return _REMOTE_JOB_HELPERS + f'''
_job, _cmd, _cwd, _dir = {job!r}, {command!r}, {cwd!r}, {job_dir!r}
_o.makedirs(_dir, exist_ok=True)
_p = _cg_paths(_dir, _job)
_res = None
if _o.path.exists(_p["meta"]) and not _o.path.exists(_p["rc"]):
    _m = _j.load(open(_p["meta"]))
    if _cg_alive(int(_m.get("pid", -1))):
        _res = {{"job_state": "already_running", "pid": _m.get("pid"), "log": _p["log"]}}
if _res is None:
    for _k in ("rc",):
        if _o.path.exists(_p[_k]):
            _o.remove(_p[_k])
    with open(_p["log"], "a") as _f:
        _f.write(f"\\n===== [colab_guard] launch {{_t.strftime('%Y-%m-%d %H:%M:%S')}} :: {{_cmd}}\\n")
    _rc_q = _sh.quote(_p["rc"])
    _inner = (
        f"cd {{_sh.quote(_cwd)}} && export PYTHONUNBUFFERED=1 && ( {{_cmd}} ) >> {{_sh.quote(_p['log'])}} 2>&1 < /dev/null; "
        f"echo $? > {{_rc_q}}.tmp && mv {{_rc_q}}.tmp {{_rc_q}}"
    )
    _proc = _sp.Popen(
        ["bash", "-c", _inner],
        stdin=_sp.DEVNULL,
        stdout=_sp.DEVNULL,
        stderr=_sp.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    _pid = int(_proc.pid)
    _j.dump({{"job": _job, "pid": _pid, "cmd": _cmd, "cwd": _cwd, "log": _p["log"], "started_at": _t.time()}}, open(_p["meta"], "w"))
    _res = {{"job_state": "launched", "pid": _pid, "log": _p["log"]}}
print({JOB_RESULT_PREFIX!r} + _j.dumps(_res), flush=True)
'''


def build_job_status_code(job: str, job_dir: str = DEFAULT_JOB_DIR, tail_lines: int = 30) -> str:
    return _REMOTE_JOB_HELPERS + f'''
_job, _dir, _n = {job!r}, {job_dir!r}, {int(tail_lines)}
_p = _cg_paths(_dir, _job)
if not _o.path.exists(_p["meta"]):
    _res = {{"job_state": "not_found"}}
else:
    _m = _j.load(open(_p["meta"]))
    _pid = int(_m.get("pid", -1))
    _idle = round(_t.time() - _o.path.getmtime(_p["log"]), 1) if _o.path.exists(_p["log"]) else None
    _res = {{"pid": _pid, "cmd": _m.get("cmd"), "log": _p["log"], "log_idle_sec": _idle,
            "runtime_sec": round(_t.time() - float(_m.get("started_at", _t.time())), 1),
            "tail": _cg_tail(_p["log"], _n)}}
    if _o.path.exists(_p["rc"]):
        _rc = int(open(_p["rc"]).read().strip() or -1)
        _res.update({{"job_state": "succeeded" if _rc == 0 else "failed", "exit_code": _rc}})
    elif _cg_alive(_pid):
        _res["job_state"] = "running"
    else:
        _res["job_state"] = "died"
print({JOB_RESULT_PREFIX!r} + _j.dumps(_res), flush=True)
'''


def build_job_kill_code(job: str, job_dir: str = DEFAULT_JOB_DIR) -> str:
    return _REMOTE_JOB_HELPERS + f'''
_job, _dir = {job!r}, {job_dir!r}
_p = _cg_paths(_dir, _job)
if not _o.path.exists(_p["meta"]):
    _res = {{"job_state": "not_found"}}
else:
    _pid = int(_j.load(open(_p["meta"])).get("pid", -1))
    _was = _cg_alive(_pid)
    for _s in (_sig.SIGTERM, _sig.SIGKILL):
        try:
            _o.killpg(_pid, _s)
        except OSError:
            pass
        for _ in range(10):
            if not _cg_alive(_pid):
                break
            _t.sleep(0.5)
    if not _o.path.exists(_p["rc"]):
        open(_p["rc"], "w").write("143")
    _res = {{"job_state": "killed" if _was else "not_running", "pid": _pid}}
print({JOB_RESULT_PREFIX!r} + _j.dumps(_res), flush=True)
'''


# ---------------------------------------------------------------------------
# Guarda de alto nível
# ---------------------------------------------------------------------------


class ColabGuard:
    def __init__(
        self,
        session: str = DEFAULT_SESSION,
        colab_bin: str = DEFAULT_COLAB_BIN,
        step_id: Optional[str] = None,
        register_blockers: bool = True,
        open_browser: bool = False,
        echo: bool = True,
        state_path: Optional[Path] = None,
        user_md_path: Optional[Path] = None,
    ):
        self.session = session
        self.colab_bin = colab_bin
        self.step_id = step_id
        self.register_blockers = register_blockers
        self.open_browser = open_browser
        self.echo = echo
        self.state_path = state_path
        self.user_md_path = user_md_path

    # -- helpers ----------------------------------------------------------
    def _finalize(self, outcome: ColabGuardOutcome) -> ColabGuardOutcome:
        if outcome.status == "needs_user":
            if self.register_blockers:
                try:
                    outcome = register_needs_user_blocker(
                        outcome, self.step_id, self.state_path, self.user_md_path
                    )
                except Exception as exc:  # estado nunca deve impedir o aviso ao usuário
                    outcome = outcome.model_copy(update={"data": {**outcome.data, "blocker_error": str(exc)}})
            if self.open_browser and outcome.auth_url and sys.platform == "darwin":
                try:
                    subprocess.run(["open", outcome.auth_url], timeout=10, check=False)
                except Exception:
                    pass
            if self.echo:
                print_user_action_banner(outcome)
        return outcome

    def _from_run(self, run: GuardedRun, command: str, *, expect_sentinel: bool, data=None) -> ColabGuardOutcome:
        status, reason = classify_run(run, expect_sentinel=expect_sentinel)
        user_action = run.interaction[1] if (status == "needs_user" and run.interaction) else None
        auth_url = run.interaction[2] if (status == "needs_user" and run.interaction) else None
        if status == "session_lost":
            user_action = None
        return build_outcome(
            status,
            command,
            run.elapsed_sec,
            reason=reason,
            user_action=user_action,
            auth_url=auth_url,
            output=run.output,
            data=data,
        )

    def _bridge(self, mode: str, timeout: float, extra: Sequence[str] = ()) -> ColabGuardOutcome:
        py = resolve_colab_cli_python(self.colab_bin)
        argv = [py, str(BRIDGE_SCRIPT), mode, "--session", self.session, *extra]
        command = f"bridge {mode}"
        if not Path(py).exists():
            return build_outcome("cli_error", command, 0.0, reason="colab_cli_python_not_found", output=py)
        run = run_guarded(argv, hard_timeout=timeout, echo=False)
        payload = parse_prefixed_json(run.output, BRIDGE_RESULT_PREFIX)
        if run.killed_reason or payload is None:
            out = self._from_run(run, command, expect_sentinel=False)
            if out.status == "ok":  # saiu 0 sem payload → bug na ponte
                out = build_outcome("cli_error", command, run.elapsed_sec, reason="bridge_no_result", output=run.output)
            return out
        status = str(payload.get("status", "cli_error"))
        if status not in COLAB_GUARD_EXIT_CODES:
            status = "cli_error"
        data = {k: v for k, v in payload.items() if k not in ("status", "reason", "user_action", "auth_url")}
        return build_outcome(
            status,
            command,
            run.elapsed_sec,
            reason=payload.get("reason"),  # type: ignore[arg-type]
            user_action=payload.get("user_action"),  # type: ignore[arg-type]
            auth_url=payload.get("auth_url"),  # type: ignore[arg-type]
            output=run.output,
            data=data,
        )

    # -- subcomandos ------------------------------------------------------
    def _exec_raw(
        self,
        code: str,
        timeout: float = DEFAULT_SYNC_EXEC_TIMEOUT_SEC,
        idle_timeout: Optional[float] = None,
        interrupt_on_timeout: bool = True,
        label: str = "exec",
        echo: Optional[bool] = None,
    ) -> Tuple[ColabGuardOutcome, GuardedRun]:
        validate_sync_timeout(timeout)
        argv = [self.colab_bin, "exec", "-s", self.session, "--timeout", str(int(timeout))]
        run = run_guarded(
            argv,
            stdin_text=wrap_remote_code(code),
            hard_timeout=timeout + 30.0,  # margem para conexão/autenticação do CLI
            idle_timeout=idle_timeout,
            echo=self.echo if echo is None else echo,
        )
        outcome = self._from_run(run, label, expect_sentinel=True)
        if outcome.status in ("timeout", "idle_timeout") and interrupt_on_timeout:
            intr = self._bridge("interrupt", timeout=60.0)
            outcome = outcome.model_copy(
                update={"data": {**outcome.data, "kernel_interrupt": intr.status, **intr.data}}
            )
        return outcome, run

    def exec_code(
        self,
        code: str,
        timeout: float = DEFAULT_SYNC_EXEC_TIMEOUT_SEC,
        idle_timeout: Optional[float] = None,
        interrupt_on_timeout: bool = True,
        label: str = "exec",
    ) -> ColabGuardOutcome:
        outcome, _ = self._exec_raw(
            code,
            timeout=timeout,
            idle_timeout=idle_timeout,
            interrupt_on_timeout=interrupt_on_timeout,
            label=label,
        )
        return self._finalize(outcome)

    def drive(self, mount_path: str = "/content/drive", timeout: float = 240.0) -> ColabGuardOutcome:
        outcome = self._bridge("drive-mount", timeout=timeout, extra=["--mount-path", mount_path])
        if outcome.status == "ok" and self.register_blockers:
            try:
                resolved = resolve_blockers_with_prefix("blocker_colab_drive", self.state_path, self.user_md_path)
                if resolved:
                    outcome = outcome.model_copy(update={"data": {**outcome.data, "resolved_blockers": resolved}})
            except Exception:
                pass
        return self._finalize(outcome)

    def interrupt(self) -> ColabGuardOutcome:
        return self._finalize(self._bridge("interrupt", timeout=60.0))

    def preflight(self) -> ColabGuardOutcome:
        start = time.monotonic()
        checks: Dict[str, object] = {}
        sess = self._bridge("session-info", timeout=90.0)
        checks["session"] = {"status": sess.status, **sess.data}
        if sess.status != "ok":
            return self._finalize(sess.model_copy(update={"command": "preflight", "data": checks}))

        ping, _ = self._exec_raw(
            "print('pong')", timeout=45.0, interrupt_on_timeout=False, label="preflight:ping", echo=False
        )
        checks["kernel"] = {"status": ping.status, "responsive": ping.status == "ok"}

        drive = self._bridge("drive-status", timeout=90.0)
        checks["drive"] = {"status": drive.status, **drive.data}

        worst = next(
            (o for o in (drive, ping) if o.status != "ok"),
            None,
        )
        elapsed = time.monotonic() - start
        if worst is None:
            return self._finalize(build_outcome("ok", "preflight", elapsed, data=checks))
        return self._finalize(
            build_outcome(
                worst.status,
                "preflight",
                elapsed,
                reason=worst.reason or ("kernel_busy" if worst is ping else None),
                user_action=worst.user_action,
                auth_url=worst.auth_url,
                output=worst.output_tail,
                data=checks,
            )
        )

    def _job_call(self, code: str, label: str, timeout: float = 60.0) -> Tuple[ColabGuardOutcome, Optional[Dict]]:
        outcome, run = self._exec_raw(code, timeout=timeout, label=label, echo=False)
        payload = parse_prefixed_json(run.output, JOB_RESULT_PREFIX)
        return outcome, payload

    def launch(self, job: str, command: str, cwd: str, job_dir: str) -> ColabGuardOutcome:
        outcome, payload = self._job_call(build_launch_code(job, command, cwd, job_dir), f"launch:{job}")
        if outcome.status != "ok" or payload is None:
            return self._finalize(outcome)
        status = "ok"
        if payload.get("job_state") == "already_running":
            status = "job_running"
        return self._finalize(build_outcome(status, f"launch:{job}", outcome.elapsed_sec, data=payload))

    def job_status(self, job: str, job_dir: str, tail_lines: int) -> ColabGuardOutcome:
        outcome, payload = self._job_call(build_job_status_code(job, job_dir, tail_lines), f"job-status:{job}")
        if outcome.status != "ok" or payload is None:
            return self._finalize(outcome)
        state = payload.get("job_state")
        tail = str(payload.get("tail", ""))
        status_map = {
            "succeeded": "ok",
            "failed": "remote_error",
            "died": "remote_error",
            "running": "job_running",
            "not_found": "job_not_found",
        }
        status = status_map.get(str(state), "remote_error")
        reason = None if status in ("ok", "job_running") else f"job_{state}"
        user_action = auth_url = None
        found = detect_interaction_request(tail) if status == "job_running" else None
        if found:
            status, reason, user_action, auth_url = "needs_user", found[0], found[1], found[2]
        if self.echo and tail:
            sys.stdout.write(tail + "\n")
        return self._finalize(
            build_outcome(
                status,
                f"job-status:{job}",
                outcome.elapsed_sec,
                reason=reason,
                user_action=user_action,
                auth_url=auth_url,
                output=tail,
                data={k: v for k, v in payload.items() if k != "tail"},
            )
        )

    def job_kill(self, job: str, job_dir: str) -> ColabGuardOutcome:
        outcome, payload = self._job_call(build_job_kill_code(job, job_dir), f"job-kill:{job}")
        if outcome.status != "ok" or payload is None:
            return self._finalize(outcome)
        status = "job_not_found" if payload.get("job_state") == "not_found" else "ok"
        return self._finalize(build_outcome(status, f"job-kill:{job}", outcome.elapsed_sec, data=payload))

    def cli(self, args: Sequence[str], timeout: float = DEFAULT_CLI_PASSTHROUGH_TIMEOUT_SEC) -> ColabGuardOutcome:
        blocked = {"exec", "drivemount", "auth", "repl", "console", "ssh"}
        if args and args[0] in blocked:
            return build_outcome(
                "cli_error",
                f"cli {' '.join(args)}",
                0.0,
                reason="interactive_subcommand_forbidden",
                output=f"Subcomando '{args[0]}' é proibido via passthrough (Spec 06). Use exec/drive/launch do guarda.",
            )
        run = run_guarded([self.colab_bin, *args], hard_timeout=timeout, echo=self.echo)
        return self._finalize(self._from_run(run, f"cli {' '.join(args)}", expect_sentinel=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _emit_result(outcome: ColabGuardOutcome) -> int:
    sys.stdout.write("\n" + RESULT_PREFIX + outcome.model_dump_json() + "\n")
    sys.stdout.flush()
    return outcome.exit_code


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--session", "-s", default=DEFAULT_SESSION)
    common.add_argument("--colab-bin", default=DEFAULT_COLAB_BIN)
    common.add_argument("--step", default=None, help="Etapa para registrar bloqueios (padrão: current_step_id).")
    common.add_argument("--no-blocker", action="store_true", help="Não registrar bloqueio em USER.md.")
    common.add_argument("--open-browser", action="store_true", help="Abre a URL de autorização no navegador (macOS).")
    common.add_argument("--quiet", action="store_true", help="Não ecoar a saída remota.")

    parser = argparse.ArgumentParser(description="Operação remota não-bloqueante do Google Colab (Spec 06).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("preflight", parents=[common], help="Sessão viva + kernel responsivo + Drive montado.")
    p_drive = sub.add_parser("drive", parents=[common], help="Monta o Drive sem bloquear (exit 10 se exigir autorização).")
    p_drive.add_argument("--mount-path", default="/content/drive")
    sub.add_parser("interrupt", parents=[common], help="Interrompe kernels ocupados.")

    p_exec = sub.add_parser("exec", parents=[common], help="Executa código Python no kernel com limite de tempo.")
    g = p_exec.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", "-f")
    g.add_argument("--code", "-c")
    p_exec.add_argument("--timeout", type=float, default=DEFAULT_SYNC_EXEC_TIMEOUT_SEC)
    p_exec.add_argument("--idle-timeout", type=float, default=None)
    p_exec.add_argument("--no-interrupt", action="store_true")

    p_launch = sub.add_parser("launch", parents=[common], help="Inicia job longo em background na VM.")
    p_launch.add_argument("--job", required=True)
    p_launch.add_argument("--cmd", dest="job_cmd", required=True)
    p_launch.add_argument("--cwd", default=DEFAULT_REMOTE_CWD)
    p_launch.add_argument("--job-dir", default=DEFAULT_JOB_DIR)

    p_status = sub.add_parser("job-status", parents=[common], help="Consulta job em background.")
    p_status.add_argument("--job", required=True)
    p_status.add_argument("--job-dir", default=DEFAULT_JOB_DIR)
    p_status.add_argument("--tail", type=int, default=30)

    p_kill = sub.add_parser("job-kill", parents=[common], help="Encerra job em background.")
    p_kill.add_argument("--job", required=True)
    p_kill.add_argument("--job-dir", default=DEFAULT_JOB_DIR)

    p_cli = sub.add_parser("cli", parents=[common], help="Passthrough limitado para o colab CLI (ex.: cli -- sessions).")
    p_cli.add_argument("--timeout", type=float, default=DEFAULT_CLI_PASSTHROUGH_TIMEOUT_SEC)
    p_cli.add_argument("cli_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    guard = ColabGuard(
        session=args.session,
        colab_bin=args.colab_bin,
        step_id=args.step,
        register_blockers=not args.no_blocker,
        open_browser=args.open_browser,
        echo=not args.quiet,
    )
    try:
        if args.cmd == "preflight":
            outcome = guard.preflight()
        elif args.cmd == "drive":
            outcome = guard.drive(mount_path=args.mount_path)
        elif args.cmd == "interrupt":
            outcome = guard.interrupt()
        elif args.cmd == "exec":
            code = Path(args.file).read_text(encoding="utf-8") if args.file else args.code
            outcome = guard.exec_code(
                code, timeout=args.timeout, idle_timeout=args.idle_timeout, interrupt_on_timeout=not args.no_interrupt
            )
        elif args.cmd == "launch":
            outcome = guard.launch(args.job, args.job_cmd, args.cwd, args.job_dir)
        elif args.cmd == "job-status":
            outcome = guard.job_status(args.job, args.job_dir, args.tail)
        elif args.cmd == "job-kill":
            outcome = guard.job_kill(args.job, args.job_dir)
        elif args.cmd == "cli":
            cli_args = args.cli_args[1:] if args.cli_args[:1] == ["--"] else list(args.cli_args)
            outcome = guard.cli(cli_args, timeout=args.timeout)
        else:  # pragma: no cover
            raise ValueError(args.cmd)
    except ValueError as exc:
        outcome = build_outcome("cli_error", args.cmd, 0.0, reason="invalid_arguments", output=str(exc))
        sys.stderr.write(f"[colab_guard] {exc}\n")
    return _emit_result(outcome)


if __name__ == "__main__":
    sys.exit(main())
