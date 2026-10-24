"""Ponte NÃO-INTERATIVA para os internals do google-colab-cli (Spec 06).

Este arquivo é executado pelo interpretador do `google-colab-cli`
(ex.: ~/.local/share/uv/tools/google-colab-cli/bin/python), NÃO pelo `.venv` do projeto.
Por isso ele não importa nada de `src.*` e só importa `colab_cli` de forma preguiçosa
(dentro das funções), permitindo que os testes do projeto importem as constantes.

Modos:
  session-info  -> confirma que a sessão existe no backend (token renovado).
  drive-status  -> Drive montado? (API REST de contents, sem kernel) / autorizado? (dryrun).
  drive-mount   -> monta o Drive SEM bloquear: se não autorizado, retorna needs_user com a URL.
  interrupt     -> interrompe kernels ocupados (libera a fila após timeout).

Protocolo: a última linha impressa é `COLAB_BRIDGE_RESULT {json}` e o exit code segue
COLAB_GUARD_EXIT_CODES (src/schemas.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, Optional, Tuple

RESULT_PREFIX = "COLAB_BRIDGE_RESULT "

# Deve coincidir com src.schemas.COLAB_GUARD_EXIT_CODES (validado em tests/test_colab_guard.py).
EXIT_CODES: Dict[str, int] = {
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

DRIVE_MOUNT_EXEC_TIMEOUT_SEC = 180.0
HTTP_TIMEOUT_SEC = 30.0


class _NeedsUser(Exception):
    def __init__(self, reason: str, user_action: str, auth_url: Optional[str] = None):
        super().__init__(reason)
        self.reason = reason
        self.user_action = user_action
        self.auth_url = auth_url


def emit(status: str, **data: Any) -> None:
    payload = {"status": status, **data}
    sys.stdout.write("\n" + RESULT_PREFIX + json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()
    sys.exit(EXIT_CODES[status])


# ---------------------------------------------------------------------------
# Sessão e HTTP
# ---------------------------------------------------------------------------


def _load_state(auth_provider: Optional[str]):
    from colab_cli.common import state  # import preguiçoso (interpretador do colab-cli)

    if auth_provider:
        from colab_cli.auth import AuthProvider

        state.auth_provider = AuthProvider(auth_provider)
    return state


def _load_session(state, name: str, verify_backend: bool = False):
    s = state.get_session(name, ignore_missing_session=True)
    if s is None:
        emit("session_lost", reason="session_not_found", session=name)
    if verify_backend:
        endpoints = {a.endpoint for a in state.client.list_assignments()}
        if s.endpoint not in endpoints:
            state.prune_session(name)
            emit("session_lost", reason="assignment_gone", session=name, endpoint=s.endpoint)
    return s


def _api_params(s) -> Dict[str, str]:
    return {"authuser": "0", "colab-runtime-proxy-token": s.token}


def _connect_warm_runtime(state, s):
    """Conecta ao kernel com o mesmo retry de `execution.py` para `Connection was lost`."""
    from colab_cli.runtime import ColabRuntime

    def on_started(kid: str) -> None:
        s.kernel_id = kid
        state.store.add(s)

    def on_sess_started(sid: str) -> None:
        s.session_id = sid
        state.store.add(s)

    last_exc: Optional[Exception] = None
    for attempt in range(3):
        runtime = ColabRuntime(
            s.url,
            s.token,
            session_name=s.name,
            history=state.history,
            kernel_id=s.kernel_id,
            session_id=None,
            on_kernel_started=on_started,
            on_session_started=on_sess_started,
        )
        try:
            runtime.execute_code("import os; os.chdir('/content')", timeout=20.0)
            return runtime
        except Exception as exc:
            runtime.stop()
            last_exc = exc
            if attempt < 2 and "Connection was lost" in str(exc):
                s.kernel_id = None
                s.session_id = None
                state.store.add(s)
                continue
            raise
    raise last_exc or RuntimeError("Falha ao conectar ao kernel Colab.")


def _drive_is_mounted(state, s, mount_path: str) -> bool:
    """Verifica montagem via `/proc/mounts` + `os.path.isdir` no kernel (nunca bloqueia em FUSE)."""
    code = (
        "import os\n"
        "with open('/proc/mounts') as _f:\n"
        "    _m = _f.read()\n"
        f"_ok = ({mount_path!r} in _m) and os.path.isdir({(mount_path.rstrip('/') + '/MyDrive')!r})\n"
        "print('__MOUNTED__=' + ('1' if _ok else '0'))\n"
    )
    runtime = _connect_warm_runtime(state, s)
    try:
        outputs = runtime.execute_code(code, timeout=20.0)
    except Exception:
        return False
    finally:
        runtime.stop()
    text = "".join(o.get("text", "") for o in outputs if isinstance(o, dict))
    return "__MOUNTED__=1" in text


def _propagation_request(state, s, dryrun: bool) -> Tuple[bool, Optional[str], int]:
    """Chama `credentials-propagation` (`dfs_ephemeral`) do Colab SEM esperar /dev/tty.

    Nota importante: o endpoint do Google Colab com `dryrun=true` SEMPRE retorna
    `{"success": false, "unauthorized_redirect_uri": "..."}` mesmo quando a conta já
    possui consentimento OAuth ativo ("O app Google Drive for desktop já tem algum acesso").
    Já `dryrun=false` retorna `{"success": true}` imediatamente quando o consentimento existe,
    e `{"success": false}` apenas quando o usuário realmente precisa aprovar no navegador.
    """
    from colab_cli.auth import get_credentials
    from colab_cli.utils import get_status_code

    url = f"{state.client.colab_domain}/tun/m/credentials-propagation/{s.endpoint}"
    params = {
        "authuser": "0",
        "authtype": "dfs_ephemeral",
        "version": "2",
        "dryrun": "true" if dryrun else "false",
        "propagate": "true",
        "record": "false",
    }
    creds = get_credentials(state.client_oauth_config, provider=state.auth_provider)
    # O token XSRF é obtido sempre com dryrun=true (mesmo comportamento do colab-cli).
    resp = creds.request("GET", url, params={**params, "dryrun": "true"})
    token = (
        json.loads(resp.text.split("\n", 1)[-1]).get("token")
        if get_status_code(resp) == 200
        else None
    )
    resp = creds.request(
        "POST",
        url,
        params=params,
        headers={"x-goog-colab-token": token},
        files={"file_id": (None, "empty.ipynb")},
    )
    status = get_status_code(resp)
    if status != 200:
        return False, None, status
    try:
        data = json.loads(resp.text.split("\n", 1)[-1])
    except ValueError:
        data = {}
    return bool(data.get("success")), (data.get("unauthorized_redirect_uri") or None), status


def _try_propagate_or_get_auth_url(state, s) -> Tuple[bool, Optional[str], int]:
    """Tenta propagar credenciais (`dryrun=false`); se falhar, obtém a `unauthorized_redirect_uri` via `dryrun=true`."""
    ok, uri, http = _propagation_request(state, s, dryrun=False)
    if ok:
        return True, None, http
    if not uri:
        _, uri, _ = _propagation_request(state, s, dryrun=True)
    return False, uri, http


def _drive_auth_needed(uri: Optional[str]) -> _NeedsUser:
    return _NeedsUser(
        reason="drive_auth",
        user_action=(
            "Abra a URL de autorização do Google Drive no navegador, aprove o acesso para esta VM do Colab "
            "e responda 'ok' ao agente."
        ),
        auth_url=uri,
    )


# ---------------------------------------------------------------------------
# Modos
# ---------------------------------------------------------------------------


def cmd_session_info(state, name: str) -> None:
    s = _load_session(state, name, verify_backend=True)
    emit(
        "ok",
        session=s.name,
        endpoint=s.endpoint,
        variant=str(getattr(s, "variant", "")),
        accelerator=str(getattr(s, "accelerator", "")),
        machine_shape=str(getattr(s, "machine_shape", "")),
        token_expires_at=str(getattr(s, "token_expires_at", "")),
        kernel_id=getattr(s, "kernel_id", None),
    )


def cmd_drive_status(state, name: str, mount_path: str) -> None:
    s = _load_session(state, name)
    if _drive_is_mounted(state, s, mount_path):
        emit("ok", mounted=True, authorized=True, mount_path=mount_path)
    authorized, uri, http = _try_propagate_or_get_auth_url(state, s)
    if not authorized:
        raise _drive_auth_needed(uri)
    emit("ok", mounted=False, authorized=True, mount_path=mount_path, http_status=http)


def cmd_drive_mount(state, name: str, mount_path: str, exec_timeout: float) -> None:
    s = _load_session(state, name)
    if _drive_is_mounted(state, s, mount_path):
        emit("ok", mounted=True, already_mounted=True, mount_path=mount_path)

    authorized, uri, _ = _try_propagate_or_get_auth_url(state, s)
    if not authorized:
        # Nada é enviado ao kernel: nenhum drive.mount() fica pendurado esperando resposta.
        raise _drive_auth_needed(uri)

    events: Dict[str, Any] = {"propagated": False, "error": None, "unauthorized_uri": None}

    def non_blocking_hook(msg: Dict[str, Any], wsclient: Any) -> bool:
        content = msg.get("content", {})
        if content.get("request", {}).get("authType") != "dfs_ephemeral":
            return False
        msg_id = msg.get("metadata", {}).get("colab_msg_id")
        ok, uri_err, http = _try_propagate_or_get_auth_url(state, s)
        if not ok:
            events["unauthorized_uri"] = uri_err
            events["error"] = f"propagation_http_{http}"
            return True  # não responde; o execute_code termina pelo timeout curto
        reply = wsclient.session.msg(
            "input_reply", {"value": {"type": "colab_reply", "colab_msg_id": msg_id}}
        )
        if "header" in msg:
            reply["parent_header"] = msg["header"]
        wsclient.stdin_channel.send(reply)
        events["propagated"] = True
        return True

    def refuse_stdin(prompt: str) -> str:
        raise _NeedsUser(
            reason="stdin_prompt",
            user_action=f"O kernel remoto pediu entrada interativa durante drive.mount(): {prompt!r}.",
        )

    runtime = _connect_warm_runtime(state, s)
    runtime.colab_request_hook = non_blocking_hook
    code = (
        "from google.colab import drive\n"
        f"drive.mount({mount_path!r})\n"
        "import os\n"
        f"print('__MOUNT_OK__=' + ('1' if os.path.isdir({(mount_path.rstrip('/') + '/MyDrive')!r}) else '0'))\n"
    )
    outputs = []
    try:
        outputs = runtime.execute_code(
            code, allow_stdin=True, stdin_hook=refuse_stdin, timeout=exec_timeout
        )
    except _NeedsUser:
        raise
    except Exception as exc:  # TimeoutError do jupyter_kernel_client etc.
        events["error"] = events["error"] or f"{type(exc).__name__}: {exc}"
    finally:
        runtime.stop()

    if events["unauthorized_uri"]:
        raise _drive_auth_needed(events["unauthorized_uri"])

    text = "".join(o.get("text", "") for o in outputs if isinstance(o, dict))
    if "__MOUNT_OK__=1" in text or "Mounted at" in text or _drive_is_mounted(state, s, mount_path):
        emit("ok", mounted=True, already_mounted=False, mount_path=mount_path, propagated=events["propagated"])

    emit(
        "remote_error",
        reason="drive_mount_failed",
        error=events["error"],
        propagated=events["propagated"],
        kernel_output=text[-2000:],
    )


def cmd_interrupt(state, name: str) -> None:
    import requests

    s = _load_session(state, name)
    base = s.url.rstrip("/")
    resp = requests.get(f"{base}/api/kernels", params=_api_params(s), timeout=HTTP_TIMEOUT_SEC)
    if resp.status_code in (401, 403, 404):
        emit("session_lost", reason=f"kernels_api_http_{resp.status_code}")
    resp.raise_for_status()
    interrupted = []
    for k in resp.json():
        if k.get("execution_state") in ("busy", "starting") or k.get("id") == s.kernel_id:
            r = requests.post(
                f"{base}/api/kernels/{k['id']}/interrupt", params=_api_params(s), timeout=HTTP_TIMEOUT_SEC
            )
            interrupted.append({"id": k["id"], "state": k.get("execution_state"), "http": r.status_code})
    emit("ok", interrupted=interrupted)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Ponte não-interativa para o google-colab-cli (Spec 06).")
    parser.add_argument("mode", choices=["session-info", "drive-status", "drive-mount", "interrupt"])
    parser.add_argument("--session", "-s", required=True)
    parser.add_argument("--mount-path", default="/content/drive")
    parser.add_argument("--exec-timeout", type=float, default=DRIVE_MOUNT_EXEC_TIMEOUT_SEC)
    parser.add_argument("--auth", default=None, choices=[None, "oauth2", "adc"])
    args = parser.parse_args(argv)

    try:
        state = _load_state(args.auth)
        if args.mode == "session-info":
            cmd_session_info(state, args.session)
        elif args.mode == "drive-status":
            cmd_drive_status(state, args.session, args.mount_path)
        elif args.mode == "drive-mount":
            cmd_drive_mount(state, args.session, args.mount_path, args.exec_timeout)
        elif args.mode == "interrupt":
            cmd_interrupt(state, args.session)
    except _NeedsUser as nu:
        emit("needs_user", reason=nu.reason, user_action=nu.user_action, auth_url=nu.auth_url)
    except SystemExit:
        raise
    except ImportError as exc:
        emit("cli_error", reason="colab_cli_import_failed", error=str(exc))
    except EOFError as exc:
        emit(
            "needs_user",
            reason="colab_cli_oauth",
            user_action="O google-colab-cli pediu login OAuth. Rode `~/.local/bin/colab sessions` no seu terminal e conclua o login.",
            error=str(exc),
        )
    except Exception as exc:
        emit("cli_error", reason="bridge_exception", error=f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
