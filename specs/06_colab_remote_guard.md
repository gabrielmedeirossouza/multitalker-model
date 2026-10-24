# Spec 06: Operação Remota Não-Bloqueante do Google Colab (`src.colab_guard`)

## 1. Objetivo
Garantir que **nenhuma** interação de um agente de IA com o Google Colab deixe o terminal travado indefinidamente.
Toda chamada remota deve **terminar em tempo limitado** com um **código de saída determinístico** e, quando depender
do humano (ex.: autorização OAuth do Google Drive), o agente deve **parar imediatamente e comunicar o usuário**,
registrando o bloqueio em `state/pipeline_state.json` / `USER.md` (`Spec 05`).

---

## 2. Diagnóstico (Causas-Raiz dos Travamentos Observados)

| # | Origem | Comportamento observado |
|---|---|---|
| 1 | `colab drivemount` | Cada VM nova exige autorização `dfs_ephemeral`. O CLI imprime a URL e executa `open('/dev/tty').readline()` (“Press Enter after you have granted access...”), bloqueando o terminal do agente **para sempre**. |
| 2 | `drive.mount()` dentro de `colab exec` | `colab exec` **não** instala o *hook* `colab_request`; o kernel envia o pedido de autenticação e espera uma resposta que nunca chega. O processo só termina no `--timeout` (frequentemente configurado em horas). |
| 3 | `colab exec --timeout 3600+` para tarefas longas | Treino/preparação síncronos prendem o terminal; se o kernel sofre SIGABRT/SIGSEGV o `execute_reply` nunca chega (`DISC-014`). |
| 4 | OAuth do próprio CLI expirado | `colab-cli` imprime “To authorize colab-cli, visit this URL” e chama `input()`. |
| 5 | `colab repl` / `colab console` / `colab auth` | Exigem TTY e esperam entrada humana. |

---

## 3. Decisão de Arquitetura: CLI + Guarda Local (e não `colab-mcp`)

O `googlecolab/colab-mcp` foi avaliado e **não** é adotado como caminho principal:

- Ele é um *proxy* para um notebook **aberto no navegador** do usuário (abre `colab.research.google.com` e espera a conexão do frontend por 60 s). Se a aba fechar, o notebook perder foco ou o laptop dormir, a conexão cai.
- Exige suporte do cliente MCP a `notifications/tools/list_changed`; as ferramentas de notebook só aparecem depois da conexão do navegador.
- A execução de uma célula continua sendo uma chamada **síncrona** de ferramenta: um `drive.mount()` aguardando o popup ou um treino de horas também bloquearia o agente — o problema de fundo (chamadas sem limite) permanece.
- O fluxo headless já existente (`colab new/stop` alocando A100 via CLI) seria perdido.

A solução adotada é um **wrapper local** (`src/colab_guard.py`) sobre o `~/.local/bin/colab`, mais uma **ponte**
(`src/colab_cli_bridge.py`) executada pelo interpretador do `google-colab-cli` para operações que o CLI só oferece
de forma interativa (montagem do Drive, interrupção de kernel, checagem de sessão).

---

## 4. Contrato de Resultado (`ColabGuardOutcome` em `src/schemas.py`)

Todo subcomando de `python -m src.colab_guard` imprime, como **última linha**, `COLAB_GUARD_RESULT {json}` e sai com:

| `status` | Exit code | Significado / Ação obrigatória do agente |
|---|---|---|
| `ok` | `0` | Sucesso. |
| `remote_error` | `1` | Código remoto falhou (traceback, exit code ≠ 0, job `failed`/`died`). Diagnosticar e corrigir. |
| `needs_user` | `10` | **Depende do humano.** Bloqueio já registrado em `USER.md`. **Parar, informar o usuário (URL/ação) e encerrar o turno.** Não tentar novamente em loop. |
| `timeout` | `11` | Limite total atingido. O kernel remoto é interrompido automaticamente (`interrupt`). |
| `idle_timeout` | `12` | Nenhuma saída por `--idle-timeout` s. Mesmo tratamento de `timeout`. |
| `session_lost` | `13` | VM/sessão inexistente ou desalocada. Informar o usuário antes de alocar nova VM (custo). |
| `cli_error` | `14` | Falha local (CLI ausente, credencial, bug na ponte). |
| `job_running` | `20` | Job em background ainda em execução (resposta de `job-status`). |
| `job_not_found` | `21` | Job inexistente no diretório de jobs. |

Campos mínimos: `status`, `exit_code`, `command`, `elapsed_sec`, `reason`, `user_action`, `auth_url`, `blocker_id`,
`output_tail`, `data`.

---

## 5. Regras do Executor Guardado (`run_guarded`)

1. O subprocesso do CLI roda em **nova sessão** (`start_new_session=True`), sem TTY controlador (`/dev/tty` falha com
   `ENXIO`) e com `stdin` fechado após enviar o código — nenhum `readline()` pode bloquear.
2. **Limite total obrigatório** (`--timeout`, padrão `120 s`, máximo `MAX_SYNC_EXEC_TIMEOUT_SEC = 600 s`).
   Tarefas mais longas **devem** usar `launch` + `job-status`.
3. **Limite de inatividade opcional** (`--idle-timeout`).
4. **Detecção de interação humana** em fluxo contínuo (não só por linha, pois prompts não terminam em `\n`):
   padrões como `REQUIRED: Google Drive Authorization`, `Press Enter after you have granted access`,
   `To authorize colab-cli, visit this URL`, `Enter the authorization code`, `Enter verification code`,
   `Go to this URL in a browser`. Ao detectar: coletar a URL por até 2 s, matar o grupo de processos
   (`SIGTERM` → `SIGKILL`) e retornar `needs_user`.
5. Código remoto enviado por `exec` recebe um **sentinela** final (`__COLAB_GUARD_DONE__`); a ausência do sentinela
   com processo encerrado caracteriza `remote_error` (o `colab exec` sai com `0` mesmo com traceback remoto).
6. Em `timeout`/`idle_timeout` de `exec`, o guarda chama `interrupt` na ponte para liberar o kernel.
7. Em `needs_user`, o guarda registra automaticamente `add_blocker(id="blocker_colab_<reason>")` na etapa atual
   (`--step`, padrão `current_step_id`), a menos que `--no-blocker` seja passado. `--open-browser` abre a URL no Mac.

---

## 6. Google Drive Sem Bloqueio (`drive`)

1. Verifica se `/content/drive/MyDrive` está montado lendo `/proc/mounts` + `os.path.isdir` no kernel (com reconexão automática `_connect_warm_runtime` caso o `kernel_id` em cache lance `RuntimeError: Connection was lost.`, evitando timeouts da API `/api/contents`).
2. Se não montado, consulta `/tun/m/credentials-propagation/{endpoint}` com `dryrun=false` (`_try_propagate_or_get_auth_url`):
   - **Nota de protocolo Colab:** `dryrun=true` retorna `{"success": false, "unauthorized_redirect_uri": "..."}` mesmo quando a conta Google já concedeu acesso ao *Google Drive for desktop*; já `dryrun=false` retorna `{"success": true}` imediatamente quando o consentimento já existe.
   - **não autorizado** (`dryrun=false` retorna `success: false`) → busca a `unauthorized_redirect_uri` via `dryrun=true` e retorna `needs_user` com `auth_url`; **nada é executado no kernel**;
   - **autorizado** (`dryrun=false` retorna `success: true`) → executa `drive.mount()` com *hook* `colab_request` não-bloqueante (propaga credencial para o daemon `DriveFS` e envia `input_reply`), com limite de `180 s`, e confirma a montagem.
3. Após sucesso, bloqueios de Drive ativos (`blocker_colab_drive*`) são resolvidos automaticamente em `USER.md`.

---

## 7. Jobs Longos em Background (`launch`, `job-status`, `job-kill`)

- `launch --job <nome> --cmd "<comando>"` inicia o comando na VM com `setsid nohup bash -c ...` (desacoplado do kernel,
  `stdin=/dev/null`, `PYTHONUNBUFFERED=1`), grava `<dir>/<job>.log`, `<job>.json` (pid, cmd) e, ao terminar,
  `<job>.exitcode`. Retorna em segundos. Recusa iniciar se o mesmo job já estiver vivo.
- `job-status --job <nome> [--tail N]` retorna `job_running` (com `log_idle_sec` para detectar travamento), `ok`
  (exit 0), `remote_error` (exit ≠ 0 ou processo morto sem exit code) ou `job_not_found`. Padrões de interação
  humana no log retornam `needs_user`.
- `job-kill --job <nome>` encerra o grupo de processos do job.
- Diretório padrão: `/content/jobs`; para treinos, preferir `/content/drive/MyDrive/parakeet_ptbr/jobs` (sobrevive à VM).
- O agente monitora jobs com **timers** (`schedule`, ex.: a cada 10–15 min), nunca com `exec` bloqueante.

---

## 8. Regras Obrigatórias para Agentes

1. **Proibido** chamar diretamente `~/.local/bin/colab exec|drivemount|auth|repl|console`. Use `src.colab_guard`.
   Demais subcomandos do CLI (`new`, `stop`, `sessions`, `usage`, `status`, `ls`, `upload`, `download`) devem passar por
   `python -m src.colab_guard cli -- <args>` (limite padrão de `300 s`).
2. Invocar o guarda com `run_command` usando `WaitMsBeforeAsync ≤ 10000`; se for para background, aguardar a notificação.
3. Ao iniciar trabalho no Colab: `python -m src.colab_guard preflight` (sessão viva, kernel responsivo, Drive montado).
4. Exit code `10` ⇒ **parar e comunicar o usuário imediatamente** (mensagem com a URL/ação). Exit `11/12/13/14` ⇒
   diagnosticar; no máximo **uma** nova tentativa antes de reportar.

---

## 9. Critérios de Aceite (`tests/test_colab_guard.py`)

1. Os códigos de saída de `ColabGuardOutcome` coincidem com os da ponte (`src/colab_cli_bridge.py`).
2. `run_guarded` encerra em < 10 s um processo que imprime um prompt de autorização do Drive sem `\n` e fica
   esperando entrada, retornando `needs_user` com a URL extraída.
3. `run_guarded` retorna `timeout` / `idle_timeout` para processos silenciosos, e `ok` / `remote_error` conforme
   sentinela e exit code.
4. `needs_user` registra o bloqueio no `PipelineState` e regenera `USER.md`.
5. Timeouts síncronos acima de `MAX_SYNC_EXEC_TIMEOUT_SEC` são rejeitados.
6. Os códigos remotos gerados (`launch`, `job-status`, `job-kill`) compilam como Python válido.
