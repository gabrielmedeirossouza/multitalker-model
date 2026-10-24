# Spec 05: Orquestração Stateful entre Agentes de IA, `USER.md` e Conhecimento Incremental

## 1. Objetivo
Garantir que o desenvolvimento e o treinamento (divididos em múltiplas etapas e sessões) sejam **100% stateful e determinísticos para agentes de IA sem contexto prévio**, mantendo um painel enxuto para o usuário humano (`USER.md`) e um repositório de aprendizado incremental (`state/AGENT_KNOWLEDGE.md`).

---

## 2. Os 3 Artefatos de Continuidade (Fonte de Estado)

| Arquivo | Destinatário | Formato / Regra de Atualização |
|---|---|---|
| `state/pipeline_state.json` | **Agentes de IA (Máquina)** | JSON validado pelo schema `PipelineState` (`src/schemas.py`). Manipulado via `python -m src.state`. |
| `state/AGENT_KNOWLEDGE.md` | **Agentes de IA (Memória Incremental)** | Markdown estruturado. Todo agente que descobrir um gargalo, bug, script útil ou cuidado operacional **DEVE** registrar a descoberta via `python -m src.state log-discovery` ou edição direta. |
| `USER.md` | **Usuário Humano** | Markdown enxuto, sem prosa desnecessária, gerado/sincronizado automaticamente a partir de `state/pipeline_state.json` via `python -m src.state render-user-md`. Destaca no topo qualquer ação exigida do usuário (impedimentos/bloqueios). |

---

## 3. Etapas Canônicas do Pipeline (`CANONICAL_STEP_IDS`)

O pipeline é dividido em 8 etapas sequenciais obrigatórias (onde o computador local não armazena datasets de áudio, delegando `step_01` a `step_04` diretamente para a VM do Colab + Google Drive):

1. `step_00_audit_and_contracts` (`local_cpu`): Auditoria SDD, validação das specs (`00`–`05`), contratos Pydantic e testes de contrato em CPU com áudio sintético em memória (`< 0.3 GB`).
2. `step_01_data_prep_clean` (`colab_cpu`): Download e normalização (16 kHz mono, `<= 20.0s`) dos corpora PT-BR diretamente em `/content/local_dataset/` na VM do Colab, gerando `clean_train_manifest.json` e `clean_val_manifest.json`.
3. `step_02_tokenizer_ptbr` (`colab_cpu`): Treinamento e validação do tokenizador SentencePiece BPE PT-BR (`vocab_size = 1024`) no Colab, salvando em `/content/drive/MyDrive/parakeet_ptbr/tokenizer_ptbr_1024`.
4. `step_03_mixer_multitalker` (`colab_cpu`): Geração no Colab (`/content/local_dataset/`) do dataset sintético sobreposto em PT-BR (70% misturas de 2 locutores distintos com máscaras de 80 ms + 30% fala limpa) para `multitalker_train_manifest.json` e `multitalker_val_manifest.json`.
5. `step_04_pack_and_upload_drive` (`colab_cpu`): Empacotamento de `/content/local_dataset/` em `/content/drive/MyDrive/parakeet_ptbr/ptbr_dataset_16khz.tar.gz` para reutilização imediata nas sessões de GPU A100. (As etapas `01` a `04` são orquestradas idempotentemente por `python -m src.data_prep --ensure_colab_dataset`).
6. `step_05_colab_stage1_warmup` (`colab_a100`): Execução da Fase 1 (`warmup`: FastConformer congelado, treino do Decoder RNN-T + Joint Network) na GPU A100 do Google Colab.
7. `step_06_colab_stage2_multitalker` (`colab_a100`): Execução da Fase 2 (`multitalker`: modelo completo descongelado, *cache-aware* + *Speaker Kernels*) na GPU A100 do Google Colab.
8. `step_07_eval_and_export` (`colab_a100`): Avaliação final de WER/cpWER em PT-BR (fala limpa e sobreposta) via `src.evaluator` e exportação do modelo final `.nemo`.

---

## 4. Protocolo Obrigatório de Handover para Agentes de IA

Todo agente de IA ao iniciar uma sessão neste repositório **DEVE** seguir este protocolo:

### 4.1. Ao Iniciar a Sessão (Bootstrapping sem Contexto)
1. Executar `.venv/bin/python -m src.state status` (ou ler `state/pipeline_state.json` e `state/AGENT_KNOWLEDGE.md`).
2. Verificar se há bloqueios ativos (`overall_status == "blocked_by_user"`). Se houver e o usuário ainda não os resolveu, informar imediatamente ao usuário o que consta em `USER.md`.
3. Executar `.venv/bin/pytest tests/ -v` para garantir integridade dos contratos antes de qualquer ação.

### 4.2. Durante a Execução de uma Etapa (Idempotência, Retomada e Falhas)
1. **Idempotência e Retomada (`resume=True`):**
   - `src/data_prep.py` e `src/mixer.py` devem suportar retomada incremental (`resume=True` por padrão): caso o manifesto de saída já exista parcialmente de uma sessão anterior interrompida, o módulo valida as linhas existentes, pula os índices já processados e continua em modo *append* (`"a"`).
2. **Marcar início da etapa:**
   ```bash
   .venv/bin/python -m src.state start-step --step <step_id> --cmd "<comando>"
   ```
3. **Em caso de falha técnica inesperada:**
   ```bash
   .venv/bin/python -m src.state fail-step --step <step_id> --error "<causa raiz e diagnóstico para o próximo agente>"
   ```
4. **Espelhamento de Estado no Google Drive durante execução no Colab (`PARAKEET_DRIVE_STATE_DIR`):**
   - Como o disco `/content/multitalker-model` da VM do Colab é efêmero, sempre que `/content/drive/MyDrive` estiver montado (ou `PARAKEET_DRIVE_STATE_DIR` estiver definido), `src/state.py` espelha automaticamente `pipeline_state.json`, `AGENT_KNOWLEDGE.md` e `USER.md` no Google Drive (`/content/drive/MyDrive/parakeet_ptbr/state/`) e permite restaurá-los com `python -m src.state sync-from-drive`.
5. **Caso encontre um impedimento externo que dependa do humano:**
   ```bash
   .venv/bin/python -m src.state add-blocker --id <blocker_id> --step <step_id> --desc "<descrição curta>" --action "<o que o usuário precisa fazer>"
   ```
6. **Caso descubra um comportamento importante, armadilha técnica ou crie uma automação:**
   ```bash
   .venv/bin/python -m src.state log-discovery --category "<categoria>" --title "<título>" --problem "<problema>" --solution "<solução/regra>"
   ```

### 4.3. Ao Concluir uma Etapa ou Encerrar a Sessão
1. Atualizar o estado da etapa, registrar artefatos/métricas e definir o comando exato para o próximo agente:
   ```bash
   .venv/bin/python -m src.state complete-step \
       --step <step_id> \
       --next-step <next_step_id> \
       --next-action "<ação clara para o próximo agente>" \
       --next-cmd "<comando exato para o próximo agente>"
   ```
2. Validar sincronia entre `state/pipeline_state.json`, `USER.md` e `state/AGENT_KNOWLEDGE.md`:
   ```bash
   .venv/bin/python -m src.state verify
   .venv/bin/pytest tests/ -v
   ```

---

## 5. Critérios de Aceite (Testes Automatizados)

- `tests/test_state_workflow.py` deve validar que:
  1. `state/pipeline_state.json` existe, é válido segundo `PipelineState` e contém todas as 8 etapas canônicas em ordem.
  2. `USER.md` existe na raiz do repositório e seu conteúdo está estritamente sincronizado com `render_user_markdown(state)`.
  3. `state/AGENT_KNOWLEDGE.md` existe e contém o registro estruturado de descobertas incrementais dos agentes.
  4. As transições de estado (`start_step`, `complete_step`, `add_blocker`, `resolve_blocker`, `log_discovery`) mantêm a consistência invariante do `PipelineState` e regeneram `USER.md`.

---

## 6. Operação Remota Autônoma via Google Colab CLI (`~/.local/bin/colab`)

O ambiente local possui o `google-colab-cli` instalado em `~/.local/bin/colab` e autenticado via OAuth2 (`~/.config/colab-cli/token.json`). Agentes de IA podem alocar, inspecionar, executar código e trocar o tipo de runtime (`CPU High-RAM` $\leftrightarrow$ `GPU A100`) diretamente do terminal.

> **Obrigatório (`Spec 06`):** toda interação com o Colab passa por `.venv/bin/python -m src.colab_guard`, que impõe limites de tempo, detecta pedidos de autorização (ex.: Google Drive), registra o bloqueio em `USER.md` e retorna códigos de saída determinísticos (`10 = needs_user`: pare e comunique o usuário). É **proibido** chamar `colab exec|drivemount|auth|repl|console` diretamente. Os exemplos abaixo descrevem o CLI subjacente.

- **Listar sessões e saldo de Compute Units:** `.venv/bin/python -m src.colab_guard cli -- sessions` e `... cli -- usage`
- **Executar código Python remotamente:** `.venv/bin/python -m src.colab_guard exec -f script.py --timeout 120` (tarefas > 10 min: `launch` + `job-status`)
- **Montar o Google Drive sem bloquear:** `.venv/bin/python -m src.colab_guard drive --open-browser`
- **Adotar uma sessão já aberta no navegador (`[?]`):** Se o usuário já montou o Google Drive (`/content/drive`) no navegador, o agente pode adotar o endpoint listado em `state.client.list_assignments()` salvando um `SessionState` em `~/.config/colab-cli/sessions.json` via `/Users/gabriel.medeiros/.local/share/uv/tools/google-colab-cli/bin/python`.
- **Trocar tipo de runtime (de CPU High-RAM para GPU A100):** Encerrar a sessão CPU após concluir o `Step 04` (`.venv/bin/python -m src.colab_guard cli -- stop -s parakeet`) e criar uma nova sessão com GPU A100 (`.venv/bin/python -m src.colab_guard cli -- new -s parakeet --gpu A100 --high-mem`). Toda VM nova exige nova autorização do Drive: rode `.venv/bin/python -m src.colab_guard drive --open-browser` e, se sair com código `10`, comunique o usuário.
