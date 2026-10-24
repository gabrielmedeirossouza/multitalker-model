# USER.md — Painel de Status e Ações do Usuário

> Arquivo gerado e sincronizado automaticamente via `python -m src.state`. Sem prosa desnecessária.

## 1. Resumo Executivo

- **Projeto:** `parakeet-ptbr-multitalker` (`nvidia/multitalker-parakeet-streaming-0.6b-v1`)
- **Status Geral:** 🟢 Pronto para Iniciar / Continuar
- **Etapa Atual:** `step_01_data_prep_clean` — Download e Padronização 16kHz Mono dos Datasets PT-BR direto no Colab SSD (`colab_cpu`)
- **Última Atualização (UTC):** `2026-10-24T17:24:47Z`

## 2. O Que o Usuário Precisa Fazer Agora (Impedimentos / Ações)

- **Nenhum impedimento bloqueante no momento.** Os agentes de IA podem prosseguir com a etapa atual.
- **Pré-requisitos futuros (Etapas 04 a 07):** Ter conta Google Colab Pro/Pro+ com GPU **NVIDIA A100** e espaço no Google Drive (`/content/drive/MyDrive/parakeet_ptbr`).

## 3. Progresso das Etapas (Pipeline Stateful)

| # | ID da Etapa | Ambiente | Status | Artefatos / Métricas |
|---|---|---|---|---|
| 0 | `step_00_audit_and_contracts` | `local_cpu` | ✅ Concluído | `contract_tests_passed=28`, specs: `specs/00..05_*.md`, schemas: `src/schemas.py`, state_file: `state/pipeline_state.json`, knowledge_base: `state/AGENT_KNOWLEDGE.md` |
| 1 | `step_01_data_prep_clean` | `colab_cpu` | ⏳ Pendente | — |
| 2 | `step_02_tokenizer_ptbr` | `colab_cpu` | ⏳ Pendente | — |
| 3 | `step_03_mixer_multitalker` | `colab_cpu` | ⏳ Pendente | — |
| 4 | `step_04_pack_and_upload_drive` | `colab_cpu` | ⏳ Pendente | — |
| 5 | `step_05_colab_stage1_warmup` | `colab_a100` | ⏳ Pendente | — |
| 6 | `step_06_colab_stage2_multitalker` | `colab_a100` | ⏳ Pendente | — |
| 7 | `step_07_eval_and_export` | `colab_a100` | ⏳ Pendente | — |

## 4. Handover para o Próximo Agente de IA

- **Resumo da Última Sessão:** Auditoria completa e configuração 100% Colab+Drive concluídas (step_00). Zero GB de áudio será baixado no Mac local; Steps 01 a 04 rodam direto na VM do Colab via --ensure_colab_dataset.
- **Próxima Ação do Agente:** No Google Colab (Célula 3 do colab_runner.ipynb), executar --ensure_colab_dataset para baixar/normalizar em /content/local_dataset, treinar o tokenizador, gerar misturas 70/30 e salvar o .tar.gz no Google Drive.
- **Comando Exato para o Próximo Agente:**
  ```bash
  python -m src.data_prep --ensure_colab_dataset --dataset fleurs_pt --local_dataset_dir /content/local_dataset --drive_root_dir /content/drive/MyDrive/parakeet_ptbr --update_state
  ```
- **Alertas Técnicos Críticos:**
  - NUNCA baixe datasets de áudio reais no Mac local; use sempre a VM do Colab (/content/local_dataset) + Google Drive.
  - No ambiente local (Mac), use sempre os binários `.venv/bin/python` e `.venv/bin/pytest`.
  - Mantenha `max_duration <= 20.0` segundos para evitar descarte ou OOM na Joint Network 4D na A100.
  - Nunca passe vírgula em string separada por vírgulas para `user_defined_symbols` do SentencePiece.
