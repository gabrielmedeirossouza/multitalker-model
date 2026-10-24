# USER.md — Painel de Status e Ações do Usuário

> Arquivo gerado e sincronizado automaticamente via `python -m src.state`. Sem prosa desnecessária.

## 1. Resumo Executivo

- **Projeto:** `parakeet-ptbr-multitalker` (`nvidia/multitalker-parakeet-streaming-0.6b-v1`)
- **Status Geral:** 🟢 Pronto para Iniciar / Continuar
- **Etapa Atual:** `step_05_colab_stage1_warmup` — Treinamento Fase 1 (Warm-up: Encoder Congelado, Decoder/Joint PT-BR) na GPU A100 (`colab_a100`)
- **Última Atualização (UTC):** `2026-10-24T19:12:21Z`

## 2. O Que o Usuário Precisa Fazer Agora (Impedimentos / Ações)

- **Nenhum impedimento bloqueante no momento.** Os agentes de IA podem prosseguir com a etapa atual.
- **Pré-requisitos futuros (Etapas 04 a 07):** Ter conta Google Colab Pro/Pro+ com GPU **NVIDIA A100** e espaço no Google Drive (`/content/drive/MyDrive/parakeet_ptbr`).

## 3. Progresso das Etapas (Pipeline Stateful)

| # | ID da Etapa | Ambiente | Status | Artefatos / Métricas |
|---|---|---|---|---|
| 0 | `step_00_audit_and_contracts` | `local_cpu` | ✅ Concluído | `contract_tests_passed=28`, specs: `specs/00..05_*.md`, schemas: `src/schemas.py`, state_file: `state/pipeline_state.json`, knowledge_base: `state/AGENT_KNOWLEDGE.md` |
| 1 | `step_01_data_prep_clean` | `colab_cpu` | ✅ Concluído | clean_train_manifest: `/content/local_dataset/clean_train_manifest.json`, clean_val_manifest: `/content/local_dataset/clean_val_manifest.json` |
| 2 | `step_02_tokenizer_ptbr` | `colab_cpu` | ✅ Concluído | tokenizer_dir: `/content/drive/MyDrive/parakeet_ptbr/tokenizer_ptbr_1024` |
| 3 | `step_03_mixer_multitalker` | `colab_cpu` | ✅ Concluído | multitalker_train_manifest: `/content/local_dataset/multitalker_train_manifest.json`, multitalker_val_manifest: `/content/local_dataset/multitalker_val_manifest.json` |
| 4 | `step_04_pack_and_upload_drive` | `colab_cpu` | ✅ Concluído | colab_bundle_tar_gz: `/content/drive/MyDrive/parakeet_ptbr/ptbr_dataset_16khz.tar.gz` |
| 5 | `step_05_colab_stage1_warmup` | `colab_a100` | ⏳ Pendente | — |
| 6 | `step_06_colab_stage2_multitalker` | `colab_a100` | ⏳ Pendente | — |
| 7 | `step_07_eval_and_export` | `colab_a100` | ⏳ Pendente | — |

## 4. Handover para o Próximo Agente de IA

- **Resumo da Última Sessão:** Etapas 01 a 04 concluídas no Colab. Pacote salvo em /content/drive/MyDrive/parakeet_ptbr/ptbr_dataset_16khz.tar.gz.
- **Próxima Ação do Agente:** Iniciar o treinamento da Fase 1 (warmup) na GPU A100 (Célula 4 do colab_runner.ipynb).
- **Comando Exato para o Próximo Agente:**
  ```bash
  python -m src.trainer --stage warmup --manifest /content/local_dataset/clean_train_manifest.json --val_manifest /content/local_dataset/clean_val_manifest.json --checkpoint_dir /content/drive/MyDrive/parakeet_checkpoints/exp_v1_warmup --update_state
  ```
- **Alertas Técnicos Críticos:**
  - NUNCA baixe datasets de áudio reais no Mac local; use sempre a VM do Colab (/content/local_dataset) + Google Drive.
  - No ambiente local (Mac), use sempre os binários `.venv/bin/python` e `.venv/bin/pytest`.
  - Mantenha `max_duration <= 20.0` segundos para evitar descarte ou OOM na Joint Network 4D na A100.
  - Nunca passe vírgula em string separada por vírgulas para `user_defined_symbols` do SentencePiece.
