# USER.md — Painel de Status e Ações do Usuário

> Arquivo gerado e sincronizado automaticamente via `python -m src.state`. Sem prosa desnecessária.

## 1. Resumo Executivo

- **Projeto:** `parakeet-ptbr-multitalker` (`nvidia/multitalker-parakeet-streaming-0.6b-v1`)
- **Status Geral:** 🔄 Em Execução por Agente
- **Etapa Atual:** `step_01_data_prep_clean` — Download e Padronização 16kHz Mono dos Datasets PT-BR direto no Colab SSD (`colab_cpu`)
- **Última Atualização (UTC):** `2026-10-24T23:21:17Z`

## 2. O Que o Usuário Precisa Fazer Agora (Impedimentos / Ações)

- **Nenhum impedimento bloqueante no momento.** Os agentes de IA podem prosseguir com a etapa atual.
- **Pré-requisitos futuros (Etapas 04 a 07):** Ter conta Google Colab Pro/Pro+ com GPU **NVIDIA A100** e espaço no Google Drive (`/content/drive/MyDrive/parakeet_ptbr`).

## 3. Progresso das Etapas (Pipeline Stateful)

| # | ID da Etapa | Ambiente | Status | Artefatos / Métricas |
|---|---|---|---|---|
| 0 | `step_00_audit_and_contracts` | `local_cpu` | ✅ Concluído | `contract_tests_passed=28`, specs: `specs/00..05_*.md`, schemas: `src/schemas.py`, state_file: `state/pipeline_state.json`, knowledge_base: `state/AGENT_KNOWLEDGE.md` |
| 1 | `step_01_data_prep_clean` | `colab_cpu` | 🔄 Em Execução | clean_train_manifest: `/content/local_dataset/clean_train_manifest.json`, clean_val_manifest: `/content/local_dataset/clean_val_manifest.json` |
| 2 | `step_02_tokenizer_ptbr` | `colab_cpu` | ✅ Concluído | tokenizer_dir: `/content/drive/MyDrive/parakeet_ptbr/tokenizer_ptbr_1024` |
| 3 | `step_03_mixer_multitalker` | `colab_cpu` | ✅ Concluído | multitalker_train_manifest: `/content/local_dataset/multitalker_train_manifest.json`, multitalker_val_manifest: `/content/local_dataset/multitalker_val_manifest.json` |
| 4 | `step_04_pack_and_upload_drive` | `colab_cpu` | ✅ Concluído | colab_bundle_tar_gz: `/content/drive/MyDrive/parakeet_ptbr/ptbr_dataset_16khz.tar.gz` |
| 5 | `step_05_colab_stage1_warmup` | `colab_a100` | ⏳ Pendente | — |
| 6 | `step_06_colab_stage2_multitalker` | `colab_a100` | ⏳ Pendente | — |
| 7 | `step_07_eval_and_export` | `colab_a100` | ⏳ Pendente | — |

## 4. Handover para o Próximo Agente de IA

- **Resumo da Última Sessão:** Corpus da Fase 1 definido: CORAA-MUPE-ASR (nilc-nlp/CORAA-MUPE-ASR, 365 h, caixa/pontuação, locutores; uso não comercial confirmado pelo usuário). Spec 01, HFDatasetSource, normalize_transcript, streaming e artefatos por corpus implementados e testados.
- **Próxima Ação do Agente:** Rodar Steps 01-04 com coraa_mupe numa VM CPU High-RAM do Colab via src.colab_guard launch (job data_mupe) e acompanhar com job-status; depois ajustar o agendamento da Fase 1 (max_epochs/validação por passos) ao tamanho do MUPE e treinar em exp_coraa_mupe_warmup.
- **Comando Exato para o Próximo Agente:**
  ```bash
  python -m src.data_prep --ensure_colab_dataset --dataset coraa_mupe --val_stride 4 --local_dataset_dir /content/local_dataset --drive_root_dir /content/drive/MyDrive/parakeet_ptbr
  ```
- **Alertas Técnicos Críticos:**
  - NUNCA baixe datasets de áudio reais no Mac local; use sempre a VM do Colab (/content/local_dataset) + Google Drive.
  - No ambiente local (Mac), use sempre os binários `.venv/bin/python` e `.venv/bin/pytest`.
  - Mantenha `max_duration <= 20.0` segundos para evitar descarte ou OOM na Joint Network 4D na A100.
  - Nunca passe vírgula em string separada por vírgulas para `user_defined_symbols` do SentencePiece.
  - Fase 2 bloqueada por formato de dados: LhotseSpeechToTextSpkBpeDataset ignora spk_targets do manifesto (Spec 04 §8.2, DISC-019); o mixer precisa emitir cuts Lhotse com uma supervision por locutor.
  - Cada .ckpt/.nemo tem ~2,5 GB e arquivos rotacionados no Drive vão para a Lixeira (consomem cota): mantenha save_top_k=1 e validação espaçada (Spec 04 §4).
  - Lightning 2.x preserva o modo train/eval dos submódulos no fit: não chame model.eval() antes de trainer.fit.
  - FLEURS não expõe speaker_id: cada amostra recebe um locutor único (fleurs_pt_<idx>), o que enfraquece a garantia de locutores distintos do mixer.
