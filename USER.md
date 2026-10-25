# USER.md — Painel de Status e Ações do Usuário

> Arquivo gerado e sincronizado automaticamente via `python -m src.state`. Sem prosa desnecessária.

## 1. Resumo Executivo

- **Projeto:** `parakeet-ptbr-multitalker` (`nvidia/multitalker-parakeet-streaming-0.6b-v1`)
- **Status Geral:** 🔄 Em Execução por Agente
- **Etapa Atual:** `step_06_colab_stage2_multitalker` — Treinamento Fase 2 (Multitalker Cache-Aware + Speaker Kernels) na GPU A100 (`colab_a100`)
- **Última Atualização (UTC):** `2026-10-25T04:08:17Z`

## 2. O Que o Usuário Precisa Fazer Agora (Impedimentos / Ações)

- **Nenhum impedimento bloqueante no momento.** Os agentes de IA podem prosseguir com a etapa atual.
- **Pré-requisitos futuros (Etapas 04 a 07):** Ter conta Google Colab Pro/Pro+ com GPU **NVIDIA A100** e espaço no Google Drive (`/content/drive/MyDrive/parakeet_ptbr`).

## 3. Progresso das Etapas (Pipeline Stateful)

| # | ID da Etapa | Ambiente | Status | Artefatos / Métricas |
|---|---|---|---|---|
| 0 | `step_00_audit_and_contracts` | `local_cpu` | ✅ Concluído | `contract_tests_passed=28`, specs: `specs/00..05_*.md`, schemas: `src/schemas.py`, state_file: `state/pipeline_state.json`, knowledge_base: `state/AGENT_KNOWLEDGE.md` |
| 1 | `step_01_data_prep_clean` | `colab_cpu` | ✅ Concluído | `dataset=coraa_mupe`, `clean_train_samples=251702`, `clean_val_samples=2242`, `val_stride=4`, clean_train_manifest: `/content/local_dataset/clean_train_manifest.json`, clean_val_manifest: `/content/local_dataset/clean_val_manifest.json` |
| 2 | `step_02_tokenizer_ptbr` | `colab_cpu` | ✅ Concluído | `dataset=coraa_mupe`, tokenizer_dir: `/content/drive/MyDrive/parakeet_ptbr/coraa_mupe/tokenizer_ptbr_1024` |
| 3 | `step_03_mixer_multitalker` | `colab_cpu` | ✅ Concluído | `train_mixtures=1000`, `train_lines=2857`, `val_mixtures=200`, `val_lines=400`, multitalker_train_manifest: `/content/local_dataset/multitalker_train_manifest.json`, multitalker_val_manifest: `/content/local_dataset/multitalker_val_manifest.json` |
| 4 | `step_04_pack_and_upload_drive` | `colab_cpu` | ✅ Concluído | `bundle_size_gb=26`, colab_bundle_tar_gz: `/content/drive/MyDrive/parakeet_ptbr/coraa_mupe/ptbr_dataset_16khz.tar.gz` |
| 5 | `step_05_colab_stage1_warmup` | `colab_a100` | ✅ Concluído | `val_wer_step_4000=0.7439`, `steps_per_sec=6.5`, `max_steps=24000`, `val_wer_step_8000=0.6720`, `val_wer_step_12000=0.6411`, `multitalker_train_lines=142866`, `multitalker_val_lines=1000`, `train_mixtures=50004`, `val_wer_final=0.6114`, warmup_checkpoint_dir: `/content/drive/MyDrive/parakeet_checkpoints/exp_coraa_mupe_warmup`, warmup_nemo: `/content/drive/MyDrive/parakeet_checkpoints/exp_coraa_mupe_warmup/parakeet_ptbr_warmup/checkpoints/parakeet_ptbr_warmup.nemo` |
| 6 | `step_06_colab_stage2_multitalker` | `colab_a100` | 🔄 Em Execução | `batches_per_sec_v1=2.93`, `gpu_util_v1=0.5`, `max_steps=53628`, `batches_per_sec_v2=3.47` |
| 7 | `step_07_eval_and_export` | `colab_a100` | ⏳ Pendente | — |

## 4. Handover para o Próximo Agente de IA

- **Resumo da Última Sessão:** Fase 2 em andamento na A100 (142.866 cuts, max_steps=53.628). Otimizações de vazão: guard de NaN com 1 sync (+15%: 3,03 -> 3,47 lotes/s); AdamW fused pendente de push (Lightning exige recorte manual).
- **Próxima Ação do Agente:** Job multitalker_mupe rodando desde ~04:05 UTC com /content/a100_multitalker_nofused.yaml (sem fused, guard de NaN ativo), ~3,47 lotes/s (~8,6 h). Após o push do guard em on_before_optimizer_step + split_fused_gradient_clipping, esperar o checkpoint do fim da época 0 (~04:50 UTC), dar job-kill e relançar sem --config (YAML do repo, fused) para retomar do -last.ckpt (resume_if_exists).
- **Comando Exato para o Próximo Agente:**
  ```bash
  .venv/bin/python -m src.colab_guard job-status --job multitalker_mupe --job-dir /content/jobs --tail 30
  ```
- **Alertas Técnicos Críticos:**
  - NUNCA baixe datasets de áudio reais no Mac local; use sempre a VM do Colab (/content/local_dataset) + Google Drive.
  - No ambiente local (Mac), use sempre os binários `.venv/bin/python` e `.venv/bin/pytest`.
  - Mantenha `max_duration <= 20.0` segundos para evitar descarte ou OOM na Joint Network 4D na A100.
  - Nunca passe vírgula em string separada por vírgulas para `user_defined_symbols` do SentencePiece.
  - Fase 2 lê CutSet Lhotse (cuts_path) derivado do manifesto espelhado por configure_multitalker_cuts() (Spec 02 §3.7, DISC-026); cuts escritos à mão precisam de custom={} para o speaker_to_target() do NeMo.
  - Lightning 2.x preserva o modo train/eval dos submódulos no fit: não chame model.eval() antes de trainer.fit.
  - FLEURS não expõe speaker_id: cada amostra recebe um locutor único (fleurs_pt_<idx>), o que enfraquece a garantia de locutores distintos do mixer.
  - Cada .ckpt/.nemo tem ~2,5 GB: mantenha save_top_k=1 e validação espaçada por custo de I/O (Spec 04 §4); o Drive tem 5 TB (DISC-025), cota não é gargalo.
