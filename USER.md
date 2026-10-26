# USER.md — Painel de Status e Ações do Usuário

> Arquivo gerado e sincronizado automaticamente via `python -m src.state`. Sem prosa desnecessária.

## 1. Resumo Executivo

- **Projeto:** `parakeet-ptbr-multitalker` (`nvidia/multitalker-parakeet-streaming-0.6b-v1`)
- **Status Geral:** ✅ Pipeline Finalizado
- **Etapa Atual:** `step_07_eval_and_export` — Avaliação Final (WER Limpo e Sobreposto) e Exportação do Checkpoint .nemo (`colab_a100`)
- **Última Atualização (UTC):** `2026-10-26T17:50:22Z`

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
| 6 | `step_06_colab_stage2_multitalker` | `colab_a100` | ✅ Concluído | `batches_per_sec_v1=2.93`, `gpu_util_v1=0.5`, `max_steps=53628`, `batches_per_sec_v2=3.47`, `val_wer_epoch_0=0.5786`, `batches_per_sec_fused=4.0`, `val_wer_step_8932=0.5258`, `samples_per_sec_b32=117`, `val_wer_step_13400=0.5022`, `val_wer_step_22336=0.4716`, `val_wer_step_26804=0.4614`, `val_wer_step_35740=0.4456`, `val_wer_step_44676=0.4347`, `val_wer_best=0.4347`, `val_wer_epoch_11=0.4417`, multitalker_checkpoint_dir: `/content/drive/MyDrive/parakeet_checkpoints/exp_coraa_mupe_multitalker`, multitalker_nemo: `/content/drive/MyDrive/parakeet_checkpoints/exp_coraa_mupe_multitalker/parakeet_ptbr_multitalker/checkpoints/parakeet_ptbr_multitalker.nemo`, multitalker_best_ckpt: `/content/drive/MyDrive/parakeet_checkpoints/exp_coraa_mupe_multitalker/parakeet_ptbr_multitalker/checkpoints/parakeet_ptbr_multitalker--val_wer=0.4347-epoch=7.ckpt`, multitalker_last_ckpt: `/content/drive/MyDrive/parakeet_checkpoints/exp_coraa_mupe_multitalker/parakeet_ptbr_multitalker/checkpoints/parakeet_ptbr_multitalker--val_wer=0.4417-epoch=11-last.ckpt` |
| 7 | `step_07_eval_and_export` | `colab_a100` | ✅ Concluído | `clean_wer=0.3622`, `clean_wer_normalized=0.2997`, `multitalker_wer=0.4646`, `multitalker_wer_normalized=0.4069`, `clean_samples=2242`, `multitalker_samples=1000`, final_nemo_model: `/content/drive/MyDrive/parakeet_checkpoints/exp_coraa_mupe_multitalker/parakeet_ptbr_multitalker/checkpoints/parakeet_ptbr_multitalker.nemo`, eval_report: `/content/drive/MyDrive/parakeet_checkpoints/exp_coraa_mupe_multitalker/eval_report.json` |

## 4. Handover para o Próximo Agente de IA

- **Resumo da Última Sessão:** Step 07 concluída em 26/10 17:49 UTC na A100: clean_wer=0,3622 (0,2997 normalizado, 2.242 amostras) e multitalker_wer=0,4646 (0,4069 normalizado, 1.000 locutores-alvo em 500 misturas). Modelo final = .nemo da Fase 2 (melhor val_wer 0,4347, época 7). Avaliado nos manifestos de validação, os mesmos usados para escolher o checkpoint: números levemente otimistas.
- **Próxima Ação do Agente:** Pipeline finalizado. Próximos passos opcionais: avaliar num conjunto de teste separado (o pacote coraa_mupe só tem train/val; o MUPE tem split test no HF) e, para melhorar o WER, ampliar o corpus ou treinar mais épocas.
- **Comando Exato para o Próximo Agente:**
  ```bash
  .venv/bin/python -m src.state status
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
