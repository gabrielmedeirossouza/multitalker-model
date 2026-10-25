# USER.md — Painel de Status e Ações do Usuário

> Arquivo gerado e sincronizado automaticamente via `python -m src.state`. Sem prosa desnecessária.

## 1. Resumo Executivo

- **Projeto:** `parakeet-ptbr-multitalker` (`nvidia/multitalker-parakeet-streaming-0.6b-v1`)
- **Status Geral:** 🔄 Em Execução por Agente
- **Etapa Atual:** `step_05_colab_stage1_warmup` — Treinamento Fase 1 (Warm-up: Encoder Congelado, Decoder/Joint PT-BR) na GPU A100 (`colab_a100`)
- **Última Atualização (UTC):** `2026-10-25T01:46:37Z`

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
| 5 | `step_05_colab_stage1_warmup` | `colab_a100` | 🔄 Em Execução | `val_wer_step_4000=0.7439`, `steps_per_sec=6.6`, `max_steps=24000` |
| 6 | `step_06_colab_stage2_multitalker` | `colab_a100` | ⏳ Pendente | — |
| 7 | `step_07_eval_and_export` | `colab_a100` | ⏳ Pendente | — |

## 4. Handover para o Próximo Agente de IA

- **Resumo da Última Sessão:** Warmup CORAA-MUPE na A100 (val_wer=0,7439 no passo 4.000). Encontrado e corrigido localmente o bug de colisão de nomes das misturas train/val e o desvio 70/30 do CutSet; misturas do pacote atual precisam ser regeneradas antes da Fase 2.
- **Próxima Ação do Agente:** Após o push do commit com a correção (Spec 02 §3.6/§3.7), lançar na A100 o job mix_mupe a partir de um clone separado (/content/mt-prep, para não mexer no repo do warmup em execução): apagar wavs/mix_* e multitalker_*_manifest.json antigos, gerar 50.000 misturas de treino (clean_ratio 0.30) e 500 de validação, e reempacotar /content/local_dataset em .tar.gz.tmp no Drive (pack_dataset_for_colab_ssd, sem reescrever manifestos lidos pelo warmup) antes de renomear. Ao fim do warmup (~02:40 UTC), iniciar a step_06.
- **Comando Exato para o Próximo Agente:**
  ```bash
  .venv/bin/python -m src.colab_guard job-status --job warmup_mupe --job-dir /content/drive/MyDrive/parakeet_ptbr/jobs --tail 40
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
