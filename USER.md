# USER.md — Painel de Status e Ações do Usuário

> Arquivo gerado e sincronizado automaticamente via `python -m src.state`. Sem prosa desnecessária.

## 1. Resumo Executivo

- **Projeto:** `parakeet-ptbr-multitalker` (`nvidia/multitalker-parakeet-streaming-0.6b-v1`)
- **Status Geral:** 🛑 Aguardando Ação do Usuário
- **Etapa Atual:** `step_06_colab_stage2_multitalker` — Treinamento Fase 2 (Multitalker Cache-Aware + Speaker Kernels) na GPU A100 (`colab_a100`)
- **Última Atualização (UTC):** `2026-10-25T12:54:53Z`

## 2. O Que o Usuário Precisa Fazer Agora (Impedimentos / Ações)

> [!IMPORTANT]
> Existem impedimentos ativos que exigem ação humana antes do próximo passo de GPU/Rede:

| ID | Etapa | Impedimento | Ação Necessária do Usuário |
|---|---|---|---|
| `blocker_colab_drive_auth` | `step_06_colab_stage2_multitalker` | Colab pediu interação humana (drive_auth) durante `bridge drive-mount`. | **Abra a URL de autorização do Google Drive no navegador, aprove o acesso para esta VM do Colab e responda 'ok' ao agente. URL: https://accounts.google.com/o/oauth2/v2/auth?access_type=offline&client_id=947318989803-rvmefmc9rok273go8589cmo21dkvjb2q.apps.googleusercontent.com&enable_granular_consent=true&login_hint=gmsgabrielmedeiros%40gmail.com&prompt=consent&redirect_uri=https%3A%2F%2Fcolab.research.google.com%2Ftun%2Fm%2Fauthorize-for-drive-credentials-ephem&response_type=none+gsession&scope=email+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fclient_channel+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdocs.test+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive.activity.readonly+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive.photos.readonly+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fexperimentsandconfigs+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fpeopleapi.readonly&state=%7B%22token%22%3A%22CgvjWKNRqDWULGSHOPwa8FNiGhk%3A1792932874121%22%2C%22endpoint%22%3A%22m-hm-kkb-euw4b1-2o2er9mzcanng%22%2C%22scopes%22%3A%5B%22email%22%2C%22https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fclient_channel%22%2C%22https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdocs.test%22%2C%22https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive%22%2C%22https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive.activity.readonly%22%2C%22https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive.photos.readonly%22%2C%22https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fexperimentsandconfigs%22%2C%22https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fpeopleapi.readonly%22%5D%7D** |

## 3. Progresso das Etapas (Pipeline Stateful)

| # | ID da Etapa | Ambiente | Status | Artefatos / Métricas |
|---|---|---|---|---|
| 0 | `step_00_audit_and_contracts` | `local_cpu` | ✅ Concluído | `contract_tests_passed=28`, specs: `specs/00..05_*.md`, schemas: `src/schemas.py`, state_file: `state/pipeline_state.json`, knowledge_base: `state/AGENT_KNOWLEDGE.md` |
| 1 | `step_01_data_prep_clean` | `colab_cpu` | ✅ Concluído | `dataset=coraa_mupe`, `clean_train_samples=251702`, `clean_val_samples=2242`, `val_stride=4`, clean_train_manifest: `/content/local_dataset/clean_train_manifest.json`, clean_val_manifest: `/content/local_dataset/clean_val_manifest.json` |
| 2 | `step_02_tokenizer_ptbr` | `colab_cpu` | ✅ Concluído | `dataset=coraa_mupe`, tokenizer_dir: `/content/drive/MyDrive/parakeet_ptbr/coraa_mupe/tokenizer_ptbr_1024` |
| 3 | `step_03_mixer_multitalker` | `colab_cpu` | ✅ Concluído | `train_mixtures=1000`, `train_lines=2857`, `val_mixtures=200`, `val_lines=400`, multitalker_train_manifest: `/content/local_dataset/multitalker_train_manifest.json`, multitalker_val_manifest: `/content/local_dataset/multitalker_val_manifest.json` |
| 4 | `step_04_pack_and_upload_drive` | `colab_cpu` | ✅ Concluído | `bundle_size_gb=26`, colab_bundle_tar_gz: `/content/drive/MyDrive/parakeet_ptbr/coraa_mupe/ptbr_dataset_16khz.tar.gz` |
| 5 | `step_05_colab_stage1_warmup` | `colab_a100` | ✅ Concluído | `val_wer_step_4000=0.7439`, `steps_per_sec=6.5`, `max_steps=24000`, `val_wer_step_8000=0.6720`, `val_wer_step_12000=0.6411`, `multitalker_train_lines=142866`, `multitalker_val_lines=1000`, `train_mixtures=50004`, `val_wer_final=0.6114`, warmup_checkpoint_dir: `/content/drive/MyDrive/parakeet_checkpoints/exp_coraa_mupe_warmup`, warmup_nemo: `/content/drive/MyDrive/parakeet_checkpoints/exp_coraa_mupe_warmup/parakeet_ptbr_warmup/checkpoints/parakeet_ptbr_warmup.nemo` |
| 6 | `step_06_colab_stage2_multitalker` | `colab_a100` | 🔄 Em Execução | `batches_per_sec_v1=2.93`, `gpu_util_v1=0.5`, `max_steps=53628`, `batches_per_sec_v2=3.47`, `val_wer_epoch_0=0.5786`, `batches_per_sec_fused=4.0`, `val_wer_step_8932=0.5258`, `samples_per_sec_b32=117`, `val_wer_step_13400=0.5022`, `val_wer_step_22336=0.4716`, `val_wer_step_26804=0.4614`, `val_wer_step_35740=0.4456`, `val_wer_step_44676=0.4347` |
| 7 | `step_07_eval_and_export` | `colab_a100` | ⏳ Pendente | — |

## 4. Handover para o Próximo Agente de IA

- **Resumo da Última Sessão:** Fase 2: val_wer 0,5786 -> 0,5258 -> 0,5022 -> 0,4716 -> 0,4614 -> 0,4456 -> 0,4347 (passo 44.676, ~09:05 UTC). A VM A100 foi recolhida entre 10:37 e 12:53 UTC; falta confirmar no Drive se o treino chegou a 53.628 passos e salvou o .nemo.
- **Próxima Ação do Agente:** Usuário autoriza o Drive na VM CPU nova (blocker_colab_drive_auth). Depois: inspecionar /content/drive/MyDrive/parakeet_checkpoints/exp_coraa_mupe_multitalker/parakeet_ptbr_multitalker/ (checkpoints/*.nemo, *-last.ckpt, nemo_log_globalrank-0.txt e state/ espelhado). Se o .nemo final existir e o log mostrar max_steps=53628 atingido: concluir a step_06 e rodar a step_07 numa VM com GPU (extrair o pacote, src.evaluator). Se não: relançar a Fase 2 numa A100 a partir do -last.ckpt com --max_steps 53628.
- **Comando Exato para o Próximo Agente:**
  ```bash
  .venv/bin/python -m src.colab_guard drive
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
