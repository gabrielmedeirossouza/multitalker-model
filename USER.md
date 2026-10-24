# USER.md — Painel de Status e Ações do Usuário

> Arquivo gerado e sincronizado automaticamente via `python -m src.state`. Sem prosa desnecessária.

## 1. Resumo Executivo

- **Projeto:** `parakeet-ptbr-multitalker` (`nvidia/multitalker-parakeet-streaming-0.6b-v1`)
- **Status Geral:** 🛑 Aguardando Ação do Usuário
- **Etapa Atual:** `step_05_colab_stage1_warmup` — Treinamento Fase 1 (Warm-up: Encoder Congelado, Decoder/Joint PT-BR) na GPU A100 (`colab_a100`)
- **Última Atualização (UTC):** `2026-10-24T21:21:08Z`

## 2. O Que o Usuário Precisa Fazer Agora (Impedimentos / Ações)

> [!IMPORTANT]
> Existem impedimentos ativos que exigem ação humana antes do próximo passo de GPU/Rede:

| ID | Etapa | Impedimento | Ação Necessária do Usuário |
|---|---|---|---|
| `blocker_step05_corpus_scale` | `step_05_colab_stage1_warmup` | O warmup com FLEURS pt_br (8,9 h, 1.380 frases únicas) não aprende ancoragem acústica em nenhuma variante testada (val_wer >= 0,993; Spec 04 §8.5). Pipeline NeMo 3.0 corrigido e verificado em inglês (val_wer 0,21). | **Decidir a ampliação do corpus PT-BR (mls_pt ~160 h aberto; coraa ~290 h; common_voice_pt exige token HF com termos aceitos) e se a Fase 1 pode usar encoder descongelado. Esvaziar a Lixeira do Drive e, se quiser, apagar parakeet_checkpoints/exp_v*_lmshortcut (12,7 GB sem valor).** |

## 3. Progresso das Etapas (Pipeline Stateful)

| # | ID da Etapa | Ambiente | Status | Artefatos / Métricas |
|---|---|---|---|---|
| 0 | `step_00_audit_and_contracts` | `local_cpu` | ✅ Concluído | `contract_tests_passed=28`, specs: `specs/00..05_*.md`, schemas: `src/schemas.py`, state_file: `state/pipeline_state.json`, knowledge_base: `state/AGENT_KNOWLEDGE.md` |
| 1 | `step_01_data_prep_clean` | `colab_cpu` | ✅ Concluído | clean_train_manifest: `/content/local_dataset/clean_train_manifest.json`, clean_val_manifest: `/content/local_dataset/clean_val_manifest.json` |
| 2 | `step_02_tokenizer_ptbr` | `colab_cpu` | ✅ Concluído | tokenizer_dir: `/content/drive/MyDrive/parakeet_ptbr/tokenizer_ptbr_1024` |
| 3 | `step_03_mixer_multitalker` | `colab_cpu` | ✅ Concluído | multitalker_train_manifest: `/content/local_dataset/multitalker_train_manifest.json`, multitalker_val_manifest: `/content/local_dataset/multitalker_val_manifest.json` |
| 4 | `step_04_pack_and_upload_drive` | `colab_cpu` | ✅ Concluído | colab_bundle_tar_gz: `/content/drive/MyDrive/parakeet_ptbr/ptbr_dataset_16khz.tar.gz` |
| 5 | `step_05_colab_stage1_warmup` | `colab_a100` | 🛑 Bloqueado (Ação do Usuário) | — |
| 6 | `step_06_colab_stage2_multitalker` | `colab_a100` | ⏳ Pendente | — |
| 7 | `step_07_eval_and_export` | `colab_a100` | ⏳ Pendente | — |

## 4. Handover para o Próximo Agente de IA

- **Resumo da Última Sessão:** Pipeline NeMo 3.0 da Fase 1 corrigido e validado na A100 (Lhotse, logger, max_steps, Speaker Kernels congelados, RNNTLoss, setters do RNNTJoint). Warmup com FLEURS 8,9 h falhou por atalho de LM em 5 variantes (val_wer >= 0,993); pipeline verificado em inglês (val_wer 0,21). Aguardando decisão do usuário sobre ampliar o corpus.
- **Próxima Ação do Agente:** Após resolver blocker_step05_corpus_scale: preparar o corpus ampliado (Steps 01-04 com o novo dataset) e reexecutar a Fase 1 em checkpoint_dir limpo; acompanhar val_wer a cada validação e abortar se ficar >= 0,99 após ~1.500 passos.
- **Comando Exato para o Próximo Agente:**
  ```bash
  python -m src.trainer --stage warmup --manifest /content/local_dataset/clean_train_manifest.json --val_manifest /content/local_dataset/clean_val_manifest.json --checkpoint_dir /content/drive/MyDrive/parakeet_checkpoints/exp_v1_warmup --tokenizer_dir /content/drive/MyDrive/parakeet_ptbr/tokenizer_ptbr_1024 --update_state
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
