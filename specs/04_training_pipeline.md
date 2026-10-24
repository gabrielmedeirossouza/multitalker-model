# Spec 04: Pipeline de Treinamento em 2 Estágios na GPU NVIDIA A100 (Google Colab)

## 1. Objetivo
Especificar os hiperparâmetros, fases de treinamento, otimizações de hardware para a GPU **NVIDIA A100 (40GB/80GB)** e a política de tolerância a falhas (*checkpointing* contínuo no Google Drive) para adaptar o `nvidia/multitalker-parakeet-streaming-0.6b-v1` ao Português Brasileiro.

---

## 2. Estratégia em 2 Estágios (Prevenção de Esquecimento Catastrófico)

Como a troca de vocabulário (`Spec 03`) reinicializa o Decoder RNN-T e a Joint Network com pesos aleatórios, treinar o modelo inteiro desde o passo 0 geraria gradientes ruidosos que destruiriam os **Speaker Kernels** e as representações acústicas do **FastConformer**. Por isso, o treinamento é obrigatoriamente dividido em duas fases:

| Atributo | Fase 1: `warmup` (Alinhamento Textual PT-BR) | Fase 2: `multitalker` (Ajuste Fino Cache-Aware + Sobreposição) |
|---|---|---|
| **Estado do FastConformer** | **Congelado** (`asr_model.encoder.freeze()`) | **Descongelado** (`asr_model.encoder.unfreeze()`) |
| **Módulos Treinados** | Apenas Decoder RNN-T + Joint Network (~5% dos parâmetros) | Modelo completo: FastConformer + Speaker Kernels + RNN-T (600M) |
| **Composição de Dados** | 100% Fala Limpa (Single-Speaker PT-BR, `spk_targets=1`, `bg_spk_targets=0`) | **70% Misturas Sobrepostas** (`Spec 02`) + **30% Fala Limpa** |
| **Contextos de Streaming** | Contexto padrão do checkpoint | Multi-latência: `att_context_size: [[70, 0], [70, 1], [70, 6], [70, 13]]` |
| **Learning Rate (`AdamW`)** | `1e-3` (Cosine Annealing, `warmup_steps=1000`) | `1e-4` a `5e-5` (Cosine Annealing, `warmup_steps=2000`) |
| **Épocas Recomendadas** | 3 a 5 épocas (~3–4 horas na A100) | 10 a 12 épocas (~15–25 horas na A100) |
| **Throughput na A100** | ~350–450h de áudio / hora de GPU | ~160–220h de áudio / hora de GPU |

---

## 3. As 3 Otimizações Obrigatórias para GPU NVIDIA A100

Todo arquivo de configuração em `configs/` e o módulo `src/trainer.py` **DEVEM** aplicar estas três otimizações:

1. **Precisão Mista Nativa (`precision: "bf16-mixed"`) + `warprnnt_numba` (`fused_batch_size`):**
   - A GPU A100 possui Tensor Cores nativos para `bfloat16`, acelerando o treinamento em ~2,2x sem risco de *underflow* de gradiente.
   - O cálculo da perda RNN-T deve usar `loss_name: "warprnnt_numba"` com `fused_batch_size: 16` para evitar explosão de VRAM na Joint Network.
2. **Agrupamento por Duração (`bucketing_strategy: "fully_randomized"`):**
   - Agrupa áudios de comprimentos semelhantes no mesmo mini-batch, reduzindo o desperdício com *padding* de zeros e cortando o tempo total de treinamento em **35% a 40%**.
3. **Vocabulário Enxuto (`1024` tokens):**
   - Mantém a última dimensão do tensor 4D `(Batch, Time, Token, Vocab)` em `1025`, viabilizando `batch_size` de `16` a `32` por GPU A100.

---

## 4. Contrato de Resiliência a Desconexões no Google Colab

Como as sessões do Google Colab expiram após 12h (Pro) ou 24h (Pro+), o `exp_manager` do NeMo deve ser configurado com:
- `exp_dir`: apontando para `/content/drive/MyDrive/parakeet_checkpoints/`
- `resume_if_exists: true`
- `resume_ignore_no_checkpoint: true`
- `create_checkpoint_callback: true`
- `checkpoint_callback_params`:
  - `monitor: "val_wer"`
  - `mode: "min"`
  - `save_top_k: 3`
  - `always_save_nemo: true`
  - `every_n_train_steps: 1000` (garante perda máxima de ~15–20 minutos caso a VM do Colab seja desconectada abruptamente).

---

## 5. Aplicação Efetiva das Configurações no Modelo NeMo (`src/trainer.py`)

Ao carregar o modelo via `from_pretrained` (Fase 1) ou `restore_from` (Fase 2), o `src/trainer.py` **DEVE**:
1. Validar o manifesto completo (ou amostragem configurável) e verificar a existência física dos arquivos de áudio referenciados antes de alocar a GPU.
2. Injetar explicitamente no `asr_model.cfg` as configurações de `joint` (`fuse_loss_wer`, `fused_batch_size`), `loss` (`warprnnt_numba`) e, na Fase 2, `encoder.att_context_size` (`[[70, 0], [70, 1], [70, 6], [70, 13]]`).
3. Permitir sobrescrever `init_from_nemo_model` via argumento CLI `--init_model` na Fase 2.

---

## 6. Avaliação Final e Exportação (`step_07_eval_and_export` / `src/evaluator.py`)

Após a conclusão da Fase 2 (`multitalker`), o módulo `src/evaluator.py` executa a validação final do modelo adaptado:
1. Calcula o **WER (Word Error Rate)** em fala limpa (`clean_val_manifest.json` / `clean_test_manifest.json`) e o **Multitalker WER** em fala sobreposta (`multitalker_val_manifest.json` / `multitalker_test_manifest.json`) condicionada pelas máscaras `spk_targets` e `bg_spk_targets`.
2. Valida o relatório de métricas via contrato `EvaluationReport` (`src/schemas.py`), exigindo `clean_wer` e `multitalker_wer` no intervalo `[0.0, 1.0]`, contagens de amostras avaliadas `> 0` e verificação do artefato `.nemo` exportado.
3. Atualiza `step_07_eval_and_export` para `completed` em `state/pipeline_state.json` e `USER.md`.

---

## 7. Critérios de Aceite

- `src/schemas.py` (`TrainingStageConfig`, `EvaluationReport`) e `tests/` devem validar que:
  1. Na fase `warmup`, `freeze_encoder` é obrigatoriamente `True`.
  2. Na fase `multitalker`, `freeze_encoder` é obrigatoriamente `False`, `return_speaker_targets` é `True`, `speaker_kernel_enabled` é `True` e `att_context_size` contém os 4 contextos de latência `[[70, 0], [70, 1], [70, 6], [70, 13]]`.
  3. `precision` é `"bf16-mixed"` (ou `"16-mixed"`).
  4. `bucketing_strategy` está ativo (`"fully_randomized"` ou `"synced_randomized"`).
  5. `resume_if_exists` é `True`.
  6. `src/evaluator.py` calcula WER exato por distância de edição em nível de palavras e gera um `EvaluationReport` válido.
