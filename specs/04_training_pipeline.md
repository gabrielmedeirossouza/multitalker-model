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
| **Learning Rate (`AdamW`)** | `1e-3` (Cosine Annealing, `warmup_ratio=0.1` — ver §8.4) | `1e-4` a `5e-5` (Cosine Annealing, `warmup_ratio=0.1`) |
| **Épocas Recomendadas** | 3 a 5 épocas para corpora de centenas de horas; para corpora pequenos (< 20 h, ex.: FLEURS ~8 h) usar 30 épocas (ver §8.4) | 10 a 12 épocas (~15–25 horas na A100) |
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
  - `save_top_k: 1`
  - `always_save_nemo: false` + `save_best_model: true` (+ `save_nemo_on_train_end`, padrão `true`): **um único** `.nemo` gravado no fim do treino, com os pesos de menor `val_wer`, usado como `init_from_nemo_model` da Fase 2.
  - **Orçamento de I/O no Drive:** cada `.ckpt`/`.nemo` do modelo de 0,6B tem **~2,5 GB** (pesos fp32 completos, mesmo com encoder congelado). Arquivos removidos pela rotação de top-k/`-last` no Drive montado vão para a **Lixeira do Drive e continuam consumindo cota**. Por isso: `save_top_k: 1`, sem `.nemo` a cada checkpoint e validação espaçada (`check_val_every_n_epoch`). Após cada fase, o usuário deve esvaziar a Lixeira do Drive.
  - **Gatilho de checkpoint:** **sem** `every_n_train_steps` → o `NeMoModelCheckpoint` salva top-k + `-last` ao fim de **cada validação**, sempre com `val_wer` atualizado.
    - Com `every_n_train_steps` definido, o Lightning **não** salva top-k ao fim da validação (e o salvamento por passos ocorre antes da validação do mesmo passo, usando `val_wer` defasado); `every_n_train_steps` e `every_n_epochs` não podem coexistir.
    - A frequência de salvamento é controlada por `trainer.val_check_interval`: `1.0` (fim de época) para corpora pequenos; para épocas > 20 min, usar um inteiro de lotes (ex.: `1000`) com `check_val_every_n_epoch: null`, garantindo perda máxima de ~15–20 minutos caso a VM do Colab seja desconectada abruptamente.
    - Como o dataloader Lhotse não tem `len()`, o Lightning rejeita `val_check_interval` fracionário (`< 1.0`).

---

## 5. Aplicação Efetiva das Configurações no Modelo NeMo (`src/trainer.py`)

Ao carregar o modelo via `from_pretrained` (Fase 1) ou `restore_from` (Fase 2), o `src/trainer.py` **DEVE**:
1. Validar o manifesto completo (ou amostragem configurável) e verificar a existência física dos arquivos de áudio referenciados antes de alocar a GPU.
2. Injetar explicitamente no `asr_model.cfg` as configurações de `joint` (`fuse_loss_wer`, `fused_batch_size`), `loss` (`warprnnt_numba`) e, na Fase 2, `encoder.att_context_size` (`[[70, 0], [70, 1], [70, 6], [70, 13]]`).
   - No objeto `RNNTJoint` do NeMo 3.0, `fuse_loss_wer` e `fused_batch_size` são **propriedades somente leitura**: usar `set_fused_batch_size()` e, apenas quando o valor mudar, `set_fuse_loss_wer(valor, loss=asr_model.loss, metric=asr_model.wer)` — esse método sobrescreve a perda/WER fundidos (passar `None` os desligaria).
3. **Reconstruir o objeto `asr_model.loss` (`RNNTLoss`) a partir de `asr_model.cfg.loss` e `cfg.rnnt_reduction`** depois de `change_vocabulary()`, pois no NeMo 3.0 `change_vocabulary()` recria a perda com `RNNTLoss(num_classes=...)` padrão, descartando `fastemit_lambda` e a redução configurada. Com `fuse_loss_wer`, a nova perda deve ser religada via `joint.set_loss()`.
4. Permitir sobrescrever `init_from_nemo_model` via argumento CLI `--init_model` na Fase 2.
5. Na Fase 1, congelar o FastConformer (`encoder.freeze()`) **e** os Speaker Kernels (`spk_kernels` e `bg_spk_kernels`), que no NeMo 3.0 são módulos de topo do modelo (fora de `encoder`) acoplados por *hooks* — `encoder.freeze()` sozinho os deixaria treináveis.
6. Na Fase 1, **preservar os pesos pré-treinados do Joint/Decoder que independem do vocabulário** (`reuse_vocab_independent_weights: true`): `change_vocabulary()` reinicializa aleatoriamente *todo* o Joint e o Decoder, inclusive a projeção acústica `joint.enc` (1024→640). O `src/trainer.py` copia, antes da troca, e restaura, depois dela, apenas a *allowlist* `joint.enc.*`, `joint.pred.*` e `decoder.prediction.dec_rnn.*` (LSTM). `decoder.prediction.embed.*` e `joint.joint_net.*` (dependentes dos IDs de token) permanecem reinicializados — atenção: o vocabulário original também tem 1024 tokens, então esses tensores têm **o mesmo shape** e não podem ser copiados por casamento de shape (ver §8.5).

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
  7. `train_ds.use_lhotse` e `validation_ds.use_lhotse` são `True` e `trainer.logger` é `False` em ambas as fases (§8.2, §8.3).
  8. `checkpoint_callback_params` não define `every_n_train_steps` e `every_n_epochs` simultaneamente (§4).
  9. `compute_max_steps()` aplica a fórmula da §8.4 e o `src/trainer.py` injeta `trainer.max_steps > 0` quando o YAML não o define.
  10. Na Fase 1, `freeze_speaker_kernels()` desabilita gradientes de `spk_kernels` e `bg_spk_kernels` (§5.5).
  11. `snapshot_vocab_independent_weights()` / `restore_vocab_independent_weights()` copiam somente a *allowlist* `joint.enc.*`, `joint.pred.*` e `decoder.prediction.dec_rnn.*`, nunca `embed`/`joint_net`, mesmo com shapes iguais (§5.6).

---

## 8. Integração Real com o NVIDIA NeMo 3.0 (`EncDecMultiTalkerRNNTBPEModel`)

Verificado na VM A100 com `nemo_toolkit==3.0.0` (PyPI) e o checkpoint `multitalker-parakeet-streaming-0.6b-v1.nemo`.

### 8.1 Classe do Modelo e Speaker Kernels
- `target`: `nemo.collections.asr.models.multitalker_asr_models.EncDecMultiTalkerRNNTBPEModel` (herda `EncDecRNNTBPEModel` + `SpeakerKernelMixin`), disponível no `nemo_toolkit>=3.0.0` do PyPI.
- Encoder: FastConformer de 24 camadas, `d_model=1024`, `feat_in=128`, `subsampling_factor=8`, `att_context_size` multi-latência já presente no checkpoint (`[[70, 13], [70, 6], [70, 1], [70, 0]]`).
- Speaker Kernels: `spk_kernels` / `bg_spk_kernels` (`nn.ModuleDict` de FF com `dropout=0.5`) injetados por *forward hooks* nas camadas de `spk_kernel_layers`.

### 8.2 Dataloader Lhotse Obrigatório
- `EncDecMultiTalkerRNNTBPEModel._setup_dataloader_from_config` **só** cria dataloader quando `use_lhotse: true` (caso contrário retorna `None` silenciosamente). Logo `train_ds.use_lhotse` e `validation_ds.use_lhotse` **DEVEM** ser `true`.
- O dataset `LhotseSpeechToTextSpkBpeDataset` **ignora** os campos `spk_targets`/`bg_spk_targets` do manifesto: as máscaras são derivadas das *supervisions* do `Cut` Lhotse (`speaker_to_target`), e o locutor-alvo é sorteado por amostra.
  - Manifesto NeMo JSON (não-tarred) → `MonoCut` com **1 supervision** cobrindo toda a duração → `spk_targets` = 1 e `bg_spk_targets` = 0. **Compatível com a Fase 1 (fala limpa).**
  - **Pendência da Fase 2 (`step_06`):** misturas sobrepostas exigem *cuts* Lhotse com uma supervision por locutor (`speaker`, `start`, `duration`, `text`) ou `rttm_filepath` por cut; o manifesto espelhado atual (Spec 02) seria lido como fala única do locutor-alvo em toda a duração.
- Campos do `train_ds`/`validation_ds` não reconhecidos pelo `LhotseDataLoadingConfig` são descartados com aviso (ex.: `bucketing_strategy`, `return_speaker_targets`); o dataset ainda lê `num_speakers`, `num_sample_per_mel_frame` (160) e `num_mel_frame_per_asr_frame` (8) do mesmo config.
- `text_field: "text"` (campo do nosso manifesto; o checkpoint original usa `answer`).
- Manifesto NeMo **não-tarred** não é repetido (`.repeat()` só ocorre em dados tarred/`input_cfg`), então épocas terminam normalmente.

### 8.3 Trainer Lightning + `exp_manager`
- `trainer.logger: false` — o `exp_manager` cria o TensorBoard e lança `LoggerMisconfigurationError` se o `Trainer` já tiver logger.
- `trainer.enable_checkpointing: false` — o `exp_manager` cria o `NeMoModelCheckpoint`.
- `trainer.use_distributed_sampler: false` — a amostragem é feita pelo sampler Lhotse.

### 8.4 `max_steps` Explícito e Agendamento de LR
- Com Lhotse, `ModelPT.setup_optimization` não consegue inferir `max_steps` a partir de `max_epochs` (chama `len(train_dataloader.dataset)`, inexistente em `LhotseSpeechToTextSpkBpeDataset`). Portanto `trainer.max_steps` **DEVE** ser `> 0`.
- Se o YAML não definir `trainer.max_steps` (ou definir `-1`), o `src/trainer.py` o calcula a partir do manifesto validado:
  $$\text{max\_steps} = \left\lceil \frac{\text{max\_epochs} \cdot \left(\lceil N_{\text{train}} / \text{batch\_size} \rceil + \text{num\_buckets}\right)}{\text{accumulate\_grad\_batches}} \right\rceil$$
  - `num_buckets` (0 sem bucketing) é a margem de lotes parciais: cada bucket Lhotse fecha a época com um lote incompleto (medido na A100: 84 lotes/época para FLEURS contra 82 de $\lceil 2596/32 \rceil$).
  - O limite **precisa** ser um teto: se `max_steps` cortar a última época, a validação final não roda e `save_best_model` restaura um checkpoint anterior, descartando as últimas épocas. Assim o treino termina por `max_epochs`, e `max_steps` só define a curva do scheduler (LR final ≈ `min_lr`).
  - O CLI aceita `--max_steps` para sobrescrever (ex.: *smoke test*).
- Fase 1 usa `warmup_ratio: 0.1` (proporcional a `max_steps`) em vez de `warmup_steps` fixo: com FLEURS (~2,6 mil amostras, `batch_size=32`) 5 épocas dariam ~410 passos, menos que `warmup_steps=1000`, e o LR nunca atingiria o pico.
- Fase 1 com corpus pequeno: `max_epochs: 30` (~2,5 mil passos, ~4 it/s na A100 ≈ 11 min de treino) e `check_val_every_n_epoch: 5` (6 validações/checkpoints, limitando a escrita no Drive). Decoder/Joint são reinicializados aleatoriamente pela troca de vocabulário e precisam de milhares de passos; o encoder congelado torna cada passo barato e o top-k por `val_wer` com `save_best_model` protege contra *overfitting*.

### 8.5 Diagnóstico do Warmup v1 (Atalho de Modelo de Linguagem)
Primeira execução real da Fase 1 (FLEURS pt_br: 8,9 h, 2.596 amostras, **1.380 frases únicas** ≈ 1,9 gravações/frase; frases de validação disjuntas), com Joint/Decoder totalmente reinicializados:
- `train_loss` 24,4 → 0,74 em 30 épocas, mas `training_batch_wer` ≈ 0,996 e melhor `val_wer` = 0,9931 (hipóteses de 1 token: "A", "O").
- Alinhamento de Viterbi do caminho de referência: dezenas de tokens emitidos **num único frame**; `nll/token` 0,27–0,36 no treino contra **11–17** na validação → o Decoder memorizou as transcrições via rede de predição (LM) e o Joint ignorou o eixo temporal/acústico.
- Áudio e encoder verificados: Whisper transcreve os WAVs corretamente; o modelo pré-treinado transcreve inglês corretamente pelo mesmo caminho (`forward` + `spk_targets=1`, `bg_spk_targets=0`); as features do encoder congelado têm norma pequena (~2 por frame), pouco visíveis para uma projeção `joint.enc` aleatória.
- Verificação do pipeline com verdade conhecida: o modelo pré-treinado (vocabulário inglês) pelo **mesmo** caminho Lhotse/`validation_ds` em LibriSpeech obtém `val_wer=0,21` (referências sem caixa/pontuação) e perda RNN-T de 0,48 em modo de treino → dataloader, `spk_targets`, perda fundida e decodificação estão corretos. Pareamento áudio↔texto conferido com Whisper em 40 amostras aleatórias (WER mediano 0,12 treino / 0,05 val).
- Experimentos (todos com `val_wer` ≥ 0,993, sem quebra do regime de *blank*):

| Variante | Encoder | Passos | `train_loss` final | Melhor `val_wer` |
|---|---|---|---|---|
| v1: Joint/Decoder aleatórios | congelado | 2.520 | 0,74 (memorização) | 0,9931 |
| v2: §5.6 (`joint.enc`/`joint.pred`/LSTM preservados) | congelado | 1.260 (interrompido) | ~2,5 | 0,9947 |
| v2 + encoder descongelado (LR 1e-4, kernels congelados) | treinável | 420 | 3,47 | 0,9931 |
| v2 + transferência de vocabulário por string (358/1024 peças + *blank*) | congelado | 900 | 4,38 | 1,0 |
| idem | treinável (LR 1e-4) | 900 | 3,63 | 1,0 |

- Conclusão: com 8,9 h / 1.380 frases únicas, nenhuma variante desenvolveu ancoragem acústica dentro do orçamento testado. O fator limitante é a **escala/diversidade do corpus**; a Fase 1 só deve ser reexecutada após ampliar os dados (MLS/CORAA/Common Voice) e, então, avaliar ajuste fino com encoder descongelado por mais passos (decisão do usuário, registrada em `USER.md`).
- §5.6 permanece ativo (inicialização mais informativa; não prejudica). A transferência de vocabulário por string **não** foi adotada: as 666 peças sem correspondência recebem linhas aleatórias cujos logits superam o *blank* calibrado (`val_wer=29,5` antes do treino, por inserções).
