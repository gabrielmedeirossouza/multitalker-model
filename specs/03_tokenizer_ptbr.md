# Spec 03: Tokenizador SentencePiece BPE em PT-BR (1.024 Tokens)

## 1. Objetivo
Construir um tokenizador **SentencePiece Byte-Pair Encoding (BPE)** otimizado para a morfologia, acentuação e pontuação do **Português Brasileiro (PT-BR)**, e substituir a cabeça de vocabulário do modelo `nvidia/multitalker-parakeet-streaming-0.6b-v1` preservando ~95% dos pesos pré-treinados (codificador FastConformer + *Speaker Kernels*).

---

## 2. Regras de Negócio e Restrições de Memória (VRAM)

1. **Tamanho Fixo do Vocabulário (`vocab_size = 1024`):**
   - O vocabulário **DEVE** conter exatamente **1.024 tokens** (idêntico ao checkpoint original da NVIDIA).
   - **Justificativa Técnica:** No treinamento RNN-T, a *Joint Network* materializa um tensor 4D de formato `(Batch, Time, Token, Vocab)`. Aumentar o vocabulário para 4.096 ou 8.192 multiplica por 4x–8x o consumo de VRAM na *Joint Network* e no `warprnnt_numba`, reduzindo drasticamente o `batch_size` na GPU NVIDIA A100.
2. **Preservação de Acentuação e Caracteres do Português:**
   - O conjunto de caracteres obrigatórios no vocabulário deve incluir todas as vogais acentuadas e o cedilha (maiúsculas e minúsculas):
     - Minúsculas: `a b c d e f g h i j k l m n o p q r s t u v w x y z á é í ó ú â ê ô ã õ ç à ü`
     - Maiúsculas correspondentes (já que o Parakeet emite capitalização nativa em tempo real).
3. **Pontuação Nativa em Streaming:**
   - Sinais gráficos obrigatórios no vocabulário: `.`, `,`, `?`, `!`, `-`.
   - **Cuidado de Implementação (SentencePiece):** Nunca passe `","` dentro de uma string separada por vírgulas em `user_defined_symbols` do SentencePiece, pois o parser C++ divide a string por `","` e falha com símbolo vazio. Utilize `required_chars` ou lista Python sem colisão de delimitador.
4. **Cobertura de Caracteres (`character_coverage`):**
   - `character_coverage = 0.9999` sobre o corpus textual consolidado das transcrições em PT-BR (CORAA + MLS + Common Voice), com `normalization_rule_name = "identity"` para preservar maiúsculas e acentos.

---

## 3. Procedimento de Troca de Vocabulário no NeMo

Ao executar `change_vocabulary()` sobre o `nvidia/multitalker-parakeet-streaming-0.6b-v1`:

```python
import nemo.collections.asr as nemo_asr

asr_model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/multitalker-parakeet-streaming-0.6b-v1"
)
asr_model.change_vocabulary(
    new_tokenizer_dir="/content/drive/MyDrive/parakeet_ptbr/tokenizer_ptbr_1024",
    new_tokenizer_type="bpe",
)
```

### Efeito nos Pesos da Rede:
- **Preservados (100%):** Todas as 17 camadas do codificador **FastConformer** + camadas de **Speaker Kernel Injection** (camada 0).
- **Reinicializados (0%):** Camada de *Embedding* do **Decoder RNN-T** (Prediction Network) e camada linear de projeção da **Joint Network** (`1024 + 1 blank token = 1025` saídas).

---

## 4. Critérios de Aceite (Testes Automatizados)

- `tests/test_tokenizer_spec.py` deve validar que:
  1. `TokenizerConfig` exige `vocab_size == 1024` e rejeita valores diferentes para proteger a VRAM da A100.
  2. A função de validação de cobertura verifica a presença obrigatória dos caracteres diacríticos do PT-BR (`ã`, `õ`, `ç`, `á`, `é`, `í`, `ó`, `ú`, `â`, `ê`, `ô`, `à`, `ü`) e da pontuação (`.`, `,`, `?`, `!`, `-`).
