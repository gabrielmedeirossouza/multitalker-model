# Spec 01: Contratos de Dados e Manifestos NeMo (PT-BR)

## 1. Objetivo
Definir os formatos acústicos, fontes de dados em Português Brasileiro (PT-BR) e o contrato estrito dos arquivos de manifesto JSON Lines (`.json` / `.jsonl`) consumidos pelo NVIDIA NeMo nas Fases 1 (Warm-up) e 2 (Multitalker).

---

## 2. Fontes de Dados Abertas em PT-BR

Os corpora abaixo compõem a base de fala isolada (single-speaker) que alimenta tanto o estágio de Warm-up quanto o simulador de mistura sintética (`Spec 02`):

| Chave (`--dataset`) | Dataset | Horas Aprox. | Estilo | Caixa/Pontuação | Locutor | Identificador no Hugging Face |
|---|---|---|---|---|---|---|
| `coraa_mupe` | **CORAA-MUPE-ASR (Museu da Pessoa)** | ~365h | Entrevistas biográficas espontâneas, sotaques e idades diversos | ✅ (`original_text`) | ✅ (`speaker_code` + `speaker_type`) | `nilc-nlp/CORAA-MUPE-ASR` |
| `coraa` | **CORAA ASR v1.1** | ~290h | Fala espontânea, entrevistas, aulas | ❌ (minúsculas, sem pontuação) | ❌ | `Racoci/CORAA-v1.1` (espelho parquet; licença CC BY-NC-ND 4.0) |
| `mls_pt` | **Multilingual LibriSpeech (MLS)** | ~160h | Audiolivros (ortografia antiga: "elle", "janella") | ❌ (minúsculas, sem pontuação) | ✅ (`speaker_id`) | `facebook/multilingual_librispeech` (`portuguese`) |
| `common_voice_pt` | **Mozilla Common Voice (PT)** | ~140h–300h+ | Frases curtas lidas | ✅ | ✅ (`client_id`) | `mozilla-foundation/common_voice_17_0` (`pt`; exige token HF, disponibilidade não verificada) |
| `fleurs_pt` | **Google FLEURS** | ~12h | Leitura limpa (reservado prioritariamente para `val`/`test`) | ✅ (`raw_transcription`) | ❌ | `google/fleurs` (`pt_br`) |
| — | **CETUC (Alcaim)** | ~145h | 100 locutores × 1.000 frases foneticamente balanceadas | — | — | Repositório `falabrasil/speech-datasets` (não integrado) |

- **Corpus da Fase 1 (`step_05`):** `coraa_mupe`. É o único corpus grande com caixa e pontuação (exigidas pela `Spec 03`) e com identificação de locutor (exigida pelo mixer da `Spec 02`). O FLEURS sozinho (8,9 h, 1.380 frases únicas) é insuficiente (`Spec 04 §8.5`).
- **Licenças:** CORAA v1.1 é CC BY-NC-ND 4.0 (uso não comercial). O projeto é de uso não comercial.

### 2.1. Contrato de Fonte Hugging Face (`HFDatasetSource` em `src/schemas.py`)

Cada chave de `SUPPORTED_HF_DATASETS` (`src/data_prep.py`) é um `HFDatasetSource`:

| Campo | Significado |
|---|---|
| `hf_path` / `hf_config` | Repositório e configuração do dataset no Hub. |
| `text_field` | Coluna de transcrição (com fallback para `transcription`). |
| `split_aliases` | Mapeia o split canônico do pipeline (`train`, `validation`, `test`) para o nome no Hub (ex.: CORAA v1.1 usa `dev`). Os nomes dos `.wav` usam sempre o split canônico. |
| `speaker_fields` | Colunas concatenadas com `_` para formar `speaker_id` (ex.: `speaker_code` + `speaker_type` distingue entrevistado e entrevistador da mesma entrevista). Vazio → `speaker_id`/`client_id`/`speaker` ou, na falta, um locutor único por amostra. |
| `allowed_values` | Filtro de metadados: a amostra só entra se `item[campo]` estiver na lista (ex.: `audio_quality: ["high"]` no MUPE). |
| `strip_annotations` | Ativa a normalização de anotações de transcrição da §2.2. |

O download é sempre em modo **streaming** (`load_dataset(..., streaming=True)`): o MUPE tem 42 GB de parquet, e o modo não-streaming duplicaria esse volume no disco da VM (cache parquet + cache Arrow) antes de gerar os `.wav`.

### 2.2. Normalização de Transcrições

Aplicada a toda transcrição antes do manifesto (`normalize_transcript()`):
1. Normalização Unicode NFC e colapso de espaços (inclui `\xa0`).
2. Com `strip_annotations` (MUPE):
   - Anotações de eventos não-verbais entre parênteses são removidas: `(risos)`, `(pausa)`, `(tosse)`, `()`, `( )`.
   - Amostras com `<` ou `>` são **descartadas**: marcam palavras truncadas na borda do segmento (ex.: `"Então, mata um pouco a sauda>"`), em que o texto não corresponde ao áudio.
   - Espaços antes de `, . ? !` deixados pela remoção são eliminados.
3. Transcrição vazia após a normalização → amostra descartada.

### 2.3. Subamostragem da Validação (`val_stride`)

`--val_stride K` mantém 1 a cada `K` amostras do split de validação (distribuídas ao longo de todas as entrevistas, ao contrário de um corte nas primeiras `N`). O MUPE tem 9.894 segmentos de validação (~11 h); `K = 4` reduz o custo de cada validação na A100 a ~2,5 mil amostras.

---

## 3. Regras e Restrições Acústicas Obrigatórias

1. **Taxa de Amostragem (`sample_rate`):** Obrigatoriamente **16.000 Hz** (`16 kHz`).
2. **Canais (`channels`):** Obrigatoriamente **1 canal (mono)**. Áudios estéreo devem ser convertidos para mono pela média dos canais antes de qualquer processamento.
3. **Codificação e Amplitude:** Arquivos `.wav` (`PCM_16` em disco; `float32` normalizado em `[-1.0, 1.0]` em memória).
4. **Limites de Duração:**
   - Duração mínima: **`0.5` segundos** (evita falhas de subsampling 8x no FastConformer).
   - Duração máxima no contrato (`MultitalkerManifestEntry`): **`30.0` segundos**.
   - Duração máxima padrão no pipeline de preparação e treino A100: **`20.0` segundos** (evita descarte silencioso pelo dataloader do NeMo e *Out-Of-Memory* na Joint Network 4D do RNN-T na A100).
5. **Execução Direta no Colab + Google Drive (Zero Armazenamento de Áudio Local):**
   - Nenhum dataset de áudio deve ser baixado no computador local do usuário (que é reservado exclusivamente para código, specs e testes de contrato em memória `< 0.3 GB`).
   - O download dos datasets (`step_01`), treino do tokenizador (`step_02`), mixagem sintética (`step_03`) e empacotamento `.tar.gz` (`step_04`) rodam diretamente no SSD NVMe da VM do Google Colab (`/content/local_dataset/`) com persistência automática em `/content/drive/MyDrive/parakeet_ptbr/<dataset>/` via `python -m src.data_prep --ensure_colab_dataset`.
   - **Artefatos por corpus:** pacote e tokenizador ficam em `<drive_root_dir>/<dataset>/` (ex.: `/content/drive/MyDrive/parakeet_ptbr/coraa_mupe/ptbr_dataset_16khz.tar.gz` e `.../coraa_mupe/tokenizer_ptbr_1024`). Assim, trocar `--dataset` nunca reaproveita o pacote ou o tokenizador de outro corpus. (Os artefatos legados do FLEURS ficaram na raiz `/content/drive/MyDrive/parakeet_ptbr/`.)
   - Caso o pacote do corpus já exista no Google Drive em sessões subsequentes, `--ensure_colab_dataset` apenas o extrai para `/content/local_dataset/`.
   - O `.tar.gz` usa `compresslevel=1`: WAV PCM comprime pouco, e o nível 9 padrão do `tarfile` levaria dezenas de minutos para ~40 GB.

---

## 4. Schema do Manifesto JSON Lines (`.jsonl`)

Cada linha do arquivo de manifesto é um objeto JSON independente validado via `src/schemas.py`.

### 4.1. Manifesto Single-Speaker / Warm-up (`SingleSpeakerManifestEntry`)
Para áudios limpos de 1 único locutor, as máscaras ainda são fornecidas com `spk_targets` preenchido com `1` (locutor ativo) e `bg_spk_targets` preenchido com `0` (sem locutor concorrente), preservando a compatibilidade com os *Speaker Kernels*:

```json
{
  "audio_filepath": "/content/local_dataset/clean/audio_0001.wav",
  "duration": 4.24,
  "text": "Olá, hoje vamos revisar os resultados do trimestre no Brasil.",
  "speaker_id": "coraa_spk_042",
  "spk_targets": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  "bg_spk_targets": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}
```

### 4.2. Manifesto Multitalker (`MultitalkerManifestEntry`)
Para misturas sintéticas sobrepostas, cada linha especifica o áudio misturado, o texto do locutor-alvo daquela instância e as máscaras de atividade em quadros de **80 ms** (`0.08s`):

- **Fórmula de Alinhamento de Quadros:**
  $$\text{expected\_frames} = \left\lceil \frac{\text{duration}}{0.08} \right\rceil$$
- **Tolerância de Arredondamento:**
  $$\left|\text{len}(\texttt{spk\_targets}) - \text{expected\_frames}\right| \le 1$$
- **Igualdade de Dimensão:**
  $$\text{len}(\texttt{spk\_targets}) == \text{len}(\texttt{bg\_spk\_targets})$$

---

## 5. Critérios de Aceite (Testes Automatizados)

- `tests/test_manifest_schema.py` deve:
  1. Aceitar registros válidos onde `len(spk_targets) == ceil(duration / 0.08)`.
  2. Rejeitar com `ValidationError` qualquer registro com duração `< 0.5s` ou `> 30.0s`.
  3. Rejeitar com `ValidationError` qualquer registro onde `spk_targets` e `bg_spk_targets` divirjam da duração esperada em mais de 1 frame ou tenham comprimentos diferentes entre si.
  4. Rejeitar valores fora do intervalo `[0, 1]` nas máscaras.
  5. Validar que toda entrada de `SUPPORTED_HF_DATASETS` é um `HFDatasetSource` e que `coraa_mupe` lê `original_text`, separa locutores por `speaker_code` + `speaker_type` e filtra `audio_quality == "high"`.
  6. Validar `normalize_transcript()`: remoção de anotações entre parênteses, descarte de segmentos truncados (`<`/`>`), NFC e colapso de `\xa0`.
  7. Validar `export_hf_dataset_to_nemo_manifest()` com um dataset falso em memória: alias de split, filtro `allowed_values`, `speaker_id` composto e `val_stride`.
  8. Validar que `ensure_colab_dataset_ready()` procura pacote e tokenizador em `<drive_root_dir>/<dataset>/`.
