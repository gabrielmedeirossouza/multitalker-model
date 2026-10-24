# Spec 01: Contratos de Dados e Manifestos NeMo (PT-BR)

## 1. Objetivo
Definir os formatos acústicos, fontes de dados em Português Brasileiro (PT-BR) e o contrato estrito dos arquivos de manifesto JSON Lines (`.json` / `.jsonl`) consumidos pelo NVIDIA NeMo nas Fases 1 (Warm-up) e 2 (Multitalker).

---

## 2. Fontes de Dados Abertas em PT-BR

Os corpora abaixo compõem a base de fala isolada (single-speaker) que alimenta tanto o estágio de Warm-up quanto o simulador de mistura sintética (`Spec 02`):

| Dataset | Horas Aprox. | Estilo | Identificador / Origem |
|---|---|---|---|
| **CORAA ASR** | ~290h | Fala espontânea brasileira, entrevistas, aulas, sotaques diversos | `nilc-nlp/CORAA-ASR` |
| **Multilingual LibriSpeech (MLS)** | ~160h | Audiolivros limpos com pontuação rica | `facebook/multilingual_librispeech` (`portuguese`) |
| **Mozilla Common Voice (PT)** | ~140h–300h+ | Frases curtas lidas com variedade de microfones | `mozilla-foundation/common_voice_17_0` (`pt`) |
| **CETUC (Alcaim)** | ~145h | 100 locutores (50M/50F) × 1.000 frases foneticamente balanceadas | Repositório `falabrasil/speech-datasets` |
| **Google FLEURS** | ~12h | Leitura limpa (reservado prioritariamente para `val`/`test`) | `google/fleurs` (`pt_br`) |
| **MuPe (Museu da Pessoa)** | ~300h+ | Entrevistas biográficas espontâneas | Projeto Tarsila / C4AI (USP) |

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
   - O download dos datasets (`step_01`), treino do tokenizador (`step_02`), mixagem sintética (`step_03`) e empacotamento `.tar.gz` (`step_04`) rodam diretamente no SSD NVMe da VM do Google Colab (`/content/local_dataset/`) com persistência automática em `/content/drive/MyDrive/parakeet_ptbr/` via `python -m src.data_prep --ensure_colab_dataset`.
   - Caso o pacote `/content/drive/MyDrive/parakeet_ptbr/ptbr_dataset_16khz.tar.gz` já exista no Google Drive em sessões subsequentes, `--ensure_colab_dataset` apenas o extrai para `/content/local_dataset/`.

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
