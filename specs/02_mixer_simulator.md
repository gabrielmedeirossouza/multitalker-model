# Spec 02: Simulador de Sobreposição Acústica e Geração de Máscaras (PT-BR)

## 1. Objetivo
Combinar 2 arquivos WAV mono (16 kHz) de locutores distintos em um único WAV misturado e gerar automaticamente os vetores binários de atividade por frame de **80 ms** (`spk_targets` e `bg_spk_targets`) compatíveis com o `nvidia/multitalker-parakeet-streaming-0.6b-v1`.

---

## 2. Por Que Sobreposição Sintética é Obrigatória?

1. **Rótulo Exato por Frame (*Ground Truth* de 80 ms):** Em gravações reais de reuniões, a anotação humana falha no milissegundo exato de início/fim de cada voz sobreposta. Ao medir a energia de cada faixa isolada **antes** da soma matemática, obtemos precisão de 100% em cada frame de 80 ms (1.280 amostras).
2. **Transcrição Isolada Sem `[inaudível]`:** Cada faixa de entrada já possui sua transcrição limpa original antes da colisão acústica.
3. **Aproveitamento 2x por Mistura (Espelhamento A/B):** Cada mistura de 2 locutores ($A$ e $B$) produz **2 exemplos de treinamento** sobre o mesmo arquivo `.wav`:
   - Exemplo 1: Foco no Locutor $A$ (`spk_targets = mask_A`, `bg_spk_targets = mask_B`, `text = text_A`).
   - Exemplo 2: Foco no Locutor $B$ (`spk_targets = mask_B`, `bg_spk_targets = mask_A`, `text = text_B`).

---

## 3. Regras de Negócio e Algoritmo Matemático

### 3.1. Resolução de Amostragem e Quadros
- Taxa de amostragem fixa: $f_s = 16.000\text{ Hz}$.
- Tamanho do frame do codificador FastConformer: $T_{\text{frame}} = 0,08\text{ s}$ (**80 ms**), equivalente a exatamente:
  $$N_{\text{frame}} = 16.000 \times 0,08 = 1.280\text{ amostras/frame}$$

### 3.2. Locutores Distintos, Deslocamento Temporal (*Offset*) e Taxa de Sobreposição (*Overlap Ratio*)
- **Invariante de Locutores Distintos:** Na geração de misturas a partir de um manifesto (`generate_mixed_dataset_from_manifest`), o par sorteado $(A, B)$ deve pertencer a locutores distintos (`speaker_id_a != speaker_id_b`) sempre que houver múltiplos locutores disponíveis no manifesto.
- **Cálculo do Offset e Limite de Duração:** Dados dois sinais $x_A$ (duração $D_A$) e $x_B$ (duração $D_B$), o locutor $B$ é inserido com um atraso temporal (`offset_sec`) sorteado de forma que a **taxa de sobreposição** fique entre **15% e 50%** da duração total combinada ($r \in [0.15, 0.50]$):
  $$\text{offset\_sec} = \text{clip}\left(\frac{D_A - r \cdot D_B}{1 + r},\ 0.0,\ D_A\right)$$
- **Teto de Duração Combinada:** Caso $D_{\text{total}} = \max(D_A, \text{offset\_sec} + D_B)$ ultrapasse `max_duration_sec` (padrão `20.0s`, máximo `30.0s`), o `offset_sec` é reduzido ou o sinal excedente é truncado antes da extração das máscaras VAD, garantindo cumprimento estrito da `Spec 01`.

### 3.3. Extração de Máscaras VAD Antes da Soma
- Antes de somar os sinais alinhados no buffer de comprimento total $L_{\text{total}}$:
  - Divide-se o buffer alinhado de $A$ e de $B$ em blocos de $1.280$ amostras (o último bloco parcial é considerado para atingir $\lceil D / 0.08 \rceil$ frames).
  - Para cada frame $k \in \{0, \dots, K-1\}$, calcula-se a energia RMS:
    $$\text{RMS}(s_k) = \sqrt{\frac{1}{|s_k|} \sum_{n} s_k[n]^2}$$
  - Se $\text{RMS}(s_k) \ge \theta_{\text{vad}}$ (padrão $\theta_{\text{vad}} = 0.01$), o frame recebe `1` (fala ativa); caso contrário, recebe `0` (silêncio).

### 3.4. Variação de Ganho Relativo (*SIR — Signal-to-Interference Ratio*) e Ruído Opcional
- Para ensinar o modelo a isolar o locutor-alvo mesmo quando ele fala mais baixo que o concorrente, sorteia-se $\text{SIR}_{\text{dB}} \sim \mathcal{U}(-5.0, +5.0)\text{ dB}$.
- O fator de escala linear aplicado a $x_B$ é:
  $$g_B = 10^{-\text{SIR}_{\text{dB}} / 20}$$
- Opcionalmente (`add_gaussian_noise=True`), adiciona-se ruído acústico gaussiano de fundo $n(t)$ com SNR sorteado em `noise_snr_db_range` (`[-40.0, -20.0] dB`) **após** a extração das máscaras VAD (preservando os rótulos de voz limpa).

### 3.5. Prevenção Estrita de Clipping
- Após somar $y = x_A + g_B \cdot x_B + n$:
  $$\text{peak} = \max_{t} |y[t]|$$
  Se $\text{peak} > 0.99$:
  $$y \leftarrow y \times \frac{0.99}{\text{peak}}$$

### 3.6. Composição Híbrida para a Fase 2 (70% Multitalker + 30% Single-Speaker)
- Conforme exigido pela `Spec 04`, `generate_mixed_dataset_from_manifest` suporta o parâmetro `clean_ratio` (padrão `0.30` para o manifesto de treino da Fase 2), mesclando no manifesto final 70% de registros sobrepostos espelhados e 30% de registros single-speaker limpos (`spk_targets=1`, `bg_spk_targets=0`).

### 3.7. Exportação para `CutSet` Lhotse (Fase 2, `step_06`)
O `LhotseSpeechToTextSpkBpeDataset` do NeMo 3.0 ignora `spk_targets`/`bg_spk_targets` do manifesto: deriva as máscaras das *supervisions* de cada `Cut` e sorteia o locutor-alvo a cada leitura (`Spec 04 §8.2`). O manifesto espelhado continua sendo a fonte da verdade; `export_lhotse_cuts()` o converte num `CutSet` JSONL (`<manifesto>_cuts.jsonl`, ao lado do manifesto) que o `src/trainer.py` passa em `cuts_path` na Fase 2:
- **Par espelhado → 1 `MonoCut` com 2 supervisions:** duas linhas consecutivas com o mesmo `audio_filepath` `mix_*.wav` e máscaras invertidas (`spk_targets` de uma = `bg_spk_targets` da outra). Cada linha vira a supervision (`speaker`, `start`, `duration`, `text`) do seu locutor-alvo. Não há espelhamento no `CutSet`: o dataset sorteia o alvo.
- **Intervalo da supervision:** do primeiro ($f_0$) ao último ($f_1$) frame ativo da máscara do locutor, considerando só frames com início $< D$: $\text{start} = f_0 \cdot 0{,}08$ e $\text{end} = \min((f_1 + 1) \cdot 0{,}08,\ D)$. Pausas internas ficam dentro do intervalo (o NeMo marca a supervision inteira como fala, como no simulador `lsmix` do pré-treino). Máscara sem frame ativo (locutor abaixo de $\theta_{\text{vad}}$) → intervalo $[0, D]$.
- **Linha single-speaker** (os 30% limpos da §3.6, inclusive repetições do mesmo arquivo) → 1 `MonoCut` com 1 supervision $[0, D]$. O ID do cut é `<stem do wav>-<índice da linha>`, único no `CutSet`.
- **Rótulos de locutor distintos no cut:** se algum `speaker_id` do par estiver ausente ou ambos forem iguais, usar `spk_a`/`spk_b` (o NeMo agrupa supervisions pelo rótulo e juntaria os dois textos num alvo só).
- `Recording`: fonte `file` mono, `sampling_rate = 16000`, `num_samples = round(D · 16000)`; supervisions com `language = "pt"`; cut com `custom = {}` (o `speaker_to_target()` do NeMo chama `cut.custom.get(...)` e quebra com `None`).
- Validado na VM com `lhotse 1.33` e `nemo_toolkit 3.0`: `CutSet.from_file` + `get_lhotse_dataloader_from_config(cuts_path=...)` produzem as máscaras esperadas, com diferença de até 1 frame nas bordas pela quantização 10 ms → 80 ms do próprio NeMo.
- Linha `mix_*` sem par espelhado válido aborta a conversão com `ValueError`.

---

## 4. Critérios de Aceite (Testes Automatizados)

- `tests/test_mixer_audio.py` deve validar que:
  1. A mistura de dois sinais em 16 kHz gera um sinal mono 16 kHz cujo pico máximo nunca ultrapassa `0.99` (mesmo quando ambos os sinais de entrada têm amplitude máxima `1.0`).
  2. A função de mixagem retorna **exatamente 2 objetos `MultitalkerManifestEntry` válidos** para cada par misturado, com `spk_targets` e `bg_spk_targets` perfeitamente invertidos entre o primeiro e o segundo registro.
  3. O número de frames das máscaras corresponde a `ceil(duration / 0.08)`.
  4. `export_lhotse_cuts()` converte cada par espelhado em 1 cut com 2 supervisions de rótulos distintos, com intervalos do primeiro ao último frame ativo de cada máscara e fim dentro da duração do cut, e cada linha limpa em 1 cut com 1 supervision $[0, D]$ (§3.7).
  5. Linha `mix_*` sem par espelhado válido gera `ValueError` (§3.7).
- `tests/test_manifest_schema.py` deve validar que `MultitalkerCut` rejeita rótulos de locutor repetidos e supervisions que terminam após a duração do cut.
