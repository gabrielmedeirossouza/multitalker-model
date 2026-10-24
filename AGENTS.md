# AGENTS.md — Guia de Agentes e Regras de Spec-Driven Development (SDD)

Este repositório utiliza **Spec-Driven Development (SDD)** para adaptar o modelo **`nvidia/multitalker-parakeet-streaming-0.6b-v1`** ao **Português Brasileiro (PT-BR)** usando **NVIDIA NeMo** e **Google Colab (GPU A100)** como motor de computação remota.

Qualquer agente de IA (Antigravity, Claude Code, Cursor, GitHub Copilot, etc.) ou desenvolvedor humano que atuar neste repositório **DEVE** seguir estritamente o fluxo e as invariantes descritas abaixo.

---

## 1. Princípio Fundamental do SDD (Contract-First) e Continuidade Stateful

1. **Repositório Operado Exclusivamente por Agentes de IA (`Spec 05`):**
   - O desenvolvimento e o treinamento são divididos em múltiplas etapas e sessões independentes. Todo agente de IA começa a sessão **sem contexto prévio em memória** e **DEVE** ler e atualizar os arquivos de estado (`state/pipeline_state.json`, `state/AGENT_KNOWLEDGE.md` e `USER.md`) via `.venv/bin/python -m src.state`.
2. **As Especificações (`specs/`) são a Fonte da Verdade:**
   - Nenhum código em `src/` deve ser criado ou alterado sem que a mudança esteja refletida na respectiva especificação em `specs/`.
3. **Fluxo Obrigatório de Trabalho de Todo Agente:**
   $$\text{0. Ler } \texttt{src.state status} \text{ e } \texttt{AGENT\_KNOWLEDGE.md} \longrightarrow \text{1. Atualizar } \texttt{specs/*.md} \longrightarrow \text{2. Atualizar } \texttt{src/schemas.py} \text{ e } \texttt{tests/} \longrightarrow \text{3. Implementar em } \texttt{src/} \longrightarrow \text{4. Validar com } \texttt{.venv/bin/pytest} \text{ e atualizar } \texttt{src.state}$$
4. **Manutenção Obrigatória do `USER.md` e do Conhecimento Incremental (`state/AGENT_KNOWLEDGE.md`):**
   - `USER.md` é o painel executivo enxuto (sem prosa longa) para o usuário humano. Nunca o edite manualmente: ele é gerado e validado automaticamente a partir de `state/pipeline_state.json` via `.venv/bin/python -m src.state`.
   - Sempre que houver um impedimento que dependa do humano (ex.: montar Google Drive, fornecer credenciais, subir `.tar.gz` ou alocar GPU A100 no Colab), registre-o via `.venv/bin/python -m src.state add-blocker`.
   - Qualquer descoberta técnica, script de automação ou cuidado extra descoberto durante uma execução **DEVE** ser documentado via `.venv/bin/python -m src.state log-discovery` para que o próximo agente não repita erros.
5. **Proibido "Caos de Células" no Notebook:**
   - Lógica de negócio, manipulação de áudio, geração de máscaras e configuração de treinamento **NUNCA** devem ser escritas diretamente dentro do `colab_runner.ipynb`.
   - O notebook `colab_runner.ipynb` é apenas um **executor remoto (Worker GPU)** com 4 células imutáveis que monta o Google Drive, sincroniza o repositório Git, roda `pytest tests/` como *gatekeeper*, descompacta os dados no SSD local da VM e invoca `python -m src.trainer`.

---

## 2. Mapa da Arquitetura e Diretórios

```text
multitalker-model/
├── AGENTS.md                       # Regras para agentes de IA, SDD e handover stateful (este arquivo)
├── CLAUDE.md                       # Ponteiro para AGENTS.md
├── USER.md                         # Painel enxuto de progresso e bloqueios para o usuário (gerado via src.state)
├── README.md                       # Documentação geral e guia rápido de execução
├── pyproject.toml                  # Dependências, pytest e configuração do pacote
├── state/                          # 0. ESTADO E MEMÓRIA INCREMENTAL ENTRE AGENTES (Spec 05)
│   ├── pipeline_state.json         # Máquina de estados das 8 etapas, bloqueios e handover determinístico
│   └── AGENT_KNOWLEDGE.md          # Base incremental de lições aprendidas, armadilhas e cuidados técnicos
├── specs/                          # 1. ESPECIFICAÇÕES (Fonte da Verdade)
│   ├── 00_architecture_overview.md # Arquitetura FastConformer + RNN-T + Speaker Kernels + Colab
│   ├── 01_data_contracts.md        # Contratos de áudio (16kHz mono) e schema do manifest.json
│   ├── 02_mixer_simulator.md       # Simulador de sobreposição sintética (SIR, offset, máscaras 80ms)
│   ├── 03_tokenizer_ptbr.md        # Tokenizador BPE PT-BR (1024 tokens, pontuação e acentuação)
│   ├── 04_training_pipeline.md     # Pipeline NeMo em 2 fases (Warm-up + Multitalker) na A100
│   └── 05_stateful_agent_workflow.md # Orquestração stateful entre agentes, USER.md e AGENT_KNOWLEDGE.md
├── configs/                        # Configurações Hydra/YAML do NVIDIA NeMo
│   ├── a100_warmup.yaml            # Fase 1: Encoder congelado, treino do Decoder/Joint em PT-BR
│   └── a100_multitalker.yaml       # Fase 2: Fine-tuning completo Cache-Aware + Speaker Kernels
├── src/                            # 2. IMPLEMENTAÇÃO MODULAR (Derivada das Specs)
│   ├── __init__.py
│   ├── schemas.py                  # Contratos Pydantic v2 (manifestos, configs e PipelineState)
│   ├── state.py                    # CLI e motor stateful de progresso, bloqueios, USER.md e descobertas
│   ├── data_prep.py                # Download/normalização 16kHz de datasets PT-BR e geração de .tar.gz
│   ├── mixer.py                    # Misturador sintético multitalker (70/30) e extrator de spk_targets (80ms)
│   ├── tokenizer.py                # Treino do SentencePiece BPE (1024) e change_vocabulary()
│   ├── trainer.py                  # CLI de treinamento NeMo com checkpointing contínuo no Drive
│   └── evaluator.py                # Avaliação final de WER (limpo e sobreposto) e exportação .nemo (Step 07)
├── tests/                          # 3. TESTES DE CONTRATO (Gatekeeper em CPU antes de usar GPU)
│   ├── test_manifest_schema.py     # Valida alinhamento de frames (ceil(duration / 0.08)) e contratos
│   ├── test_mixer_audio.py         # Valida mistura sintética, ausência de clipping, locutores distintos e 70/30
│   ├── test_tokenizer_spec.py      # Valida restrições de vocabulário (1024 tokens e caracteres PT-BR)
│   └── test_state_workflow.py      # Valida sincronia obrigatória entre pipeline_state.json, USER.md e AGENT_KNOWLEDGE.md
└── colab_runner.ipynb              # 4. EXECUTOR REMOTO (Worker GPU no Google Colab)
```

---

## 3. Invariantes Técnicas Inegociáveis

Ao gerar ou revisar código neste projeto, verifique sempre estas 7 regras:

1. **Taxa de Amostragem e Formato de Áudio (`Spec 01`):**
   - Todo áudio processado ou salvo deve estar estritamente em **16.000 Hz**, **1 canal (mono)**, `float32` normalizado no intervalo `[-1.0, 1.0]`.
2. **Resolução Temporal das Máscaras de Locutor (`Spec 01` e `Spec 02`):**
   - O codificador FastConformer aplica *subsampling* de 8x sobre janelas de 10 ms, resultando em **1 frame a cada 80 ms** (**1.280 amostras** a 16 kHz, ou 12,5 frames/segundo).
   - Para qualquer entrada de manifesto com duração $D$ segundos, o comprimento de `spk_targets` e `bg_spk_targets` deve satisfazer:
     $$\left|\text{len}(\texttt{spk\_targets}) - \lceil D / 0.08 \rceil\right| \le 1 \quad \text{e} \quad \text{len}(\texttt{spk\_targets}) == \text{len}(\texttt{bg\_spk\_targets})$$
3. **Prevenção de Clipping Acústico (`Spec 02`):**
   - Na soma matemática de vozes ($A + 10^{-\text{SIR}/20} \cdot B + \text{ruído}$), se $\max(|x|) > 0.99$, todo o vetor resultante deve ser reescalonado proporcionalmente para que o pico máximo seja `0.99`.
4. **Espelhamento de Amostras Multitalker (`Spec 02`):**
   - Cada mistura de 2 locutores ($A$ e $B$) deve gerar **2 registros no manifesto** apontando para o mesmo arquivo `.wav`:
     - Registro 1: Alvo = $A$ (`spk_targets = mask_A`, `bg_spk_targets = mask_B`, `text = transcript_A`).
     - Registro 2: Alvo = $B$ (`spk_targets = mask_B`, `bg_spk_targets = mask_A`, `text = transcript_B`).
5. **Vocabulário Enxuto de 1.024 Tokens (`Spec 03`):**
   - O tokenizador SentencePiece BPE em PT-BR deve ter exatamente **`vocab_size = 1024`**. Não aumente para 4.096 ou 8.192, pois a *Joint Network* do RNN-T aloca tensores 4D `(Batch, Time, Token, Vocab)` que estouram a VRAM da A100.
6. **Zero Download de Áudio Local & I/O no Google Colab (`Spec 01` e `Spec 04`):**
   - **Nunca baixe datasets de áudio reais no computador local do usuário** (o ambiente local é exclusivo para código, specs e testes de contrato em memória `< 0.3 GB`).
   - O download (`step_01`), tokenizador (`step_02`), mixagem (`step_03`) e geração do `.tar.gz` (`step_04`) rodam diretamente na VM do Google Colab (`/content/local_dataset/`) persistindo em `/content/drive/MyDrive/parakeet_ptbr/` via `python -m src.data_prep --ensure_colab_dataset`.
   - Nunca leia milhares de arquivos `.wav` diretamente de `/content/drive/MyDrive/` durante o treinamento. Os datasets ficam salvos como `.tar.gz` no Drive e extraídos no SSD NVMe local (`/content/local_dataset/`) antes de iniciar o treino.
7. **Tolerância a Desconexões no Colab (`Spec 04`):**
   - Os checkpoints (`.ckpt` e `.nemo`) devem ser gravados diretamente no Google Drive com `resume_if_exists=True` ativado no `exp_manager` do NeMo.

---

## 4. Comandos Padrão para Agentes (Ambiente Local `.venv/bin`)

- **Consultar status atual do pipeline e handover do agente anterior (obrigatório ao iniciar sessão):**
  ```bash
  .venv/bin/python -m src.state status
  ```
- **Executar a suíte de testes de contrato e verificação de sincronia do `USER.md` (obrigatório após qualquer edição):**
  ```bash
  .venv/bin/python -m src.state verify
  .venv/bin/pytest tests/ -v
  ```
- **Registrar descoberta técnica incremental ou impedimento para o usuário:**
  ```bash
  .venv/bin/python -m src.state log-discovery --category "<cat>" --title "<título>" --problem "<problema>" --solution "<solução>"
  .venv/bin/python -m src.state add-blocker --id "<id>" --step "<step_id>" --desc "<descrição>" --action "<ação do usuário>"
  ```
- **Preparar/Extrair datasets e tokenizador diretamente no Worker Colab + Google Drive (Steps 01 a 04):**
  ```bash
  python -m src.data_prep --ensure_colab_dataset --dataset fleurs_pt --local_dataset_dir /content/local_dataset --drive_root_dir /content/drive/MyDrive/parakeet_ptbr --update_state
  ```
- **Disparar treinamento (no Worker GPU / Colab A100 — Steps 05 a 07):**
  ```bash
  python -m src.trainer --stage warmup --manifest /content/local_dataset/clean_train_manifest.json --val_manifest /content/local_dataset/clean_val_manifest.json --checkpoint_dir /content/drive/MyDrive/parakeet_checkpoints/exp_v1_warmup --update_state
  python -m src.trainer --stage multitalker --manifest /content/local_dataset/multitalker_train_manifest.json --val_manifest /content/local_dataset/multitalker_val_manifest.json --checkpoint_dir /content/drive/MyDrive/parakeet_checkpoints/exp_v1_multitalker --update_state
  ```
- **Operar o Google Colab remotamente via CLI (`~/.local/bin/colab`):**
  ```bash
  ~/.local/bin/colab sessions
  ~/.local/bin/colab status
  ~/.local/bin/colab new -s parakeet --high-mem            # Aloca CPU High-RAM (Steps 01-04)
  ~/.local/bin/colab new -s parakeet --gpu A100 --high-mem # Aloca GPU A100 (Steps 05-07)
  echo "import os; print(os.listdir('/content'))" | ~/.local/bin/colab exec -s parakeet --timeout 120
  ~/.local/bin/colab stop -s parakeet
  ```
