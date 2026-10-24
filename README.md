# Parakeet PT-BR Multitalker (`nvidia/multitalker-parakeet-streaming-0.6b-v1`)

Adaptação **Cross-Lingual** para **Português Brasileiro (PT-BR)** do modelo de reconhecimento de fala em tempo real com múltiplos locutores sobrepostos (`nvidia/multitalker-parakeet-streaming-0.6b-v1`), estruturada sob a metodologia **Spec-Driven Development (SDD)** para execução em **Google Colab (NVIDIA A100)**.

---

## Visão Geral

O modelo `nvidia/multitalker-parakeet-streaming-0.6b-v1` (600M parâmetros) combina um codificador **FastConformer** *cache-aware* com um decodificador **RNN-T** e **Speaker Kernel Injection** na camada 0, isolando a voz do locutor-alvo mesmo em fala sobreposta (*cocktail party problem*) a partir de máscaras de diarização em quadros de **80 ms** (`spk_targets` e `bg_spk_targets`).

Como o diarizador (**Streaming Sortformer**) e os *Speaker Kernels* são agnósticos ao idioma, este projeto adapta o módulo ASR para o Português Brasileiro em duas fases no **NVIDIA NeMo**:
1. **Fase 1 (`warmup`):** Troca do vocabulário (`change_vocabulary`) para um tokenizador SentencePiece BPE em PT-BR de **1.024 tokens**, congelamento do FastConformer (`encoder.freeze()`) e treinamento do Decoder RNN-T + Joint Network com fala limpa em PT-BR.
2. **Fase 2 (`multitalker`):** Descongelamento do codificador com taxa de aprendizado baixa e ajuste fino *cache-aware* (`att_context_size: [[70, 0], [70, 1], [70, 6], [70, 13]]`) utilizando misturas sintéticas sobrepostas em PT-BR (70% multi-speaker + 30% single-speaker).

---

## Estrutura SDD e Stateful do Repositório

- [`AGENTS.md`](./AGENTS.md) — Regras de Spec-Driven Development, invariantes técnicas e protocolo de handover entre agentes de IA.
- [`USER.md`](./USER.md) — Painel executivo enxuto para o usuário humano com progresso atual e ações necessárias (sincronizado via `src.state`).
- [`state/`](./state/) — Estado persistente e memória incremental entre agentes:
  - [`pipeline_state.json`](./state/pipeline_state.json) — Máquina de estados das 8 etapas, bloqueios e handover.
  - [`AGENT_KNOWLEDGE.md`](./state/AGENT_KNOWLEDGE.md) — Registro incremental de descobertas técnicas e cuidados operacionais.
- [`specs/`](./specs/) — Especificações técnicas e contratos de negócio (Fonte da Verdade):
  - [`00_architecture_overview.md`](./specs/00_architecture_overview.md)
  - [`01_data_contracts.md`](./specs/01_data_contracts.md)
  - [`02_mixer_simulator.md`](./specs/02_mixer_simulator.md)
  - [`03_tokenizer_ptbr.md`](./specs/03_tokenizer_ptbr.md)
  - [`04_training_pipeline.md`](./specs/04_training_pipeline.md)
  - [`05_stateful_agent_workflow.md`](./specs/05_stateful_agent_workflow.md)
- [`configs/`](./configs/) — Arquivos YAML (Hydra/NeMo) otimizados para GPU NVIDIA A100 (`bf16-mixed`, `bucketing`, `warprnnt_numba`).
- [`src/`](./src/) — Implementação modular:
  - [`schemas.py`](./src/schemas.py) — Contratos Pydantic v2 para validação matemática dos manifestos, configurações e estado do pipeline.
  - [`state.py`](./src/state.py) — Motor CLI de estado, handover entre agentes, bloqueios, `USER.md` e descobertas incrementais.
  - [`data_prep.py`](./src/data_prep.py) — Download e padronização em 16 kHz mono de datasets em PT-BR (MLS, CORAA, Common Voice, FLEURS).
  - [`mixer.py`](./src/mixer.py) — Simulador de sobreposição sintética com deslocamento temporal, SIR (-5 a +5 dB), VAD de 80 ms, espelhamento A/B e mix 70/30.
  - [`tokenizer.py`](./src/tokenizer.py) — Construção do tokenizador SentencePiece BPE (1.024 tokens) e injeção no modelo NeMo.
  - [`trainer.py`](./src/trainer.py) — Orquestrador de treinamento para Google Colab com checkpointing resiliente no Google Drive.
  - [`evaluator.py`](./src/evaluator.py) — Cálculo de WER (fala limpa vs. sobreposta) e exportação final `.nemo` (Step 07).
- [`tests/`](./tests/) — Testes automatizados de contrato (executados em CPU antes de alocar GPU).
- [`colab_runner.ipynb`](./colab_runner.ipynb) — Notebook enxuto (4 células) para execução remota na A100 do Google Colab.

---

## Como Começar

### 1. Ambiente Local (Desenvolvimento, Estado e Testes de Contrato em CPU)
```bash
.venv/bin/python -m src.state status
.venv/bin/python -m src.state verify
.venv/bin/pytest tests/ -v
```

### 2. Execução no Google Colab (GPU A100)
Abra o arquivo [`colab_runner.ipynb`](./colab_runner.ipynb) no Google Colab com runtime **A100 GPU** e execute as 4 células sequenciais.
