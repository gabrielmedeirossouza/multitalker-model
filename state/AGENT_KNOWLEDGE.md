# AGENT_KNOWLEDGE.md — Base de Conhecimento Incremental dos Agentes de IA

> **Propósito (`Spec 05`):** Este arquivo é a memória institucional incremental entre agentes de IA que operam este repositório sem contexto prévio.
> Qualquer descoberta técnica, armadilha corrigida, peculiaridade de ambiente ou script de automação **DEVE** ser registrado aqui via:
> ```bash
> .venv/bin/python -m src.state log-discovery \
>     --category "<categoria>" \
>     --title "<título curto>" \
>     --problem "<o que acontecia / risco>" \
>     --solution "<como resolver / regra obrigatória>"
> ```

---

## 1. Regras Operacionais de Ambiente (Leitura Obrigatória no Boot)

1. **Ambiente Python Local (`.venv`):**
   - `pytest` e `python` com as dependências do projeto **não** estão no `PATH` global do macOS.
   - Use **sempre** `.venv/bin/python` e `.venv/bin/pytest`.
2. **Sincronia Obrigatória de Estado (`state/pipeline_state.json` $\leftrightarrow$ `USER.md`):**
   - Nunca edite `USER.md` manualmente. Use `src/state.py` (`start-step`, `complete-step`, `add-blocker`, `resolve-blocker`, `render-user-md`), que valida o schema Pydantic `PipelineState` e regenera `USER.md` deterministicamente.
   - Sempre execute `.venv/bin/python -m src.state verify` e `.venv/bin/pytest tests/ -v` antes de encerrar seu turno.

---

## 2. Registro Cronológico de Descobertas Incrementais (`Discoveries`)

### [DISC-001] Binários isolados em .venv/bin (`ambiente` — `2026-10-24T15:39:05Z`)
- **Problema / Armadilha:** Executar 'pytest' ou 'python' diretamente no shell falha com 'command not found: pytest' ou usa o Python do sistema sem pydantic/soundfile.
- **Solução / Regra Obrigatória:** Sempre invocar '.venv/bin/pytest' e '.venv/bin/python' em qualquer execução local.

### [DISC-002] Colisão de delimitador vírgula em user_defined_symbols do SentencePiece (`tokenizer_sentencepiece` — `2026-10-24T15:39:05Z`)
- **Problema / Armadilha:** Passar ',' dentro de uma string separada por vírgulas em user_defined_symbols (ex.: ',.?!') faz o parser C++ do SentencePiece dividir na vírgula e falhar por símbolo vazio.
- **Solução / Regra Obrigatória:** Passar todos os caracteres PT-BR e pontuação via required_chars e passar em user_defined_symbols apenas uma lista sem o caractere ',' (já coberto por required_chars e split_by_punctuation=True).

### [DISC-003] Estouro de duração máxima (30s) e colisão de mesmo locutor na mixagem (`mixer_acustico` — `2026-10-24T15:39:06Z`)
- **Problema / Armadilha:** Misturar dois áudios longos com offset temporal podia ultrapassar 20s/30s (disparando ValidationError) ou sortear dois áudios do mesmo speaker_id.
- **Solução / Regra Obrigatória:** O mixer agora sorteia pares com speaker_id distintos (_sample_distinct_speaker_pair), calcula o offset considerando D_A e D_B, limita a duração total a max_duration_sec (20.0s padrão) e suporta clean_ratio=0.30 (70% sobreposto + 30% limpo).

### [DISC-004] Injeção explícita de warprnnt_numba, fused_batch_size e att_context_size no modelo carregado (`nemo_trainer` — `2026-10-24T15:39:06Z`)
- **Problema / Armadilha:** Carregar o modelo via from_pretrained/restore_from mantinha o cfg original do checkpoint sem aplicar as chaves joint, loss e encoder.att_context_size do YAML.
- **Solução / Regra Obrigatória:** A função apply_stage_optimizations_to_nemo_model() injeta explicitamente joint.fused_batch_size, loss (warprnnt_numba) e encoder.att_context_size no asr_model.cfg antes do trainer.fit().

### [DISC-005] Separação entre caminhos locais (Steps 01-03) e reescrita /content/local_dataset (Step 04) (`fluxo_dados_colab` — `2026-10-24T15:55:50Z`)
- **Problema / Armadilha:** Passar --target_root_dir /content/local_dataset já na Step 01 gravava caminhos remotos no manifesto local, causando FileNotFoundError ao rodar o mixer localmente na Step 03.
- **Solução / Regra Obrigatória:** Manter caminhos locais nas Steps 01-03 (com fallback _resolve_local_audio_path no mixer) e executar '--prepare_colab_bundle' na Step 04 para reescrever todos os manifestos e gerar o .tar.gz.

### [DISC-006] Retomada idempotente (resume=True), handover parcial (train+val) e espelhamento no Google Drive (`continuidade_stateful` — `2026-10-24T15:55:51Z`)
- **Problema / Armadilha:** Sessões interrompidas no meio da exportação/mixagem ou quedas de VM no Colab perdiam o progresso ou marcavam a etapa como concluída antes de gerar o split de validação.
- **Solução / Regra Obrigatória:** data_prep.py e mixer.py operam com resume=True, usam update_handover() até que train e val existam, e state.py espelha automaticamente pipeline_state.json e USER.md no Google Drive.

### [DISC-007] Limite de espaço livre no disco local (~48 GiB disponíveis) (`armazenamento_local` — `2026-10-24T17:10:41Z`)
- **Problema / Armadilha:** O SSD local da máquina possui ~48 GiB livres (228 GiB total). Baixar múltiplos corpora completos (MLS + CORAA + Common Voice > 500h) sem streaming ou gerar WAVs limpos + misturados + .tar.gz simultaneamente no disco local esgotará o armazenamento.
- **Solução / Regra Obrigatória:** Para execuções locais na CPU, usar streaming (--max_samples) ou trabalhar com subsets compatíveis com o disco (~10h a ~80h, consumindo 3 a 20 GB) e limpar o cache do Hugging Face, ou executar a preparação massiva diretamente no SSD NVMe da VM do Colab (~160-200 GB livres) salvando o .tar.gz direto no Google Drive.

### [DISC-008] Aviso benigno de conflito requests==2.32.4 no Google Colab (`ambiente_colab` — `2026-10-24T17:43:08Z`)
- **Problema / Armadilha:** Ao instalar '.[dev,colab]' na Célula 1 do Colab, o pip exibe 'google-colab 1.0.0 requires requests==2.32.4, but you have requests 2.34.2 which is incompatible'.
- **Solução / Regra Obrigatória:** Este aviso do pip no Colab é benigno (o google.colab.drive já foi montado antes do pip install e o NeMo/HuggingFace operam normalmente com requests>=2.34). Não interrompa o fluxo por causa dele.

### [DISC-009] Verificação OmegaConf.is_config antes de open_dict(asr_model.cfg) (`omegaconf_colab` — `2026-10-24T17:48:58Z`)
- **Problema / Armadilha:** No Colab (onde omegaconf está instalado), chamar open_dict(asr_model.cfg) quando cfg é um dict Python comum (ex.: em mocks de teste) falha com AttributeError: 'dict' object has no attribute '_get_node_flag'.
- **Solução / Regra Obrigatória:** Usar OmegaConf.is_config(asr_model.cfg) antes de entrar no bloco open_dict(), mantendo o fallback para dict puro.
