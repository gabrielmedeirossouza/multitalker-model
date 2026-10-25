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

### [DISC-010] Campo raw_transcription vs transcription no google/fleurs (datasets 4.8.5) (`data_prep` — `2026-10-24T18:11:03Z`)
- **Problema / Armadilha:** No dataset google/fleurs (pt_br), o campo 'transcription' remove toda pontuação e converte tudo para minúsculas, violando a Spec 03 que exige preservação de case e pontuação (., ?, -, vírgula).
- **Solução / Regra Obrigatória:** Configurar SUPPORTED_HF_DATASETS['fleurs_pt'] para ler 'raw_transcription' (que preserva maiúsculas e pontuação original) com fallback automático para 'transcription'.

### [DISC-011] Automação Remota via Google Colab CLI (~/.local/bin/colab) e Adoção de Sessões (`colab_io` — `2026-10-24T18:11:03Z`)
- **Problema / Armadilha:** Sessões iniciadas pelo navegador no Google Colab aparecem como '[?]' em 'colab sessions' por não estarem registradas localmente em ~/.config/colab-cli/sessions.json, e 'colab exec' exige código via stdin ou '-f <arquivo>'.
- **Solução / Regra Obrigatória:** Usar '~/.local/bin/colab' autenticado via OAuth2 (~/.config/colab-cli/token.json). Para adotar sessão já montada no navegador com /content/drive, registrar SessionState via state.client.list_assignments() em /Users/gabriel.medeiros/.local/share/uv/tools/google-colab-cli/bin/python e enviar código via stdin ('cat << EOF | ~/.local/bin/colab exec -s parakeet --timeout 600').

### [DISC-012] Remoção de split_by_punctuation no SentencePieceTrainer (sentencepiece 0.2.2) (`tokenizer` — `2026-10-24T18:27:08Z`)
- **Problema / Armadilha:** O parser de TrainerSpec do sentencepiece 0.2.2 lança 'RuntimeError: NOT_FOUND: unknown field name split_by_punctuation in TrainerSpec'.
- **Solução / Regra Obrigatória:** Remover o argumento 'split_by_punctuation=True' de spm.SentencePieceTrainer.train(...) em src/tokenizer.py, mantendo user_defined_symbols para pontuação sem vírgula e required_chars para a vírgula.

### [DISC-013] Crash C++ SIGABRT (bpe_model_trainer.cc:45 Check failed: freq > 0) ao sobrepor user_defined_symbols e required_chars (`tokenizer` — `2026-10-24T19:08:28Z`)
- **Problema / Armadilha:** Passar os mesmos sinais de pontuação em user_defined_symbols e required_chars faz o SentencePiece C++ consumir a pontuação antes de contar chars_ (freq=0) e abortar o processo com SIGABRT ao validar required_chars no fim do BPE.
- **Solução / Regra Obrigatória:** Não passar pontuação em user_defined_symbols quando ela já está presente em required_chars com character_coverage=1.0, e treinar o SentencePiece em tempfile.TemporaryDirectory() antes de copiar para o Google Drive.

### [DISC-014] Prevenção de Travamento Silencioso no Colab CLI: Subprocessos Isolados + Watchdog Timer (`colab_io` — `2026-10-24T19:08:28Z`)
- **Problema / Armadilha:** Se uma biblioteca C++ (SentencePiece, PyTorch, NeMo) sofrer SIGABRT/SIGSEGV dentro do processo ipykernel do Colab, o servidor Jupyter nunca envia execute_reply no WebSocket e o colab exec fica bloqueado indefinidamente.
- **Solução / Regra Obrigatória:** Sempre executar rotinas pesadas/C++ no Colab via subprocesso isolado ou job em background (gravando logs em /tmp/*.log e status de saída) e sempre agendar um watchdog timer (schedule DurationSeconds com TimerCondition='any') para monitorar o progresso sem depender do usuário.

### [DISC-015] Guarda não-bloqueante src.colab_guard vs colab-mcp e travamento em /dev/tty no drivemount (`colab_guard` — `2026-10-24T19:32:59Z`)
- **Problema / Armadilha:** O comando 'colab drivemount' abre '/dev/tty' e bloqueia indefinidamente esperando Enter quando a VM exige OAuth dfs_ephemeral; já 'drive.mount()' dentro de 'colab exec' não instala o colab_request_hook e trava até o timeout. Além disso, o 'googlecolab/colab-mcp' exige sessão aberta no navegador (WebSocket proxy local) e também bloqueia em células síncronas.
- **Solução / Regra Obrigatória:** Toda interação com o Colab DEVE usar '.venv/bin/python -m src.colab_guard' (Spec 06) + 'src/colab_cli_bridge.py'. Ele roda sem TTY controlador (start_new_session=True), testa 'credentials-propagation' em dryrun antes de tocar no kernel, detecta pedidos OAuth no stream em <2s, registra bloqueio automático em USER.md e retorna exit code 10 (needs_user) para o agente parar e avisar o usuário imediatamente.

### [DISC-016] Protocolo dfs_ephemeral: dryrun=false propaga consentimento existente sem exigir navegador e check via /proc/mounts (`colab_drive_auth` — `2026-10-24T19:43:14Z`)
- **Problema / Armadilha:** O endpoint '/tun/m/credentials-propagation/{endpoint}' com 'dryrun=true' SEMPRE retorna '{"success": false, "unauthorized_redirect_uri": "..."}' mesmo quando a conta Google já autorizou o app ('O app Google Drive for desktop já tem algum acesso'). Isso fazia o 'colab drivemount' original pedir confirmação no navegador toda vez. Além disso, checar '/api/contents/drive/MyDrive' via REST sofre ReadTimeout quando o Drive não está montado, e 'ColabRuntime' com 'kernel_id' em cache pode lançar 'RuntimeError: Connection was lost.' na primeira conexão.
- **Solução / Regra Obrigatória:** Em 'src/colab_cli_bridge.py', '_try_propagate_or_get_auth_url()' chama primeiro 'dryrun=false' (que retorna '{"success": true}' imediatamente quando o consentimento já existe) e só recorre a 'dryrun=true' para extrair a URL se 'dryrun=false' retornar 'success=false'. A checagem de montagem usa '/proc/mounts' + 'os.path.isdir' via '_connect_warm_runtime()' com retry automático para 'Connection was lost'.

### [DISC-017] NeMo 3.0 EncDecMultiTalkerRNNTBPEModel exige Lhotse, trainer.logger=false e max_steps explícito (`nemo_trainer` — `2026-10-24T20:09:16Z`)
- **Problema / Armadilha:** O checkpoint usa EncDecMultiTalkerRNNTBPEModel (nemo_toolkit 3.0.0 PyPI). Sem use_lhotse=true o _setup_dataloader_from_config retorna None silenciosamente; pl.Trainer com logger padrão faz o exp_manager lançar LoggerMisconfigurationError; com max_epochs apenas, setup_optimization chama len(train_dataloader.dataset) e quebra (LhotseSpeechToTextSpkBpeDataset não tem __len__); val_check_interval fracionário é rejeitado pelo Lightning sem len().
- **Solução / Regra Obrigatória:** configs/*.yaml: train_ds/validation_ds.use_lhotse=true, text_field=text, trainer.logger=false, use_distributed_sampler=false, val_check_interval=1.0 (ou int com check_val_every_n_epoch=null), sched.warmup_ratio. src/trainer.py calcula trainer.max_steps=ceil(max_epochs*ceil(N/batch)/accum) via compute_max_steps() quando o YAML não define (Spec 04 §8.3-8.4); CLI --max_steps para smoke tests.

### [DISC-018] Speaker Kernels fora do encoder, perda descartada por change_vocabulary e RNNTJoint somente leitura (`nemo_trainer` — `2026-10-24T20:09:16Z`)
- **Problema / Armadilha:** spk_kernels/bg_spk_kernels são módulos de topo (hooks no encoder): encoder.freeze() os deixa treináveis no warmup. change_vocabulary() do NeMo 3.0 recria RNNTLoss(num_classes) padrão descartando fastemit_lambda/rnnt_reduction. Atribuir joint.fuse_loss_wer lança AttributeError (propriedade sem setter) e set_fuse_loss_wer() zera loss/WER fundidos se chamado sem eles.
- **Solução / Regra Obrigatória:** freeze_speaker_kernels() após encoder.freeze() no warmup; rebuild_rnnt_loss_from_cfg() após apply_stage_optimizations_to_nemo_model(); usar joint.set_fused_batch_size() e só chamar set_fuse_loss_wer(v, loss=model.loss, metric=model.wer) quando o valor mudar (Spec 04 §5).

### [DISC-019] LhotseSpeechToTextSpkBpeDataset ignora spk_targets do manifesto (pendência da Fase 2) (`dados_multitalker` — `2026-10-24T20:09:16Z`)
- **Problema / Armadilha:** O dataset do NeMo deriva spk_targets/bg_spk_targets das supervisions do Cut Lhotse (speaker_to_target) e sorteia o locutor-alvo; os campos spk_targets/bg_spk_targets do nosso manifesto são ignorados. Manifesto NeMo JSON vira MonoCut com 1 supervision (duração inteira): correto para fala limpa (Fase 1), mas as misturas espelhadas da Spec 02 seriam vistas como fala única do alvo em toda a duração.
- **Solução / Regra Obrigatória:** Antes da step_06, o mixer (Spec 02) deve emitir cuts Lhotse (ex.: CutSet jsonl/shar) com uma SupervisionSegment por locutor (speaker, start, duration, text) — ou MixedCut de MonoCuts com offsets — sem espelhamento duplo (o dataset sorteia o alvo). Validar com speaker_to_target() antes de alocar GPU (Spec 04 §8.2).

### [DISC-020] Warmup v1 falhou por atalho de LM: change_vocabulary reinicializa joint.enc e o Decoder memoriza o FLEURS (`nemo_trainer` — `2026-10-24T20:41:17Z`)
- **Problema / Armadilha:** Com Joint/Decoder totalmente reinicializados e encoder congelado, train_loss caiu 24,4->0,74 mas val_wer=0,9931 (hipóteses de 1 token). Viterbi mostrou dezenas de tokens num único frame; nll/token 0,27-0,36 no treino vs 11-17 na validação. FLEURS pt_br tem só 1.380 frases únicas (8,9 h), memorizáveis pela rede de predição; as features do encoder têm norma ~2/frame, pouco visíveis para uma projeção joint.enc aleatória. Áudio OK (Whisper transcreve) e caminho forward+spk_targets OK (inglês transcrito corretamente).
- **Solução / Regra Obrigatória:** Preservar a allowlist joint.enc.*, joint.pred.*, decoder.prediction.dec_rnn.* (snapshot/restore em torno de change_vocabulary; reuse_vocab_independent_weights=true), nunca embed/joint_net (mesmo shape no vocab original de 1024). Diagnosticar falhas de RNN-T com alinhamento de Viterbi do caminho de referência (tokens agrupados num frame = atalho de LM). Para qualidade final, ampliar o corpus (MLS/CORAA/Common Voice) (Spec 04 §5.6, §8.5).

### [DISC-021] src.state: --cmd sombreava o subcomando (start-step/fail-step não executavam) (`state_cli` — `2026-10-24T20:42:33Z`)
- **Problema / Armadilha:** O subparser usava dest='cmd' e start-step/fail-step têm a opção --cmd; o valor de --cmd sobrescrevia args.cmd, nenhum ramo do dispatch casava e o comando terminava com sucesso sem alterar o estado.
- **Solução / Regra Obrigatória:** build_parser() com add_subparsers(dest='subcommand'); dispatch por args.subcommand; teste de regressão test_cli_cmd_option_does_not_shadow_subcommand. Sempre conferir a mensagem '[Spec 05] ...' de confirmação após cada comando do src.state.

### [DISC-022] Pipeline verificado em inglês; FLEURS 8,9 h insuficiente para PT-BR em todas as variantes de warmup testadas (`nemo_trainer` — `2026-10-24T21:20:53Z`)
- **Problema / Armadilha:** Após corrigir o pipeline, 5 variantes (Joint aleatório/preservado, encoder congelado/descongelado LR 1e-4, transferência de vocabulário por string) ficaram com val_wer >= 0,993 em 420-2.520 passos. O modelo pré-treinado em inglês, pelo mesmo caminho Lhotse, obtém val_wer=0,21 e perda 0,48; pareamento áudio-texto ok (Whisper, 40 amostras).
- **Solução / Regra Obrigatória:** Não reexecutar a Fase 1 só com FLEURS. Ampliar o corpus (mls_pt, coraa, common_voice_pt já suportados em SUPPORTED_HF_DATASETS; Common Voice exige token HF/termos aceitos) e então avaliar ajuste fino com encoder descongelado por mais passos. Para diagnosticar, usar: (1) modelo pré-treinado no mesmo dataloader com verdade conhecida; (2) Whisper para pareamento; (3) Viterbi do caminho de referência. Lightning 2.x preserva o modo train/eval de cada submódulo no fit: não chamar model.eval() antes de trainer.fit (cudnn RNN backward exige modo train).

### [DISC-023] CORAA-MUPE-ASR como corpus da Fase 1; ID nilc-nlp/CORAA-ASR inexistente no Hub (`data_prep` — `2026-10-24T23:21:10Z`)
- **Problema / Armadilha:** O ID nilc-nlp/CORAA-ASR responde 401 no Hugging Face. CORAA v1.1 (Racoci/CORAA-v1.1) e MLS são minúsculos e sem pontuação. O MUPE (nilc-nlp/CORAA-MUPE-ASR, 365 h, 42 GB parquet com WAV 16 kHz) tem original_text com caixa/pontuação, mas inclui anotações '(risos)', '(pausa)', '()' (~7% dos segmentos) e '<'/'>' marcando palavras truncadas na borda (~3,4%); speaker_type distingue entrevistado (R) e entrevistador (P/1).
- **Solução / Regra Obrigatória:** Usar --dataset coraa_mupe: HFDatasetSource com original_text, speaker_fields [speaker_code, speaker_type], audio_quality=high e strip_annotations (remove parênteses, descarta '<'/'>'). Download sempre em streaming (evita cache parquet + Arrow de ~84 GB). Artefatos por corpus em parakeet_ptbr/<dataset>/ e checkpoints em exp_<dataset>_warmup (Spec 01 §2-§3). datasets-server do HF limita /rows (HTTP 429) após ~35 requisições seguidas.

### [DISC-024] launch falha com timeout logo após exec com datasets em streaming; throughput do MUPE (`colab_io` — `2026-10-24T23:32:03Z`)
- **Problema / Armadilha:** Após um exec que iterou um IterableDataset do HF em streaming, o launch seguinte retornou timeout (exit 11) em 90 s sem criar o job: o kernel ainda estava ocupado com threads do streaming. Export do coraa_mupe numa VM CPU High-RAM (8 vCPU, 50 GB RAM, 226 GB disco) roda a ~62 amostras/s em 1 núcleo (~70 min para o train).
- **Solução / Regra Obrigatória:** Após timeout de launch, conferir job-status (job_not_found = não iniciou) e repetir o launch uma vez: o guarda já interrompe o kernel. Evitar exec pesado com datasets em streaming no kernel; usar launch. Estimar Steps 01-04 do MUPE em ~1,5 h (export ~70 min + tokenizador + mixer + tar ~40 GB).

### [DISC-025] Google Drive do usuário tem 5 TB (`ambiente_colab` — `2026-10-24T23:36:06Z`)
- **Problema / Armadilha:** A VM do Colab não expõe a cota do Drive (df em /content/drive mostra o disco local da VM, não a cota), e os agentes vinham tratando o espaço no Drive como gargalo.
- **Solução / Regra Obrigatória:** Segundo o usuário (2026-10-24), o Drive tem 5 TB: pacotes de ~40 GB e checkpoints na Lixeira não são gargalo. Manter save_top_k=1 por custo de I/O, mas não bloquear o pipeline por cota.

### [DISC-026] Fase 2 resolvida via CutSet Lhotse derivado do manifesto espelhado; cut.custom não pode ser None (`dados_multitalker` — `2026-10-24T23:47:44Z`)
- **Problema / Armadilha:** O LhotseSpeechToTextSpkBpeDataset ignora spk_targets do manifesto (DISC-019). Cuts Lhotse escritos à mão sem 'custom' quebram o speaker_to_target() do NeMo 3.0 com AttributeError ('NoneType' object has no attribute 'get'), pois ele chama cut.custom.get('rttm_filepath'). O NeMo também oferece o simulador on-the-fly multi_speaker_simulator (lsmix), mas sem SIR/anti-clipping e com iterador infinito, inviável para validação.
- **Solução / Regra Obrigatória:** src/mixer.py export_lhotse_cuts() converte o manifesto espelhado em <manifesto>_cuts.jsonl (par mix_* -> 1 MonoCut com 2 supervisions do 1º ao último frame ativo de cada máscara; linha limpa -> 1 supervision [0, D]; custom={}), e src/trainer.py configure_multitalker_cuts() troca manifest_filepath por cuts_path na Fase 2 (Spec 02 §3.7, Spec 04 §8.2). Não é preciso remixar nem reempacotar o .tar.gz. Validado na VM (lhotse 1.33, NeMo 3.0) pelo caminho real get_lhotse_dataloader_from_config: máscaras batem com as nossas até 1 frame nas bordas. Pendência da step_07: src/evaluator.py chama transcribe() nas misturas sem informar o locutor-alvo.

### [DISC-027] Misturas de validação sobrescreviam as de treino (mesmo mix_<idx>.wav); CutSet com 1 cut por par quebrava o 70/30 (`dados_multitalker` — `2026-10-25T01:46:37Z`)
- **Problema / Armadilha:** data_prep chama o mixer de treino e de validação com o mesmo output_dir e o mixer nomeava os WAVs mix_<idx:06d>.wav: as 200 misturas de validação sobrescreveram mix_000000..000199 do treino (no CORAA-MUPE, mix_000000.wav tem 3,72 s mas o manifesto de treino diz 5,79 s), deixando texto/máscaras sem relação com o áudio e vazando validação no treino. A Fase 1 não é afetada (só usa clean_*). Além disso, o CutSet com 1 cut por par espelhado (DISC-026) mudava a composição por amostra de 70/30 para ~54/46.
- **Solução / Regra Obrigatória:** WAVs nomeados mix_<stem do manifesto de saída>_<idx:06d>.wav (Spec 02 §3.6) e teste de regressão com treino+val no mesmo diretório. export_lhotse_cuts() emite 1 cut por linha do manifesto (o par vira 2 cuts com as 2 supervisions; o dataset sorteia o alvo em cada leitura), preservando 70/30 e max_steps. Padrões do data_prep: 50.000 misturas de treino e 500 de validação. As misturas do pacote coraa_mupe atual (1.000/200, nomes antigos) devem ser apagadas e regeneradas antes da step_06.

### [DISC-028] Mixer single-thread a ~11 misturas/s; gerar em shards paralelos (`dados_multitalker` — `2026-10-25T02:09:47Z`)
- **Problema / Armadilha:** generate_mixed_dataset_from_manifest roda em 1 processo (~11 misturas/s na VM A100): 50.000 misturas levariam ~73 min. Com o warmup rodando ao lado, 7 processos de mixer reduziram o warmup de ~6,5 para ~4,6 passos/s durante ~15 min.
- **Solução / Regra Obrigatória:** Rodar N processos de src.mixer com --output_manifest mt_train_part<k>.json e --seed 42+k (os nomes de WAV por stem do manifesto não colidem) e concatenar as partes em multitalker_train_manifest.json: os pares espelhados continuam consecutivos e a proporção 70/30 vale por parte. 6 shards + validação: ~58 misturas/s (50.004 misturas em ~15 min). Script usado: /content/mix_par.sh no job mix_mupe_par.

### [DISC-029] mv sobre arquivo grande no Drive travou o DriveFS ~25 min e bloqueou o trainer pelo stdout (`colab_drive` — `2026-10-25T03:34:49Z`)
- **Problema / Armadilha:** O reempacotamento gravou ptbr_dataset_16khz.tar.gz.tmp (35,7 GB) e fez mv sobre o .tar.gz antigo (27 GB, nunca baixado nesta VM). O DriveFS registrou SWAPPED_ITEM_NOT_FULLY_DOWNLOADED, um PushTask ficou 'Opening content cache streams' por ~25 min com active_uploads=0, e toda operação no FUSE bloqueou: job-status (lê o log no Drive), o kernel (preso em fuse_simple_request, interrupt não libera) e o trainer, cujo stdout e checkpoints vão para o Drive (o save do .ckpt final levou 22 min). Depois do treino, o processo do trainer ficou pendurado na saída (34 threads em futex_wait) mesmo com .nemo e complete_step concluídos.
- **Solução / Regra Obrigatória:** Nunca sobrescrever arquivo grande no Drive com mv: apagar o antigo antes, ou gravar com nome novo. Reempacotamento e escritas grandes no Drive não devem coincidir com saves de checkpoint. Logs de treino em job-dir local (/content/jobs), checkpoints no Drive. Kernel preso em syscall do FUSE: python -m src.colab_guard restart (Spec 06 §5.6; jobs sobrevivem). Diagnóstico sem tocar o Drive: /proc/<pid>/wchan, /proc/<pid do drive>/io, /proc/net/dev e ~/.config/Google/DriveFS/Logs/drive_fs.txt. Trainer pendurado após .nemo + estado 'completed': conferir o .nemo (tarfile) e usar job-kill.

### [DISC-030] Fase 2 limitada pela CPU: checagem skip_nan_grad por tensor (~23%) e AdamW não fundido (~30%) (`nemo_trainer` — `2026-10-25T03:44:32Z`)
- **Problema / Armadilha:** Fase 2 (622 M parâmetros treináveis, lote 16, acumulação 2) a ~2,9 lotes/s (~10 h para 12 épocas), GPU 38-55% e processo principal a 100% de um núcleo, workers do dataloader ociosos. py-spy: ~23% em ASRModel.on_after_backward, porque o checkpoint traz skip_nan_grad=true e o NeMo faz isnan/isinf .any() por parâmetro (1 sync GPU->CPU por tensor por micro-lote), e ~30% no passo do AdamW não fundido. Na Fase 1 (8,9 M treináveis) o custo era desprezível.
- **Solução / Regra Obrigatória:** install_fast_nan_grad_guard() (Spec 04 §8.6) desliga _skip_nan_grad e zera os gradientes se torch.nn.utils.get_total_norm não for finita (1 sync); optim.fused: true no a100_multitalker.yaml (o NeMo repassa chaves extras ao torch.optim.AdamW). Validado na VM: torch 2.11 tem get_total_norm e AdamW(fused=True) funciona. Para diagnosticar vazão: py-spy record --pid <trainer> --duration 15 --format raw + contagem de frames.
