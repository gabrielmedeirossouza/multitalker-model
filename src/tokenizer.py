"""Treinamento e validação do tokenizador SentencePiece BPE em PT-BR (Spec 03)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Set

from src.schemas import REQUIRED_VOCAB_SIZE, TokenizerConfig

REQUIRED_PTBR_DIACRITICS: Set[str] = {
    "á", "é", "í", "ó", "ú",
    "â", "ê", "ô",
    "ã", "õ",
    "ç", "à", "ü",
}
REQUIRED_PUNCTUATION: Set[str] = {".", ",", "?", "!", "-"}


def validate_ptbr_vocabulary_tokens(tokens: Iterable[str]) -> bool:
    """
    Verifica se a lista de tokens do vocabulário cumpre a Spec 03:
    1. Tamanho exato de 1.024 tokens.
    2. Presença de todos os caracteres diacríticos obrigatórios do PT-BR.
    3. Presença dos sinais de pontuação nativos (. , ? ! -).
    """
    token_list = list(tokens)
    if len(token_list) != REQUIRED_VOCAB_SIZE:
        raise ValueError(
            f"Spec 03 violada: vocabulário deve ter exatamente {REQUIRED_VOCAB_SIZE} tokens, "
            f"mas possui {len(token_list)}."
        )

    joined_chars = set("".join(token_list).lower())
    missing_diacritics = REQUIRED_PTBR_DIACRITICS - joined_chars
    if missing_diacritics:
        raise ValueError(
            f"Spec 03 violada: caracteres acentuados do PT-BR ausentes no vocabulário: {sorted(missing_diacritics)}"
        )

    all_chars = set("".join(token_list))
    missing_punct = REQUIRED_PUNCTUATION - all_chars
    if missing_punct:
        raise ValueError(
            f"Spec 03 violada: sinais de pontuação nativos ausentes no vocabulário: {sorted(missing_punct)}"
        )

    return True


def build_required_ptbr_charset() -> str:
    """Retorna string com todos os caracteres obrigatórios do PT-BR (minúsculas, maiúsculas e pontuação)."""
    ascii_lower = "abcdefghijklmnopqrstuvwxyz"
    diacritics_lower = "".join(sorted(REQUIRED_PTBR_DIACRITICS))
    punct = "".join(sorted(REQUIRED_PUNCTUATION))
    all_chars = ascii_lower + ascii_lower.upper() + diacritics_lower + diacritics_lower.upper() + punct
    return "".join(dict.fromkeys(all_chars))


def extract_corpus_from_manifests(manifest_paths: List[str], output_txt_path: str) -> Path:
    """
    Extrai todas as transcrições dos manifestos JSONL para um arquivo de texto plano UTF-8.
    Garante que todos os caracteres de build_required_ptbr_charset() tenham frequência >= 1
    no corpus para que o SentencePiece não omita caracteres raros como 'ü' (Spec 03).
    """
    out_path = Path(output_txt_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_chars: Set[str] = set()
    with open(out_path, "w", encoding="utf-8") as out_f:
        for m_path in manifest_paths:
            with open(m_path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    text = record.get("text", "").strip()
                    if text:
                        out_f.write(text + "\n")
                        seen_chars.update(text)

        required_charset = build_required_ptbr_charset()
        missing = [ch for ch in required_charset if ch not in seen_chars]
        if missing:
            out_f.write(" ".join(missing) + "\n")

    return out_path


def train_sentencepiece_bpe_ptbr(
    corpus_txt_path: str,
    output_dir: str,
    config: Optional[TokenizerConfig] = None,
) -> Path:
    """
    Treina o tokenizador SentencePiece BPE de 1.024 tokens preservando acentos,
    maiúsculas e pontuação sem sofrer colisão de vírgula no parser C++ (Spec 03).
    """
    import sentencepiece as spm

    cfg = config or TokenizerConfig()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = str(out_dir / "tokenizer")

    required_chars = build_required_ptbr_charset()
    corpus_file = Path(corpus_txt_path)
    corpus_text = corpus_file.read_text(encoding="utf-8")
    missing_in_corpus = [ch for ch in required_chars if ch not in corpus_text]
    if missing_in_corpus:
        with open(corpus_file, "a", encoding="utf-8") as cf:
            cf.write("\n" + " ".join(missing_in_corpus) + "\n")

    # Nunca passe "," em string separada por vírgulas para user_defined_symbols no SentencePiece
    non_comma_punct = [p for p in sorted(REQUIRED_PUNCTUATION) if p != ","]

    spm.SentencePieceTrainer.train(
        input=str(corpus_file),
        model_prefix=model_prefix,
        vocab_size=cfg.vocab_size,
        model_type=cfg.model_type,
        character_coverage=cfg.character_coverage,
        normalization_rule_name="identity" if cfg.preserve_case_and_punctuation else "nmt_nfkc_cf",
        required_chars=required_chars,
        user_defined_symbols=non_comma_punct,
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
    )

    vocab_file = out_dir / "tokenizer.vocab"
    with open(vocab_file, "r", encoding="utf-8") as vf:
        tokens = [line.split("\t")[0] for line in vf if line.strip()]

    validate_ptbr_vocabulary_tokens(tokens)

    # Salva também vocab.txt no formato esperado pelo change_vocabulary() do NeMo
    nemo_vocab_txt = out_dir / "vocab.txt"
    with open(nemo_vocab_txt, "w", encoding="utf-8") as nvf:
        for tok in tokens:
            nvf.write(tok + "\n")

    return out_dir


def apply_tokenizer_to_parakeet_model(
    pretrained_model_name: str,
    tokenizer_dir: str,
    output_nemo_path: Optional[str] = None,
):
    """
    Carrega o nvidia/multitalker-parakeet-streaming-0.6b-v1 e substitui o tokenizador
    pelo BPE PT-BR de 1.024 tokens preservando FastConformer e Speaker Kernels (Spec 03).
    """
    import nemo.collections.asr as nemo_asr

    asr_model = nemo_asr.models.ASRModel.from_pretrained(pretrained_model_name)
    asr_model.change_vocabulary(
        new_tokenizer_dir=tokenizer_dir,
        new_tokenizer_type="bpe",
    )
    if output_nemo_path:
        Path(output_nemo_path).parent.mkdir(parents=True, exist_ok=True)
        asr_model.save_to(output_nemo_path)
    return asr_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Construtor do Tokenizador BPE PT-BR (1024 tokens) - Spec 03.")
    parser.add_argument("--manifests", nargs="+", required=True, help="Caminhos dos manifestos JSONL em PT-BR.")
    parser.add_argument("--output_dir", required=True, help="Diretório onde o tokenizador será salvo.")
    parser.add_argument("--update_state", action="store_true", help="Atualiza state/pipeline_state.json e USER.md.")
    args = parser.parse_args()

    step_id = "step_02_tokenizer_ptbr"
    if args.update_state:
        from src.state import start_step

        start_step(
            step_id=step_id,
            command=f"python -m src.tokenizer --manifests {' '.join(args.manifests)} --output_dir {args.output_dir}",
            notes="Treinando tokenizador SentencePiece BPE PT-BR (1.024 tokens).",
        )

    try:
        corpus_path = Path(args.output_dir) / "corpus_ptbr.txt"
        extract_corpus_from_manifests(args.manifests, str(corpus_path))
        out_dir = train_sentencepiece_bpe_ptbr(str(corpus_path), args.output_dir)
        print(f"[Spec 03] Tokenizador BPE PT-BR (1.024 tokens) validado e salvo em: {out_dir}")

        if args.update_state:
            from src.state import complete_step

            complete_step(
                step_id=step_id,
                next_step_id="step_03_mixer_multitalker",
                next_action="Gerar o dataset sintético sobreposto (70% multitalker + 30% limpo) com máscaras de 80ms.",
                next_command=(
                    f".venv/bin/python -m src.mixer --clean_manifest {args.manifests[0]} "
                    "--output_dir data/multitalker_train "
                    "--output_manifest data/multitalker_train/multitalker_train_manifest.json "
                    "--num_mixtures 1000 --clean_ratio 0.30 --update_state"
                ),
                last_summary=f"Tokenizador SentencePiece BPE PT-BR (1.024 tokens) treinado e validado em {out_dir}.",
                artifacts={"tokenizer_dir": str(out_dir)},
                metrics={"vocab_size": REQUIRED_VOCAB_SIZE},
            )
    except Exception as exc:
        if args.update_state:
            from src.state import fail_step

            fail_step(step_id=step_id, error_message=str(exc))
        raise


if __name__ == "__main__":
    main()
