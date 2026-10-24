"""Testes de contrato para o Tokenizador SentencePiece BPE em PT-BR (Spec 03)."""

import pytest
from pydantic import ValidationError

from src.schemas import REQUIRED_VOCAB_SIZE, TokenizerConfig
from src.tokenizer import (
    REQUIRED_PTBR_DIACRITICS,
    REQUIRED_PUNCTUATION,
    build_required_ptbr_charset,
    extract_corpus_from_manifests,
    validate_ptbr_vocabulary_tokens,
)


def test_tokenizer_config_enforces_1024_vocab():
    cfg = TokenizerConfig()
    assert cfg.vocab_size == REQUIRED_VOCAB_SIZE

    with pytest.raises(ValidationError, match="vocab_size == 1024"):
        TokenizerConfig(vocab_size=4096)


def test_validate_ptbr_vocabulary_tokens_accepts_valid_vocab():
    base_tokens = list(REQUIRED_PTBR_DIACRITICS) + list(REQUIRED_PUNCTUATION)
    filler = [f"tok_{i}" for i in range(REQUIRED_VOCAB_SIZE - len(base_tokens))]
    vocab = base_tokens + filler
    assert validate_ptbr_vocabulary_tokens(vocab) is True


def test_validate_ptbr_vocabulary_rejects_missing_diacritics():
    # Vocabulário de 1.024 tokens em inglês sem caracteres acentuados do PT-BR
    english_only_vocab = [
        f"word_{i}" for i in range(REQUIRED_VOCAB_SIZE - len(REQUIRED_PUNCTUATION))
    ] + list(REQUIRED_PUNCTUATION)
    with pytest.raises(ValueError, match="caracteres acentuados do PT-BR ausentes"):
        validate_ptbr_vocabulary_tokens(english_only_vocab)


def test_validate_ptbr_vocabulary_rejects_missing_punctuation():
    no_punct_vocab = list(REQUIRED_PTBR_DIACRITICS) + [
        f"sub_{i}" for i in range(REQUIRED_VOCAB_SIZE - len(REQUIRED_PTBR_DIACRITICS))
    ]
    with pytest.raises(ValueError, match="sinais de pontuação nativos ausentes"):
        validate_ptbr_vocabulary_tokens(no_punct_vocab)


def test_build_required_ptbr_charset_includes_upper_lower_and_hyphen():
    charset = build_required_ptbr_charset()
    for ch in REQUIRED_PTBR_DIACRITICS:
        assert ch in charset
        assert ch.upper() in charset
    for p in REQUIRED_PUNCTUATION:
        assert p in charset
    assert "-" in charset and "," in charset


def test_extract_corpus_from_manifests_seeds_missing_required_chars(tmp_path):
    manifest = tmp_path / "mini_manifest.json"
    manifest.write_text('{"text": "Apenas texto simples sem trema"}\n', encoding="utf-8")
    corpus_txt = tmp_path / "corpus.txt"

    extract_corpus_from_manifests([str(manifest)], str(corpus_txt))
    content = corpus_txt.read_text(encoding="utf-8")
    for ch in build_required_ptbr_charset():
        assert ch in content
