from materials_rag.ingestion.hybrid_retrieval import build_bm25_index, rrf_score, tokenize


def test_tokenizer_preserves_technical_identifier() -> None:
    tokens = tokenize("B017-F03 tested at 600 MPa and R=0.1")
    assert "b017-f03" in tokens
    assert "600" in tokens
    assert "mpa" in tokens


def test_bm25_index_is_deterministic() -> None:
    units = [
        {"retrieval_text": "fatigue result B017-F03"},
        {"retrieval_text": "HIP process B021"},
    ]
    first = build_bm25_index(units)
    second = build_bm25_index(units)
    assert first.tokenized_documents == second.tokenized_documents
    assert first.idf == second.idf


def test_rrf_uses_both_ranked_branches() -> None:
    assert rrf_score(1, 2, 60) == (1 / 61) + (1 / 62)
    assert rrf_score(1, None, 60) == 1 / 61

