# Dataset Card: idibo-sft-en

## Summary

- **Run timestamp**: 2026-11-12T12:28:23.389543Z
- **Wall clock time**: 2.5s
- **Pipeline config hash**: `8c47eb58f0a4b752`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 374 | 374 | — | 0 | 374 |
| SchemaGate | 374 | 374 | 0 | 0 | 374 |
| ExactDeduplicator | 374 | 374 | — | — | 374 |
| MinHashDeduplicator | 374 | 368 | — | — | 368 |
| TextCleaner | 368 | 368 | — | — | 368 |
| MessagesExporter | — | 368 | — | — | 368 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 374/0 passed (N/A), 0 rejected
- **SchemaGate**: 374/374 passed (100.0%), 0 rejected


**Rejection reasons:**
  - None

## Deduplication

  - near_duplicates_removed: 6
  - surviving_count: 368

## Known Limitations

- MinHash threshold used: **0.8** — this is a starting point, not a universal answer. Adjust per domain (legal, medical, and other domain-specific corpora may need a lower threshold to avoid losing structurally similar but semantically valid examples).
- Table extraction requires `extract_tables: true` on the PDF reader and MinerU with GPU support. Disabled by default.
- Token counting uses a whitespace tokenizer by default. For exact LLM context window
  enforcement, enable `use_tiktoken: true` in the SchemaGate config.
- Both within-run (MinHash/Diversity) and cross-run (EmbeddingDeduplicator) deduplication
  are available. Cross-run dedup persists an embedding index across runs; use
  `embedding_reset_index: true` to start fresh.

## Reproduction

```bash
pip install curatorkit
curatorkit run pipeline.yaml --output-dir ./out/
```

## Tool Versions

- **curatorkit**: 0.1.21
- **python**: 3.12.14
