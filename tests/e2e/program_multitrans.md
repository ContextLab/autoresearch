# Research Program: Multidimensional Translation

## Goal
Improve the multidimensional translation system — a text-to-text
translator with a runtime alpha parameter that controls semantic
fidelity (alpha=1) vs. sonic match to the source (alpha=0).

Primary metrics (from benchmark):
- Distinct output count: fraction of sentences meeting diversity
  threshold (target: ≥80%)
- Monotonicity: semantic non-decreasing and audio non-increasing
  as alpha varies (target: ≥95% zero violations)
- Smoothness: fraction of alpha steps producing distinct output
  (target: mean ≥0.5)
- Effective alpha range: [alpha_min, alpha_max] should span most
  of [0, 1]

Secondary: semantic gap between source text and translation ≥0.1

## Setup
- Main API: src/multidimensional_translation/translate.py
- Pool generation: src/multidimensional_translation/pool_generator.py
- Reranking: src/multidimensional_translation/reranker_v2.py
- Phonetic processor: src/multidimensional_translation/phonetic_processor.py
- Audio similarity: src/multidimensional_translation/audio_similarity.py
- Acoustic model: src/multidimensional_translation/acoustic_model.py
- Config: src/multidimensional_translation/config.py
- Benchmark: scripts/run_benchmark.py
- Test data: tests/ fixtures (8-item smoke, 75+ full benchmark)
- Environment: conda env with torch, transformers, phonemizer,
  edge-tts, sentence-transformers, etc.

## How the System Works
1. **Pool generation**: 7 strategies produce 50-200+ diverse
   candidates (beam search, ASR mixing, phonetic-semantic
   neighbors, IPA logits processor, etc.)
2. **Scoring**: Each candidate scored on semantic similarity
   (sentence embeddings vs. reference) and audio similarity
   (TTS → XLS-R → DTW)
3. **Alpha selection**: score = alpha * semantic + (1-alpha) * audio,
   pick argmax per alpha value
4. **Rescaling**: Maps user alpha [0,1] to effective range where
   output actually changes

## Constraints
- Preserve the translate() API: translate(text, src, tgt, alpha)
  must continue to work
- Preserve all 5 language pairs (en↔fr, en↔de, en↔es, en↔he, en↔ar)
- Do not modify test fixtures or benchmark sentence data
- All existing tests (pytest) must continue to pass
- Time budget: 10 minutes per experiment (benchmark is expensive
  due to TTS + audio similarity)

## Experimentation
- Each experiment: modify code → commit → run benchmark → evaluate
  → keep/discard
- Run benchmark: python scripts/run_benchmark.py --quick (smoke) or
  --lang-pair en-fr for one pair
- Keep if distinct output count or smoothness improves without
  degrading monotonicity
- Discard if monotonicity degrades or existing tests fail
- Log results to results.tsv:
  commit	distinct_pct	monotonicity	smoothness	status	description

## Research Directions (ordered by expected impact)

1. **Candidate pool diversity**: The #1 bottleneck is generating
   enough diverse candidates that span the semantic–sonic axis.
   - Tune phonetic-semantic neighbor discovery (beta parameter,
     vocabulary size, similarity thresholds)
   - Expand ASR candidate generation (mix ratios, Whisper
     language forcing parameters)
   - Add new generation strategies (back-translation,
     paraphrase-then-phonemize, constraint-based substitution)
   - Increase num_beams and num_return_sequences

2. **Scoring function improvements**: Sharpen the distinction
   between semantic and audio scores.
   - Improve language detection for language_penalty (upgrade
     word-overlap heuristic to fasttext or langdetect)
   - Tune language_penalty value (default 0.15)
   - Experiment with different sentence embedding models
   - Try multi-scale audio similarity (blend XLS-R layers
     4, 8, 12 with tunable weights)

3. **Alpha rescaling**: Improve consistency of alpha behavior
   across sentences and language pairs.
   - Non-linear alpha mapping (sigmoid or power law instead
     of linear rescale)
   - Per-language-pair calibration of effective range
   - Adaptive rescaling based on candidate pool properties

4. **Acoustic model training**: Improve the learned
   text-to-acoustic embedding (faster decode-time scoring).
   - Contrastive loss temperature tuning (default 0.07)
   - Encoder unfreezing (partial XLM-R fine-tuning)
   - Projection head architecture changes
   - Use acoustic model scores in reranking (blend with
     ground-truth audio similarity)

5. **Phonetic logits processor tuning**: The IPA-based
   processor runs during beam search decoding.
   - Tune top_k (default 200)
   - Experiment with entropy matching temperature
   - Try different IPA distance metrics (weighted by
     phonetic features vs. Levenshtein)

6. **Reranker architecture**: Currently uses simple
   alpha-weighted linear combination.
   - Try learned reranking (ridge regression on features)
   - Add additional features: length ratio, proper noun
     preservation, POS-tag overlap
   - Upper-envelope monotonicity enforcement refinements

7. **Per-language optimization**: Different language pairs may
   benefit from different parameters.
   - Tune beta, language_penalty, num_beams per pair
   - Hebrew/Arabic may need special phonetic handling

8. **Speed optimization**: Reduce benchmark time to allow
   more experiments per hour.
   - Cache TTS audio across experiments
   - Batch audio similarity computation
   - Precompute sentence embeddings

## Output Format
After each benchmark run, extract:
- distinct_pct: fraction meeting diversity threshold
- monotonicity: fraction with zero violations
- smoothness: mean smoothness across sentences
- effective_range: mean [alpha_max - alpha_min]

Log to results.tsv (tab-separated):
commit	distinct_pct	monotonicity	smoothness	status	description
