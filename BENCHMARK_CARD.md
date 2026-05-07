# TokenCapBench Benchmark Card

## Overview

TokenCapBench evaluates pre-execution forecasts of verified task success under imposed generated-token caps. For each task and budget grid, a model first forecasts the probability that a fresh capped solve will pass. The harness then runs independent solver calls under each cap, verifies the outputs, and reports calibration, ranking, and fixed-budget allocation metrics.

## Intended Use

- Study whether language models can forecast their own probability of passing a verifier before spending inference budget.
- Compare forecasting and calibration methods under a fixed generated-token budget.
- Evaluate scheduling policies that select task-budget pairs without exceeding a global token cap.
- Reproduce the paper tables, figures, and validation checks from frozen artifacts.

## Out-of-Scope Use

- TokenCapBench is not a full production-agent benchmark.
- It is not a replacement for official SWE-bench, BigCodeBench, LiveCodeBench, EvalPlus, or other upstream benchmark leaderboards.
- Forecasts should not be treated as guarantees of task success or as an automated policy for allocating user access.
- The first release does not claim to measure hidden provider compute that is not exposed through API token accounting.

## Evidence Scope

| Role | Track | Verification | Release role |
|---|---|---|---|
| Core | GSM8K + MATH/MATH-500 | numeric and symbolic answer checks | main math evidence |
| Core | HumanEval+ + MBPP+ | EvalPlus | main standalone coding evidence |
| Extension | BigCodeBench-Hard | official BigCodeBench package labels | harder standalone coding extension |
| Extension | CanItEdit | provided tests | code-editing bridge |
| Appendix | LiveCodeBench-300 | official LiveCodeBench labels | freshness check |
| Appendix | Prompt variants | frozen verifier outcomes | forecast stability check |

The release deliberately separates core claims from extensions. CanItEdit uses provided tests, LiveCodeBench is a scoped freshness check, and Docker-heavy repository-repair tracks are treated as bridge or future-work substrates rather than main evidence.

## Protocol

1. Prepare task records with a fixed budget grid.
2. Ask the model for a success-probability curve before any budgeted solve is run.
3. Run fresh solver calls under each generated-token cap.
4. Verify each solver output with the task verifier or official harness bridge.
5. Score probability quality and decision quality from the frozen forecast and outcome rows.

Generated-token caps are the controlled intervention. Prompt tokens, completion tokens, total visible tokens, provider-exposed reasoning-token fields, finish reasons, cap-hit flags, retry counts, and wall-clock timings are logged as diagnostics when available.

## Limitations

- Public benchmark tasks can be contaminated by pretraining or prior benchmark exposure.
- Verifier labels inherit the coverage and failure modes of their test suites or answer extractors.
- Provider token accounting and hidden reasoning-token reporting are not uniform across providers.
- Timing measurements vary with provider load, batching, hardware, and retry behavior, so timing is diagnostic rather than a main claim in this release.
- Forecast curves may be useful for allocation while still being miscalibrated as probabilities.

## Responsible Release Notes

The repository releases benchmark code, task snapshots, forecasts, outcomes, metrics, tables, figures, and metadata. It does not release a model, private user data, or human-subjects data. Release validation includes artifact manifests, leakage checks, and broad text/secret scans. Any new public release should rerun the validation workflow after regenerating artifacts.

