# Development Timeline

> This document summarizes Ariadne's development and experiment themes in chronological order,
> for a quick overview of how the project evolved.
>
> Process planning notes and experiment reports are **not published** with this repository; the tables
> below are a topic-level index and summary — they describe *what was done and which problem it solved*,
> without internal evaluation criteria or collaboration details.

**Language**: [中文](TIMELINE.md) · [English](TIMELINE.en.md)

---

## 2026-08

### 08-15 ~ 08-22 · Pipeline baseline and evaluation groundwork

| Date | Topic |
| --- | --- |
| 08-15 | MCP pipeline issue list and fixes; in-place vector update fix; StoryRank refactor |
| 08-18 | Self-built personal-memory dataset evaluation plan; LiFEMem full experiment & data reports; worst-query deep dive |
| 08-19 | Mem0 baseline integration |
| 08-21 | Graphiti baseline integration |
| 08-22 | DBA extraction-stage evaluation |

**Outcome**: cleared known defects in the MCP and visualization pipelines, and established a reproducible memory
evaluation flow with baseline comparisons (Mem0 / Graphiti).

### 08-25 ~ 08-31 · Retrieval fairness, temporal pipeline, datasets

| Date | Topic |
| --- | --- |
| 08-25 | Retrieval-comparison fairness check and StoryRank impact quantification; intro website |
| 08-27 | TEMPORAL relation experiments and pipeline completion; temporal feature summary |
| 08-28 | Agent mixed-memory / temporal dialogue dataset construction (plan + validation report); PAR retrieval paper |
| 08-31 | e2e evaluation flow redesign |

**Outcome**: quantified StoryRank's effect on retrieval, completed the temporal (TEMPORAL) pipeline, and built a
mixed-memory dataset that reflects realistic usage.

---

## 2026-09

### 09-01 ~ 09-12 · End-to-end evaluation and pipeline fixes

| Date | Topic |
| --- | --- |
| 09-01 | e2e comprehensive evaluation |
| 09-07 | LongMemEval probe evaluation; StoryRank ablation re-test |
| 09-08 | e2e experiment paper |
| 09-11 | Person-attribution and temporal pipeline fixes; memory-fidelity plan (Source layer and rationale) |
| 09-12 | Inspection automation · memory-fidelity e2e · accuracy report |

**Outcome**: located failure modes on a long-horizon benchmark (LongMemEval), fixed person attribution and the
temporal pipeline, and added the "Source layer" for memory fidelity.

### 09-10 ~ 09-19 · ALM service: integration and self-hosting

| Date | Topic |
| --- | --- |
| 09-10 | ALM integration plan (rolling ledger) |
| 09-15 | ALM load-test report |
| 09-19 | ALM self-hosting deployment guide |

**Outcome**: added the top-level `alm/` service package (`config` / `contract` / `engine` / `space` / `server` /
`smoke` / `judge` / `cleanup` / `loadtest` / `tokens`) plus `Dockerfile.alm` / `docker-compose.alm.yml`, and
completed self-hosted deployment and load testing.

### 09-20 ~ 09-27 · ALM service: system analysis and tuning

| Date | Topic |
| --- | --- |
| 09-20 | ALM system analysis and rework plan |
| 09-21 | Thinking-mode A/B experiment |
| 09-22 | Edge-type ablation experiment |
| 09-25 | Retrieval budget and output-size cap experiments |
| 09-27 | Full-pipeline review and historical-log findings |

**Outcome**: disabled thinking mode for internal pipeline LLM calls to cut cost and latency; completed edge-type
ablation and retrieval-budget calibration; ran a full-pipeline review and fixed a batch of long-standing issues.

---

## 2026-10

### 10-01 · Write-path cost optimization and dual instances

| Date | Topic |
| --- | --- |
| 10-01 | Write-path cost optimization and model migration plan; dual-instance switch and StoryRank removal plan |
| 10-01 | Visualization panel code review and fixes; panel graph-library vs. MCP graph switching |

**Outcome**: write-path cost optimization and model migration; StoryRank retired; switching between the panel's
graph library and the MCP graph.

### 10-12 ~ 10-13 · Stability hardening, rerank integration, open-source cleanup

- **Outbound quota throttling**: a **character-rate** token bucket (waiting-style + AIMD adaptation) covering the
  embedding / LLM / rerank outbound paths.
- **Four retrieval-stability measures**: TopK capping, graph-build wait window, independent backfill batches, and
  backfill progress counted by original messages.
- **Cross-encoder rerank integration**: applies only to the pure-vector channel; failures **degrade silently** back
  to cosine order.
- **Upstream quota measurement**: measured the rerank upstream rate-limit scheme and the "characters → tokens"
  ratio, then calibrated the throttle ceiling accordingly.
- **Ops and open-source cleanup**: confirmed steady-state container memory (LRU eviction/reload without
  re-embedding); cleaned comments and docs for open source; moved internal planning and experiment reports out of
  the repository.

---

## Notes

- Dates are taken from the file markers and records of each stage's report; a single date may cover multiple
  outputs.
- For deeper technical details, see the repository's root `README.en.md` and the other documents under `docs/`.
