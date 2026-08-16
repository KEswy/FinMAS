# FinMAS v5 Scope Decisions

This document records non-obvious v5 scope decisions so future work does not
silently reintroduce abandoned components.

## 1. LLM is a factor extractor, not a final direction oracle

`MechanismExtractor` returns structured factors and evidence. Final direction is
produced by calibrated fusion. This prevents LLM confidence from being treated
as a calibrated probability.

## 2. Debate and RelDecomp are legacy-only for now

`Debate` and `RelDecomp` exist in the legacy root modules but are not part of
the v5 evaluator. They were designed as causal experiments, not stable
production factors.

Reason to defer:

- They add multiple LLM calls per event and materially increase cost/latency.
- Prior legacy results did not show a robust, significant directional gain.
- The v5 final ensemble already combines no-LLM and multi-seed LLM factors.

Re-enable only if a dedicated experiment demonstrates a positive, significant
out-of-sample contribution over the current final ensemble.

## 3. Neo4j remains optional

The v5 KG is in-memory first. Neo4j is not required for the current research
reproducibility target. Add a real Neo4j backend only when operational graph
storage becomes a requirement.

## 4. Risk scale values are horizon-specific and currently manual

The event-level VaR backtest uses horizon-specific scale values:

- T+1: `0.4`
- T+5: `-0.4`
- T+20: `-0.4`

These were selected from observed calibration. A future out-of-fold calibration
procedure should replace these manual values.

