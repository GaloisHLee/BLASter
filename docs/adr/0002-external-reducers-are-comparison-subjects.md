# ADR-0002: Keep external reducers outside the runtime backend model

- Status: Accepted
- Date: 2026-08-08

## Context

Flatter, fpylll, fplll, and BLASter have different reduction algorithms,
precision policies, stopping conditions, parameters, and output surfaces.
Treating them as interchangeable arithmetic backends would conflate algorithm
selection with integer-width handling and would erase BLASter's identity.

## Decision

Flatter and fpylll/fplll are external research comparisons. They may be used to
generate benchmark samples, compare quality/time distributions, exercise file
interoperability, and provide independently produced test cases. They are not:

- runtime fallbacks for overflow or precision exhaustion;
- implementations of FastInt64, CheckedInt64, or Multiprecision;
- specification truth for a BLASter postcondition; or
- automatic preprocessors/postprocessors in the BLASter API.

An experiment may explicitly run more than one tool, but its result must name
each algorithm and must not present the composition as one BLASter reduction.

## Consequences

- Integer promotion stays inside the BLASter arithmetic layer.
- Benchmark reports compare time and quality separately on a declared common
  input corpus; they do not produce universal routing rules.
- The independent BLASter verifier checks the BLASter specification rather than
  equality with another reducer's output.

## Evidence

The existing measured corpus is retained in
[`benchmark/FLATTER_VS_BLASTER.md`](../../benchmark/FLATTER_VS_BLASTER.md).
Current algorithm roles are summarized in
[`docs/spec/current-behavior-baseline.md`](../spec/current-behavior-baseline.md).
