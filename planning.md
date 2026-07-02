# Research Plan — Mathematical Framework for Quantifying Ethical Robustness in Adversarial ML Systems

## Motivation & Novelty Assessment

### Why This Research Matters
ML evaluation in scientific domains (astrophysics included) is dominated by a
single scalar — accuracy — which is provably blind to the *ethical* failure modes
(bias, integrity, transparency) that adversarial prompts induce. If a model can be
made to answer correctly while its ethical alignment silently collapses, an
accuracy-only report certifies it as safe. A mathematically rigorous, *composite*
robustness coefficient that fuses correctness preservation with alignment stability
is therefore a prerequisite for trustworthy scientific ML.

### Gap in Existing Work
From `literature_review.md`: correctness-robustness metrics (CLEVER — Weng 2018;
certified radius — Cohen 2019; TRADES decomposition — Zhang 2019) quantify only
**scientific correctness** under perturbation. Alignment metrics (ASR /
JailbreakBench — Chao 2024) quantify only **ethical alignment**. **No prior work
fuses them into a single coefficient with provable properties.** Likewise, no
theorem connects a *training-time ethical constraint* (CMDP / CPO — Achiam 2017;
Constitutional AI — Bai 2022) to a *lower bound* on such a coefficient.

### Our Novel Contribution
We (1) formalize the composite ethical-robustness coefficient `RC_λ` (Def. 2.9 of
the literature review), (2) prove its basic analytic properties (bounds,
monotonicity, an exact TRADES-style additive decomposition), (3) prove the
**central theorem**: a training-time alignment constraint of CMDP type yields a
lower bound on `E[RC_λ]` and a *provable separation* `Δ = E[RC_λ^{eth}] −
E[RC_λ^{acc}] ≥ 0.15` under explicit, checkable hypotheses, (4) prove that the
composite is **necessary** — any accuracy-only metric is provably blind to
alignment collapse (Tsipras-style hard instance), and (5) give an
attack-independent **Lipschitz/certified lower bound** on `RC_λ`. All results are
verified computationally (symbolic + Monte-Carlo + statistical).

### Experiment (Computational Verification) Justification
- **V1 — Bounds & monotonicity:** confirm Theorem 1 (`RC_λ∈[0,1]`, non-increasing
  in ε) on randomized inputs and nested perturbation sets. *Why:* catches sign /
  clipping errors in the definition.
- **V2 — Exact decomposition:** confirm the identity `E[1−C̄]=E[1−C]+E[C−C̄]`
  (Theorem 2) to machine precision. *Why:* the additive template is the backbone.
- **V3 — Lipschitz certificate:** confirm `RC_λ ≥ 1−ε(λL_C/c_min+(1−λ)L_A/a_min)`
  (Theorem 5) on a Lipschitz synthetic model. *Why:* validates the
  attack-independent bound.
- **V4 — Main separation:** Monte-Carlo estimate of `Δ` over λ with bootstrap 95%
  CI, and check the *theoretical* lower bound of Theorem 3 is respected. *Why:*
  directly tests the hypothesis' `Δ≥0.15` claim and the theorem's constants.
- **V5 — Necessity:** exhibit two models with identical accuracy behavior but
  different `RC_λ` (Theorem 4). *Why:* demonstrates accuracy-only insufficiency.
- **V6 — Certified radius arithmetic:** evaluate Cohen's `R=(σ/2)(Φ⁻¹(p_A)−Φ⁻¹(p_B))`
  to instantiate a fully-certified `RC_λ=1` regime. *Why:* connects abstract metric
  to a concrete computable certificate.
- **V7 — Symbolic:** `sympy` proof-checking of the decomposition identity and the
  `Δ` inequality. *Why:* removes algebra risk from the proofs.

## Research Question
Can the informal hypothesis — "ethically-constrained models are ≥0.15 more robust
than accuracy-only models under adversarial prompts, quantifiable by a composite
metric" — be turned into a *theorem*: a precisely-defined coefficient `RC_λ` with
provable bounds, and explicit sufficient conditions under which a training-time
ethical constraint guarantees `Δ = E[RC_λ^{eth}] − E[RC_λ^{acc}] ≥ 0.15`?

## Hypothesis Decomposition
1. **H1 (well-posedness):** `RC_λ` is a well-defined robustness coefficient
   (bounded in `[0,1]`, monotone in ε, correct ε=0 limit).
2. **H2 (structure):** `RC_λ`'s deficit decomposes additively into correctness- and
   alignment-stability terms (TRADES analogue).
3. **H3 (separation, main):** a CMDP alignment constraint `E[1−Ā]≤d` on the ethical
   model, plus a bounded correctness cost `δ_C`, forces `Δ ≥ (1−λ)[(1−d)−ā/a_min] −
   λδ_C`, hence `≥0.15` for an explicit parameter region.
4. **H4 (necessity):** no function of correctness alone can reproduce `RC_λ`
   (λ<1); accuracy-only evaluation is provably blind to alignment collapse.
5. **H5 (certifiability):** `RC_λ` admits attack-independent Lipschitz and
   randomized-smoothing lower bounds.

Independent variable: training regime (ethical-constrained vs accuracy-only),
weight λ, perturbation radius ε, constraint level d. Dependent variable: `RC_λ`
and `Δ`.

## Proposed Methodology

### Approach
Pure/applied mathematics: definitions → lemmas → theorems, following the TRADES
proof template (exact decomposition + calibrated bound) recommended by the
literature review, with CMDP constraint analysis (CPO) for the separation theorem
and a Tsipras-style hard-instance construction for necessity. Every algebraic step
is cross-checked with `sympy`; every quantitative claim with `numpy`/`scipy`.

### Experimental Steps
1. Fix definitions/notation (`definitions.md`), reusing Def. 2.1–2.9 of the review.
2. Prove Theorem 1 (bounds/monotonicity) — direct.
3. Prove Theorem 2 (exact decomposition + normalized bound) — TRADES template.
4. Prove Theorem 3 (constraint ⇒ separation `Δ≥0.15`) — CMDP expected-cost bound.
5. Prove Theorem 4 (necessity) — explicit two-model construction.
6. Prove Theorem 5 (Lipschitz + certified lower bounds) — mean-value + Cohen.
7. Prove Theorem 6 (admissible-λ trade-off region) — corollary of Thm 3.
8. Verify V1–V7 computationally; save to `results/`.

### Baselines
Comparison is against the existing single-term metrics (correctness-only:
CLEVER/certified radius; alignment-only: `1−ASR`) — our composite must (a) reduce
to each at λ∈{1,0} and (b) strictly dominate them in discriminating power
(Theorem 4).

### Evaluation Metrics
Symbolic identity checks (exact); Monte-Carlo `E[RC_λ]` with bootstrap 95% CI;
respected-lower-bound checks for each theorem; one-sided test that `Δ≥0.15`.

### Statistical Analysis Plan
Monte-Carlo with fixed seed (42), n=20000 inputs; bootstrap (10000 resamples) 95%
CI for `Δ`; one-sided test `H₀: Δ=0.15` vs `H₁: Δ>0.15` at α=0.05. Report effect
size (Cohen's d on per-input `RC_λ` difference where paired).

## Expected Outcomes
Theorems 1–6 proved; V1–V7 confirm every inequality; Monte-Carlo `Δ≈0.28` with CI
strictly above 0.15 in the modeled regime; necessity demonstrated exactly.

## Timeline and Milestones
- Definitions + Thm 1–2: 15 min
- Thm 3–6: 20 min
- Verification scripts V1–V7 + runs: 15 min
- REPORT.md + README.md: 10 min

## Potential Challenges
- **Discrete prompt perturbations** break continuous-Lipschitz certification →
  handle by keeping the perturbation set abstract (any nested family) and flagging
  discrete smoothing as an open problem.
- **Accuracy–robustness coupling** (Tsipras) may make raising ρ_A cost ρ_C →
  Theorem 6 bounds this and gives the admissible λ region.
- **CPO bounds expectations, not a.s. floors** → phrase Theorem 3 entirely in
  expectations (matches `E[1−Ā]≤d`).

## Success Criteria
All six theorems have complete, gap-free proofs; all computational checks pass;
`REPORT.md` states the separation theorem with explicit constants and a verified
parameter region achieving `Δ≥0.15`.
</content>
</invoke>
