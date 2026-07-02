# Definitions and Notation

All definitions are self-contained; where a construction is reused from the
literature it is cited. Numbering matches `literature_review.md` §2 where possible.

## Basic objects
- `f` — a model (classifier / LLM). `x` — an input (prompt) drawn from a
  distribution `D` on an input space `X`, with a reference/label used by the
  scoring functions below.
- `B(x, ε) ⊆ X` — the **adversarial perturbation set** of radius `ε ≥ 0` around
  `x` (an `ℓ_p` ball for continuous inputs; an abstract set of allowed adversarial
  prompts — e.g. GCG suffixes — for LLMs). We assume throughout:
  - **(P0, reflexivity)** `x ∈ B(x, 0)` and `B(x,0) = {x}`.
  - **(P1, nesting)** `ε₁ ≤ ε₂ ⇒ B(x, ε₁) ⊆ B(x, ε₂)`.
  Both hold for `ℓ_p` balls and for standard prompt families (suffix length, edit
  distance, token-substitution radius).

## Scores
- `C(f, x) ∈ [0,1]` — **scientific-correctness score** of `f` on clean input `x`
  (e.g. correctness on an astrophysics question; domain-agnostic: any measurable
  correctness works).
- `A(f, x) ∈ [0,1]` — **ethical-alignment score** of `f` on `x` (aggregate over
  the 5 dimensions bias/reproducibility/transparency/integrity/privacy;
  `A = 1 − ASR`-style, cf. JailbreakBench, Def. 2.7).
- Worst-case (adversarial) scores:
  - `C̄(f,x,ε) = min_{x'∈B(x,ε)} C(f,x')`  (**worst-case correctness**)
  - `Ā(f,x,ε) = min_{x'∈B(x,ε)} A(f,x')`  (**worst-case alignment**)
  Minima are assumed attained (finite/compact `B`, or replace `min` by `inf`).
  By **(P0)**, `C̄(f,x,0)=C(f,x)` and `Ā(f,x,0)=A(f,x)`; and always
  `C̄ ≤ C`, `Ā ≤ A` (since `x∈B(x,ε)` by (P0)+(P1)).

## Component and composite robustness (Def. 2.9)
Fix a numerical floor `τ ∈ (0,1)` (implementation `τ=10⁻⁶`).
- **Component robustness (preservation ratios), clipped to `[0,1]`:**
  - `ρ_C(f,x,ε) = clip( C̄(f,x,ε) / max(C(f,x), τ), 0, 1 )`
  - `ρ_A(f,x,ε) = clip( Ā(f,x,ε) / max(A(f,x), τ), 0, 1 )`
- **Composite ethical-robustness coefficient** (convex combination, `λ∈[0,1]`):
  - `RC_λ(f,x,ε) = λ·ρ_C(f,x,ε) + (1−λ)·ρ_A(f,x,ε)`.
- **Expected coefficient:** `RC_λ(f,ε) := E_{x∼D}[ RC_λ(f,x,ε) ]`.
- **Separation:** `Δ(ε,λ) := RC_λ(f_eth,ε) − RC_λ(f_acc,ε)` for an
  ethically-constrained model `f_eth` and an accuracy-only model `f_acc`.

## Decomposition quantities (TRADES analogue, Def. 2.1–2.3)
For a score `S∈{C,A}`:
- **natural deficit** `e_nat^S(f) = E[1 − S(f,x)]`
- **robust deficit** `e_rob^S(f) = E[1 − S̄(f,x,ε)]`
- **stability (boundary) gap** `b^S(f) = E[ S(f,x) − S̄(f,x,ε) ] ≥ 0`.

## Regularity constants
- `c_min, a_min ∈ (0,1]` — a.s. lower bounds on clean scores:
  `C(f,x) ≥ c_min`, `A(f,x) ≥ a_min` for `x∼D` (both models). "Both models are
  competent on clean inputs."
- `L_C, L_A ≥ 0` — Lipschitz constants of `C(f,·)`, `A(f,·)` on `B(x,ε)` w.r.t. the
  metric defining `B` (used for the attack-independent bound, Thm 5; cf. CLEVER).

## Ethical-constraint model (CMDP / CPO, Def. 2.8)
Model ethical training as solving a constrained problem with **alignment cost**
`ℓ_A(f,x) = 1 − Ā(f,x,ε) ∈ [0,1]`. The ethical model satisfies the expected-cost
constraint
- **(EC)** `E_x[ ℓ_A(f_eth,x) ] = E[1 − Ā(f_eth,·,ε)] ≤ d`,   i.e. `E[Ā_eth] ≥ 1−d`,
with feasibility guaranteed (up to `O(√δ)`) by CPO's per-update violation bound
(Achiam 2017, Prop. 2). The accuracy-only model has **no** such constraint; its
alignment may collapse: **(CO)** `E[Ā(f_acc,·,ε)] ≤ ā` with `ā` small.

## Symbols summary
`λ` composite weight · `ε` radius · `τ` floor · `d` constraint level ·
`c_min,a_min` clean floors · `ā` collapsed adversarial alignment · `L_C,L_A`
Lipschitz constants · `δ_C` correctness-robustness gap between the two models ·
`Φ,Φ⁻¹` standard-normal CDF/quantile · `σ` smoothing noise · `p_A,p_B` smoothed
top/runner-up class probabilities.
</content>
