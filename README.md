# Mathematical Framework for Quantifying Ethical Robustness in Adversarial ML Systems

A formal, machine-verified framework that turns the informal hypothesis *"ethically
constrained models are ≥ 0.15 more robust under adversarial prompts, measurable by a
composite metric"* into **theorems**. Domain: Mathematics. Compute: CPU-only.

## Key results (all proved in `REPORT.md`, verified in `src/`)

- **Composite coefficient** `RC_λ(f,x,ε) = λ·ρ_C + (1−λ)·ρ_A`, fusing worst-case
  scientific-correctness preservation `ρ_C` and ethical-alignment stability `ρ_A`.
- **T1 Well-posedness:** `RC_λ ∈ [0,1]`, non-increasing in ε, `=1` at ε=0; recovers
  the correctness-only (λ=1) and alignment-only (λ=0) metrics.
- **T2 Exact decomposition:** `e_rob^S = e_nat^S + b^S` (TRADES analogue), plus a
  normalized deficit bound.
- **T3 (main) Constraint ⇒ separation:** a CMDP ethical constraint `E[1−Ā_eth]≤d`
  gives `Δ ≥ (1−λ)[(1−d)−ā/a_min] − λδ_C`, which is **≥ 0.15** on an explicit
  region (e.g. λ=½, d=¼, ā=0.30, a_min=0.85, δ_C=0.05 ⇒ Δ≥0.1735).
- **T4 Necessity:** accuracy-only metrics are *provably blind* to alignment
  collapse; `RC_λ` (λ<1) separates safe vs collapsed models by `1−λ`.
- **T5 Certified bounds:** attack-independent Lipschitz + randomized-smoothing
  (Cohen) lower bounds on `RC_λ`.
- **T6 Admissible-λ region:** `Δ≥t ⟺ λ ≤ (γ_A−t)/(γ_A+δ_C)`.

**Monte-Carlo (n=20 000, seed 42):** `Δ̂(λ=0.5) = 0.282`, bootstrap 95% CI
**[0.279, 0.285]** (above 0.15), Cohen's **d=5.4**, one-sided test of `Δ=0.15`
rejected (p≈0). Empirical Δ respects the T3 lower bound (0.218).

## Reproduce

```bash
source .venv/bin/activate
python src/symbolic_checks.py     # SymPy: identities/inequalities for T1,T2,T3,T5,T6
python src/verify_theorems.py     # NumPy/SciPy: V1–V6 + bootstrap CI + figures
```
Runs in < 30 s on CPU. Outputs are deterministic (seed 42).

## File structure

```
REPORT.md                 # PRIMARY deliverable: definitions, theorems, full proofs, results
planning.md               # Phase-0/1 plan + Motivation & Novelty
definitions.md            # notation, perturbation-set axioms (P0,P1), score/constraint defs
src/symbolic_checks.py    # symbolic proof-checking  -> results/symbolic_checks.txt
src/verify_theorems.py    # numerical verification    -> results/*, figures/*
code/composite_robustness_metric.py   # reference RC_λ implementation (pre-provided)
results/                  # symbolic_checks.txt, numerical_report.txt, *.json, config.json
figures/                  # fig1 separation-vs-λ, fig2 Lipschitz certificate, fig3 monotonicity
literature_review.md, resources.md, papers/   # pre-gathered resources
```

See **`REPORT.md`** for the complete statements and proofs.
</content>
