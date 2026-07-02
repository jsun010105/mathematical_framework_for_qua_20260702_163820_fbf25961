# Literature Review

**Project:** Mathematical Framework for Quantifying Ethical Robustness in Adversarial Astrophysics ML Systems

**Hypothesis under investigation:** ML models trained with explicit ethical
constraints exhibit measurably higher robustness coefficients (≥ 0.15 improvement)
under adversarial prompts than accuracy-only models, and this robustness can be
formally quantified by a *composite metric* combining scientific-correctness
preservation and ethical-alignment stability under adversarial perturbation.

---

## 1. Research Area Overview

The hypothesis sits at the intersection of three mathematically-mature areas:

1. **Adversarial robustness theory** — formal definitions of robustness, min-max
   (saddle-point) robust optimization, and *quantitative* robustness metrics
   (certified radii, Lipschitz-based scores). This supplies the machinery for a
   rigorous "robustness coefficient."
2. **The robustness–accuracy trade-off** — provable tension between clean
   accuracy and adversarial robustness, and additive decompositions of robust
   error. This motivates why a *composite* (multi-term) metric is the right
   object, and gives the decomposition template.
3. **Alignment / constrained training** — RLHF, Constitutional AI, and
   constrained-MDP optimization, which operationalize "explicit ethical
   constraints during training," plus adversarial-prompt threat models (GCG,
   JailbreakBench) that define what "adversarial astrophysics prompts" concretely
   means and how to score alignment stability.

The "astrophysics" framing specializes the *scientific-correctness* term
`C(f,x)` to correctness on astrophysics questions; the mathematics is
domain-agnostic (any measurable correctness score works). No prior work was found
that combines a certified-robustness-style scalar with an alignment-stability
term into a single composite coefficient — this is the gap the project targets.

---

## 2. Key Definitions

Notation: `f` a model/classifier; `x` an input (prompt); `y` a ground-truth label
or reference; `B_p(x, ε) = {x' : ||x'−x||_p ≤ ε}` the adversarial perturbation set
(for LLMs, an abstract set of allowed adversarial prompts, e.g. GCG suffixes).

**Def. 2.1 (Standard / natural error).** `R_nat(f) = E_{(x,y)}[ 1{f(x) ≠ y} ]`.

**Def. 2.2 (Robust / adversarial error).**
`R_rob(f) = E_{(x,y)}[ max_{x' ∈ B_p(x,ε)} 1{f(x') ≠ y} ]`.
Robust accuracy `= 1 − R_rob(f)`.

**Def. 2.3 (Boundary error, TRADES).** With decision boundary
`DB(f) = {x : f(x) = 0}` (binary score `f`, label `y∈{−1,+1}`),
`R_bdy(f) = E[ 1{ X ∈ B(DB(f), ε),  f(X)Y > 0 } ]` —
inputs correctly classified but within `ε` of the decision boundary.

**Def. 2.4 (Minimum adversarial distortion / robustness radius).**
`r(f,x) = min{ ||δ||_p : f(x+δ) ≠ f(x) }`; the *pointwise* robustness of `f` at `x`.
A robustness metric estimates or lower-bounds `r(f,x)`.

**Def. 2.5 (Certified radius, randomized smoothing).** For the smoothed classifier
`g(x) = argmax_c P_{η∼N(0,σ²I)}[ f(x+η) = c ]`, `g` is *certified* robust at `x`
within radius `R` (Def. 2.4 with `p=2`) — see Theorem 3.3.

**Def. 2.6 (CLEVER score).** An attack-independent estimate of a *lower bound* on
`r(f,x)` built from the local Lipschitz constant of class-margin functions
(Theorem 3.2).

**Def. 2.7 (Attack Success Rate, ASR).** For a judge `JUDGE`, target model `LLM`,
and harmful goal `G`: `ASR = (1/n) Σ 1{ JUDGE(LLM(P_i), G_i) = True }` over `n`
adversarial prompts `P_i`. Alignment robustness `= 1 − ASR`. (JailbreakBench.)

**Def. 2.8 (Constrained MDP feasible set).**
`Π_C = { π : J_{C_i}(π) ≤ d_i ∀ i }`, where
`J_{C_i}(π) = E_{τ∼π}[ Σ_t γ^t C_i(s_t,a_t,s_{t+1}) ]` is the expected discounted
cost. Constrained training solves `max_{π∈Π_C} J(π)`. (Achiam et al.)

**Def. 2.9 (Proposed composite ethical-robustness coefficient).** With per-input
correctness `C(f,x)∈[0,1]`, alignment `A(f,x)∈[0,1]`, worst-case values
`C̄ = min_{x'∈B_p(x,ε)} C(f,x')`, `Ā = min_{x'∈B_p(x,ε)} A(f,x')`, define
component robustness `ρ_C = C̄ / max(C, τ)`, `ρ_A = Ā / max(A, τ)` (clipped to
`[0,1]`), and the composite
`RC_λ(f,x,ε) = λ·ρ_C + (1−λ)·ρ_A`, `λ∈[0,1]`.
The hypothesis is `E_x[RC_λ(f_eth,·,ε)] − E_x[RC_λ(f_acc,·,ε)] ≥ 0.15`.
(This project's proposed object; see `code/composite_robustness_metric.py`.)

---

## 3. Key Papers and Prior Results

### Group A — Formal robustness metrics (mathematical backbone)

#### Theorem 3.1 (TRADES: exact decomposition + calibrated upper bound) — Zhang et al. 2019 (arXiv:1901.08573)
**Exact decomposition (equality):**
> `R_rob(f) = R_nat(f) + R_bdy(f)`  (Defs. 2.1–2.3).

**Theorem 3.1 (upper bound).** Under **Assumption 1** (the surrogate `φ` is
*classification-calibrated*, `H⁻(η) > H(η)` for all `η ≠ 1/2`), for any non-negative
`φ` with `φ(0) ≥ 1`, any measurable `f`, any distribution, and any `λ > 0`:
> `R_rob(f) − R*_nat ≤ ψ⁻¹( R_φ(f) − R*_φ ) + E[ max_{X'∈B(X,ε)} φ( f(X')f(X)/λ ) ]`,
where `ψ = ψ̃**` is the convexified Bartlett–Jordan–McAuliffe transform. **Theorem 3.2**
shows this bound is *tight* (matching lower bound up to `ξ>0`) when additionally
`lim_{x→+∞} φ(x) = 0`. The TRADES objective minimizes a surrogate of exactly this
bound: `min_f E[ L(f(X),Y) + (1/λ)·max_{X'∈B(X,ε)} L(f(X), f(X')) ]` — first term
controls natural error (accuracy), second (regularizer) controls the boundary term
(stability).
- **Proof technique:** use the *exact decomposition*, bound the natural-excess term
  via the calibration inequality `R_nat − R*_nat ≤ ψ⁻¹(R_φ − R*_φ)`, and bound the
  boundary term by a surrogate of the "close-to-boundary" indicator.
- **Relevance:** the *additive decomposition* (accuracy term + stability term) is
  the direct template for the composite metric `RC_λ`: correctness preservation
  `ρ_C` plays the role of the natural-accuracy term, alignment stability `ρ_A`
  the role of the boundary/stability term.

#### Theorem 3.2 (CLEVER lower bound via local Lipschitz constant) — Weng et al. 2018 (arXiv:1801.10578)
Let `g` be a classifier with class outputs `g_c`, true class `c`. If, for all
`x' ∈ B_p(x₀, R)`, the function `g_c − g_j` has local Lipschitz constant `L_q^j`
(with `1/p + 1/q = 1`), then the minimum `ℓ_p` distortion to change the prediction
away from `c` satisfies
> `r(g, x₀) ≥ min_{j≠c} (g_c(x₀) − g_j(x₀)) / L_q^j`.
CLEVER estimates `L_q^j` by sampling gradient norms `||∇(g_c−g_j)(x')||_q` over
`x' ∈ B_p(x₀,R)` and fitting a **reverse Weibull distribution** (extreme value
theory) to estimate their supremum; the CLEVER score is the resulting lower bound.
- **Proof technique:** mean-value / Lipschitz-continuity bound on the margin
  function; extreme-value-theory (Fisher–Tippett) estimation of the max gradient
  norm.
- **Relevance:** provides an **attack-independent, scalar** robustness value — the
  archetype for a "robustness coefficient" that can be computed without a specific
  attack, and generalizable from correctness margins to alignment margins.

#### Theorem 3.3 (Certified radius, randomized smoothing) — Cohen et al. 2019 (arXiv:1902.02918)
Let `f` be any base classifier and `g` its Gaussian smoothing (Def. 2.5). Suppose
that at input `x`, the top class `c_A` and runner-up satisfy
`P(f(x+η)=c_A) ≥ p_A ≥ p_B ≥ max_{c≠c_A} P(f(x+η)=c)`. Then
`g(x+δ) = c_A` for all `||δ||_2 < R`, where
> `R = (σ/2)·( Φ⁻¹(p_A) − Φ⁻¹(p_B) )`,
with `Φ` the standard-normal CDF. The bound is tight for `ℓ_2`.
- **Proof technique:** Neyman–Pearson lemma — the worst-case base classifier
  consistent with the class probabilities is a linear half-space, whose smoothed
  decision changes exactly at the stated radius.
- **Relevance:** a *closed-form, certified* robustness value with a probabilistic
  guarantee — a rigorously-defensible term to plug into a composite coefficient
  and a model for how to certify (rather than merely estimate) the `ρ_C`, `ρ_A`
  components.

### Group B — Adversarial-robustness foundations

#### Def./Framework 3.4 (Saddle-point robust optimization) — Madry et al. 2017 (arXiv:1706.06083)
Adversarial training is cast as the min-max problem
> `min_θ ρ(θ),  ρ(θ) = E_{(x,y)∼D}[ max_{δ∈S} L(θ, x+δ, y) ]`,
with perturbation set `S` (typically an `ℓ_∞` ball). The inner max is
(approximately) solved by **projected gradient descent (PGD)**:
`x^{t+1} = Π_{x+S}( x^t + α·sign(∇_x L(θ, x^t, y)) )`, a "universal first-order
adversary" (loss from many random restarts concentrates; gradient steps on the
inner-max value are justified by **Danskin's theorem**). Robustness is reported as
accuracy under PGD (empirically MNIST > 89%, CIFAR-10 ~46% robust accuracy; robust
training demands higher model capacity).
- **Relevance:** gives the canonical *formal definition* of robustness against a
  perturbation set and the min-max objective that any "robustness coefficient"
  quantifies. `RC_λ`'s worst-case `min_{x'∈B}` mirrors the inner maximization.

#### Result 3.5 (Robustness may be at odds with accuracy) — Tsipras et al. 2018 (arXiv:1805.12152)
On a constructed distribution with one strongly-correlated "robust" feature `x₁`
(`x₁ = +y` w.p. `p`, e.g. `p = 0.95`) and `d` weakly-correlated features
`x_i ∼ N(η·y, 1)` with small `η ∝ 1/√d`, there is a **provable trade-off**
(persisting with infinite data): a standard classifier averaging the weak features
attains standard accuracy ≥ 99%, but an `ℓ_∞` adversary with `ε = 2η` flips the mean
of every weak feature, driving the standard classifier's robust accuracy below
chance (≤ 1%). *Any* robust classifier must instead rely on `x₁` alone, capping its
standard accuracy at `p`. Standard and robust accuracy cannot both approach 1.
- **Proof technique:** explicit distribution + adversary construction; the weak
  features become anti-correlated with `y` after an `ε`-perturbation.
- **Relevance:** formal justification that a *single* accuracy number is
  insufficient — a composite metric with separate clean and worst-case terms
  (as in `RC_λ`) is mathematically necessary to capture the trade-off.

#### Def. 3.6 (FGSM; linear view) — Goodfellow et al. 2014 (arXiv:1412.6572)
The **Fast Gradient Sign Method** perturbation is
`η = ε·sign(∇_x J(θ, x, y))`, yielding adversarial example `x + η`. The paper's
"linear explanation": for a linear model `w^T(x+η) = w^Tx + ε·||w||_1`, the
perturbation's effect grows with input dimension, so high-dimensional models are
generically vulnerable; adversarial examples arise from *too much linearity*, not
overfitting.
- **Relevance:** foundational, cheap adversarial construction; the `sign`-gradient
  step reappears in PGD (3.4) and motivates gradient-based prompt attacks (GCG).

### Group C — Ethical constraints during training (alignment)

#### Framework 3.7 (RLHF objective) — Ouyang et al. 2022, InstructGPT (arXiv:2203.02155)
Three stages: (1) SFT; (2) reward model `r_θ` trained on human rankings with the
pairwise loss
`−E_{(x,y_w,y_l)}[ log σ( r_θ(x,y_w) − r_θ(x,y_l) ) ]`; (3) PPO against `r_θ` with
a per-token KL penalty. The **PPO-ptx objective**:
> `E_{(x,y)∼π^RL_φ}[ r_θ(x,y) − β·log( π^RL_φ(y|x) / π^SFT(y|x) ) ] + γ·E_{x∼D_pre}[ log π^RL_φ(x) ]`.
`β` controls KL drift from the SFT policy; `γ` mixes in pretraining gradients to
reduce the "alignment tax." 1.3B InstructGPT outputs are preferred to 175B GPT-3.
- **Relevance:** the mechanism for injecting alignment objectives; the KL term is a
  *soft constraint* on drift — a candidate stability regularizer for `ρ_A`.

#### Framework 3.8 (Constitutional AI / RLAIF) — Bai et al. 2022 (arXiv:2212.08073)
Trains harmlessness from a written **constitution** (~16 natural-language
principles) with no human harm labels: (1) **SL-CAI** — self-critique-and-revise
responses using sampled critique/revision principles, then SFT on revisions;
(2) **RL-CAI (RLAIF)** — an AI feedback model, steered by sampled principles,
produces preference labels, a preference model is trained on them, and standard
RLHF/PPO optimizes against it. Yields a **Pareto improvement**: less harmful at a
given helpfulness than HH-RLHF, and non-evasive.
- **Relevance:** the paper that most *explicitly* operationalizes "ethical
  constraints during training" — the constitution is the concrete ethical
  constraint; principle-ensembling is a robustness mechanism.

#### Framework 3.9 (Helpful & Harmless RLHF) — Bai et al. 2022 (arXiv:2204.05862)
Preference-model + PPO RLHF with *separate* helpfulness and harmlessness
(red-teaming) datasets; reward `r_total = r_PM − λ_KL·D_KL(π ‖ π₀)` (`λ_KL≈0.001`).
Documents an approximately **linear relation between PPO reward and √KL(π‖π₀)** and
train-vs-test preference-model divergence under over-optimization (reward hacking).
Explicit **helpfulness–harmlessness tension** (a two-objective analogue of
correctness vs alignment).
- **Relevance:** the `√KL`–reward law and train/test PM divergence give concrete,
  reusable measures of *alignment stability / over-optimization under adversarial
  inputs*.

### Group D — Adversarial prompts & constrained training

#### Method 3.10 (GCG adversarial suffixes) — Zou et al. 2023 (arXiv:2307.15043)
Appends an optimized suffix at indices `I` to coerce an affirmative target
`x*_{n+1:n+H}`; objective `L(x_{1:n}) = −log p(x*_{n+1:n+H} | x_{1:n})`, minimized
over discrete tokens by **Greedy Coordinate Gradient**: rank candidate token
swaps per position by the one-hot embedding gradient `−∇_{e_{x_i}} L`, sample `B`
candidates across all positions, evaluate exactly, keep the best. Universal/
transferable across prompts and models. ASR up to 84% on GPT-3.5/4, 66% PaLM-2.
- **Relevance:** the canonical *optimization-based adversarial-prompt* threat model
  — concretely defines `B_p(x,ε)` for LLMs and argues post-hoc alignment is fragile
  (motivating training-time robustness).

#### Benchmark 3.11 (JailbreakBench) — Chao et al. 2024 (arXiv:2404.01318)
Formalizes jailbreaking as: find prompt `P` s.t. `JUDGE(LLM(P), G) = True`, and
measures **ASR** over 100 standardized harmful behaviors with a fixed judge
(Llama-3-70B, chosen empirically for highest human agreement). Standardizes threat
model, templates, and scoring; hosts attack + defense leaderboards.
- **Relevance:** the standardized, reproducible *robustness metric* (`1 − ASR`)
  for the alignment-stability term.

#### Framework 3.12 (Constrained Policy Optimization) — Achiam et al. 2017 (arXiv:1705.10528)
Solves the CMDP `max_{π∈Π_C} J(π)` (Def. 2.8) via a trust-region update: maximize
the advantage surrogate subject to a *linearized* cost constraint
`J_{C_i}(π_k) + (1/(1−γ))·E[A^{π_k}_{C_i}] ≤ d_i` and KL trust region
`D̄_KL(π‖π_k) ≤ δ`. **Proposition 2** bounds the worst-case per-update violation:
`J_{C_i}(π_{k+1}) ≤ d_i + √(2δ)·γ·ε^{π_{k+1}}_{C_i} / (1−γ)²`.
- **Relevance:** the mathematical template for **constrained training as a
  robustness mechanism** — cast an ethical requirement as an expected-cost
  constraint `J_C(π) ≤ d` with per-iteration satisfaction guarantees.

#### Survey 3.13 (Modern Mathematics of Deep Learning) — Berner et al. 2021 (arXiv:2105.04026)
Provides rigorous scaffolding: ERM/statistical learning theory; **Rademacher-
complexity** generalization bounds tightened via weight-matrix spectral norms
combined with a **margin** `M(f,z) = y·f(x)` (larger margin ⇒ better
generalization); and the **flatness–robustness** link (robustness of loss to
parameter perturbations correlates with generalization).
- **Relevance:** margin/Rademacher bounds and the flatness–robustness connection
  give formal footing to metrics linking training-time regularity to
  adversarial-prompt resilience.

---

## 4. Prerequisite Theorems (citable building blocks)

| Result | Source | Statement (summary) | Use in our work |
|--------|--------|---------------------|-----------------|
| Robust-error decomposition | Zhang 2019 | exact: `R_rob = R_nat + R_bdy`; calibrated bound (Thm 3.1) | Template + rigor for additive composite metric |
| Certified radius | Cohen 2019 | `R = (σ/2)(Φ⁻¹(p_A) − Φ⁻¹(p_B))` | Closed-form certified `ρ` component |
| Lipschitz margin bound | Weng 2018 | `r ≥ min_j (g_c−g_j)/L_q^j` | Attack-independent robustness coefficient |
| Saddle-point robustness | Madry 2017 | `min_θ E[max_δ L]` | Formal def. of worst-case (the `min_{x'∈B}`) |
| Accuracy–robustness trade-off | Tsipras 2018 | Provable trade-off on constructed dist. | Justifies multi-term (composite) metric |
| CMDP violation bound | Achiam 2017 | `J_C(π_{k+1}) ≤ d + O(√δ)` | Constrained-training guarantee |
| Margin generalization | Berner 2021 | Rademacher/margin bound | Links regularity ↔ robustness |

---

## 5. Proof Techniques in the Literature

- **Classification-calibrated surrogate-loss analysis** (TRADES): relate
  0-1 robust error to a differentiable surrogate; decompose into interpretable
  terms. → Directly reusable to prove properties of `RC_λ`.
- **Neyman–Pearson / worst-case linear classifier** (randomized smoothing): to get
  a *tight* certificate, identify the adversary-worst base classifier. → Template
  for certifying `ρ_C`, `ρ_A` lower bounds.
- **Extreme value theory (reverse Weibull)** (CLEVER): estimate a supremum
  (Lipschitz constant) from samples. → Estimator for worst-case terms without an
  explicit attack.
- **Explicit hard-instance construction** (Tsipras): build a distribution
  witnessing a trade-off / impossibility. → To prove *tightness* or necessity of a
  composite (single number is insufficient).
- **Trust-region linearization + surrogate bounds** (CPO): convert a hard
  constrained problem into a per-step solvable one with guarantees. → For any
  "train with an ethics constraint" theorem.
- **Convex-combination / KL-regularization** (RLHF, HH): soft constraints via KL
  penalties; the `√KL`–reward law. → Stability regularizer for `ρ_A`.

---

## 6. Related Open Problems / Gaps

- **No unified composite metric.** Robustness metrics (CLEVER, certified radius)
  quantify *correctness* robustness; ASR quantifies *alignment* robustness. No
  prior work fuses them into one coefficient with provable properties — the
  central gap this project fills.
- **Certification for discrete prompt perturbations.** Randomized smoothing and
  CLEVER assume continuous `ℓ_p` inputs; adversarial prompts are discrete tokens
  (GCG). Extending certificates to the prompt setting is open.
- **Does a training-time ethical constraint provably raise a robustness
  coefficient?** The hypothesis' `Δ ≥ 0.15` is empirical; a theorem connecting a
  CPO-style constraint `J_C(π) ≤ d` (or a CAI constitution) to a lower bound on
  `E[RC_λ]` is not in the literature.
- **Reward over-optimization vs robustness.** The HH `√KL`–reward law suggests
  alignment stability degrades past an over-optimization threshold; formalizing the
  optimal KL/constraint strength for maximal `RC_λ` is open.

---

## 7. Recommendations for Proof Strategy

- **Recommended approach:** Define `RC_λ` (Def. 2.9) and *prove an additive
  decomposition/bound* in the style of TRADES Theorem 3.1, i.e. bound
  `1 − E[RC_λ]` by a clean-error term plus separate correctness- and
  alignment-boundary terms. This inherits a proven proof pattern.
- **Key lemmas to establish:**
  1. **Well-definedness / bounds:** `RC_λ ∈ [0,1]`, monotone in `ε`
     (worst-case terms shrink as `ε` grows) — straightforward from Def. 2.9.
  2. **Certifiable lower bound** on each component via smoothing (Thm 3.3) or a
     Lipschitz margin (Thm 3.2), giving a *certified* `RC_λ`.
  3. **Constraint ⇒ robustness gain:** model ethical training as a CMDP constraint
     `J_C(π) ≤ d` (Def. 2.8 / CPO) and show it lower-bounds the alignment term `ρ_A`,
     hence `E[RC_λ]`, yielding the `Δ ≥ 0.15` separation under stated assumptions.
- **Potential obstacles:** (i) discrete prompt perturbations break continuous-Lipschitz
  certification — may need a discrete/randomized-substitution smoothing analogue;
  (ii) the Tsipras trade-off warns that raising `ρ_A` may cost `ρ_C` — the theorem
  must bound this coupling (choose `λ` and constraint level `d` to guarantee a net
  gain); (iii) reward over-optimization (HH) can *reduce* robustness, so any bound
  needs a KL/constraint-strength regularity condition.
- **Computational support:** use `code/composite_robustness_metric.py` for
  Monte-Carlo verification of candidate bounds; `sympy` for symbolic checking of
  the decomposition inequality; `scipy.stats.norm` for the `Φ`/`Φ⁻¹`
  certified-radius arithmetic.
