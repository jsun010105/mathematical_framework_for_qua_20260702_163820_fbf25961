# Resources Catalog

## Summary
Resources gathered for **"Mathematical Framework for Quantifying Ethical Robustness
in Adversarial Astrophysics ML Systems."** The project is formal/analytical: the
goal is to define and prove properties of a *composite robustness coefficient*
combining scientific-correctness preservation and ethical-alignment stability under
adversarial perturbation. Resources therefore center on (a) formal robustness-metric
theory, (b) the robustness–accuracy trade-off, and (c) constrained/aligned training.

## Papers
Total papers downloaded: **13** (all as arXiv PDFs in `papers/`; see
`papers/README.md` for full descriptions).

| # | Title | Authors | Year | File | Key result used |
|---|-------|---------|------|------|-----------------|
| 1 | Theoretically Principled Trade-off (TRADES) | Zhang et al. | 2019 | `papers/1901.08573_*.pdf` | Exact decomposition `R_rob=R_nat+R_bdy`; calibrated upper bound (Thm 3.1) |
| 2 | Evaluating Robustness via Extreme Value Theory (CLEVER) | Weng et al. | 2018 | `papers/1801.10578_*.pdf` | Margin/Lipschitz lower bound `r ≥ min_j (f_c−f_j)/L_q^j`; EVT estimator |
| 3 | Certified Robustness via Randomized Smoothing | Cohen et al. | 2019 | `papers/1902.02918_*.pdf` | Certified radius `R=(σ/2)(Φ⁻¹(p_A)−Φ⁻¹(p_B))` (Neyman–Pearson, tight) |
| 4 | Towards DL Models Resistant to Adversarial Attacks | Madry et al. | 2017 | `papers/1706.06083_*.pdf` | Saddle-point robust optimization `min_θ E[max_δ L]`; PGD |
| 5 | Robustness May Be at Odds with Accuracy | Tsipras et al. | 2018 | `papers/1805.12152_*.pdf` | Provable standard-vs-robust accuracy trade-off (constructed dist.) |
| 6 | Explaining and Harnessing Adversarial Examples | Goodfellow et al. | 2014 | `papers/1412.6572_*.pdf` | FGSM `η=ε·sign(∇_x J)`; linear-vulnerability view |
| 7 | Constitutional AI: Harmlessness from AI Feedback | Bai et al. | 2022 | `papers/2212.08073_*.pdf` | SL-CAI + RL-CAI (RLAIF); explicit "constitution" constraint |
| 8 | Training LMs to Follow Instructions (InstructGPT) | Ouyang et al. | 2022 | `papers/2203.02155_*.pdf` | RLHF objective (reward + KL penalty) |
| 9 | Training a Helpful and Harmless Assistant (RLHF) | Bai et al. | 2022 | `papers/2204.05862_*.pdf` | `√KL`–reward law; helpful/harmless tension |
| 10 | Universal & Transferable Adversarial Attacks (GCG) | Zou et al. | 2023 | `papers/2307.15043_*.pdf` | Adversarial-suffix objective; Greedy Coordinate Gradient |
| 11 | JailbreakBench | Chao et al. | 2024 | `papers/2404.01318_*.pdf` | Standardized ASR robustness metric (fixed judge) |
| 12 | Constrained Policy Optimization (CPO) | Achiam et al. | 2017 | `papers/1705.10528_*.pdf` | CMDP `max_{π∈Π_C} J(π)`; per-update violation bound (Prop. 2) |
| 13 | The Modern Mathematics of Deep Learning | Berner et al. | 2021 | `papers/2105.04026_*.pdf` | Rademacher/margin generalization bounds; flatness–robustness link |

## Prior Results Catalog (citable building blocks)

| Result | Source | Statement summary | Used for |
|--------|--------|-------------------|----------|
| Exact robust-error decomposition | Zhang 2019 | `R_rob = R_nat + R_bdy` | Additive template for composite metric |
| TRADES calibrated bound (Thm 3.1) | Zhang 2019 | `R_rob − R*_nat ≤ ψ⁻¹(R_φ−R*_φ) + E max φ(f(X')f(X)/λ)` | Rigorous bound to adapt |
| CLEVER Lipschitz margin (Thm 3.2) | Weng 2018 | `r ≥ min_j (f_c−f_j)/L_q^j` | Attack-independent robustness coefficient |
| Certified radius (Thm 1) | Cohen 2019 | `R=(σ/2)(Φ⁻¹(p_A)−Φ⁻¹(p_B))`, no assumptions on `f` | Certified component of metric |
| Saddle-point robustness | Madry 2017 | `min_θ E[max_{δ∈S} L(θ,x+δ,y)]` | Formal worst-case definition |
| Accuracy–robustness trade-off | Tsipras 2018 | Provable trade-off on constructed distribution | Justifies multi-term metric |
| CMDP violation bound (Prop. 2) | Achiam 2017 | `J_C(π_{k+1}) ≤ d + √(2δ)γε/(1−γ)²` | Constrained-training guarantee |
| Margin/Rademacher bound | Berner 2021 | generalization ≲ weight-norm/margin | Links regularity ↔ robustness |

## Computational Tools

| Tool | Purpose | Location | Notes |
|------|---------|----------|-------|
| SymPy | Symbolic verification of metric identities/bounds | pip pkg (`.venv`) | For proving the decomposition inequality |
| NumPy | Monte-Carlo estimation of `E[RC_λ]` | pip pkg (`.venv`) | Powers `composite_robustness_metric.py` |
| SciPy | `Φ`/`Φ⁻¹`, statistical tests | pip pkg (`.venv`) | Certified-radius arithmetic; significance of `Δ≥0.15` |
| `composite_robustness_metric.py` | Reference impl. of proposed `RC_λ` | `code/` | Includes hypothesis demo |
| `arxiv_search.py` / `batch_search.py` | Literature search helpers | `code/` | arXiv Atom API, paced |
| `download_papers.py` | Paper downloader | `code/` | Curated set, validated PDFs |

No external repositories were cloned: the research spec listed **no user-specified
`code_references`**, and the work is analytical rather than implementation-heavy.

## Resource Gathering Notes

### Search Strategy
The paper-finder service (`localhost:8000`) was **not running**, so literature
review used the arXiv Atom API directly (via `code/arxiv_search.py`, with
retry/backoff and a proper User-Agent — the API rate-limits and requires HTTPS).
Ten topical queries were run (`code/search_results.json`) spanning certified
robustness, Lipschitz/CLEVER metrics, the robustness–accuracy trade-off, LLM
adversarial prompts/jailbreaks, RLHF, Constitutional AI, constrained optimization,
alignment evaluation, ML-for-astronomy, and formal value-alignment.

### Selection Criteria
arXiv's `all:` relevance ranking is noisy, so the final set combines the strong
topical hits with well-established foundational papers (retrieved by known arXiv
IDs) that form the mathematical backbone: robustness metrics (TRADES, CLEVER,
smoothing), robustness foundations (Madry, Tsipras, Goodfellow), alignment
(Constitutional AI, InstructGPT, HH-RLHF), and adversarial-prompt/constrained
training (GCG, JailbreakBench, CPO). Quality was prioritized over quantity.

### Challenges Encountered
- Paper-finder service unavailable → manual arXiv API workflow.
- arXiv API returned 429/301 on `http`; fixed by switching to `https` + UA header + backoff.
- Semantic Scholar API was rate-limited (429) without a key; arXiv sufficed.
- The Read tool's PDF-image rendering lacked `poppler-utils` in subagents; agents
  fell back to `pypdf` text extraction — all theorem statements were transcribed
  verbatim from the PDFs.

## Recommendations for Proof Construction

1. **Proof strategy:** Define the composite coefficient `RC_λ` (Def. 2.9 in
   `literature_review.md`) and prove an additive bound in the TRADES style —
   bound `1 − E[RC_λ]` by a clean-error term plus separate correctness- and
   alignment-boundary terms.
2. **Key prerequisites to cite:** TRADES exact decomposition + Thm 3.1; Cohen's
   certified radius (for a *certified* variant of the metric); CLEVER margin bound;
   CPO constraint-satisfaction (to link ethical training to a robustness gain);
   Tsipras trade-off (to bound the correctness cost of raising alignment).
3. **Computational tools:** `sympy` for symbolic verification of the bound;
   `code/composite_robustness_metric.py` + `numpy` for Monte-Carlo checks;
   `scipy.stats.norm` for `Φ`/`Φ⁻¹`; `scipy.stats` for testing the `Δ ≥ 0.15`
   separation.
4. **Potential difficulties:** discrete (token) prompt perturbations break
   continuous-Lipschitz certification (may need a discrete randomized-substitution
   smoothing analogue); the accuracy–robustness trade-off couples `ρ_C` and `ρ_A`
   (choose `λ` and constraint level `d` to guarantee a *net* gain); reward
   over-optimization (HH `√KL` law) can reduce robustness, so bounds need a
   KL/constraint-strength regularity condition.
