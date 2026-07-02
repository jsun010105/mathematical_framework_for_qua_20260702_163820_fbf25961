# Computational Tools

Lightweight computational support for the mathematics research project. No heavy
external repositories were required; the work is formal/analytical, so the tools
here are (a) literature-gathering helpers and (b) a numerical scaffold for the
proposed composite robustness metric.

## Installed Python packages (in `.venv`)
| Package | Purpose |
|---------|---------|
| sympy   | Symbolic verification of metric identities / bounds |
| numpy   | Vectorized Monte-Carlo estimation of expected robustness |
| scipy   | Statistical tests, Gaussian CDF `Phi`/`Phi^{-1}` for certified-radius formulas |
| requests, httpx | arXiv / API access for literature search |
| pypdf   | PDF chunking for deep reading |

## Scripts

### `composite_robustness_metric.py`
Reference implementation of the candidate **composite ethical-robustness
coefficient** `RC_lambda = lambda * rho_C + (1-lambda) * rho_A`, where
`rho_C` = worst-case scientific-correctness preservation and `rho_A` = worst-case
ethical-alignment stability under an adversarial perturbation set. Mirrors the
additive TRADES decomposition and CLEVER's Lipschitz-based scalar-metric idea.
Includes a vectorized `expected_composite(...)` estimator and a synthetic demo
checking the hypothesis' `Delta >= 0.15` target.
Run: `python code/composite_robustness_metric.py`

### `arxiv_search.py`
arXiv Atom-API query helper (relevance-sorted, with retry/backoff and a proper
User-Agent). Usage: `python code/arxiv_search.py "all:<query>" <n>`

### `batch_search.py`
Runs a batch of topical arXiv queries with pacing; writes `search_results.json`.

### `download_papers.py`
Downloads the curated paper set to `papers/` (paced, validates PDF magic bytes).

## Notes
- No user-specified `code_references` were provided in the research spec, so no
  external repositories were cloned.
- The `search_results.json` file records raw candidate hits from the literature
  search for provenance.
