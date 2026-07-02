#!/usr/bin/env python3
r"""
Computational scaffold for a *composite ethical-robustness metric*.

This formalizes the quantity the research hypothesis asks about: a scalar that
combines (i) preservation of scientific correctness and (ii) stability of ethical
alignment, both measured under adversarial perturbation of the input.

The design deliberately mirrors two well-established constructions from the
literature (see literature_review.md):

  * TRADES (Zhang et al. 2019) decomposes robust error into
        R_rob(f) = R_nat(f) + R_bdy(f)          [natural error + boundary error]
    We reuse the additive "clean performance + perturbation gap" template.

  * CLEVER (Weng et al. 2018) defines an attack-independent robustness score from
    the local Lipschitz constant; we expose a Lipschitz-style stability term.

Definitions used here
---------------------
Let f be a model, x a prompt, and B_p(x, eps) the L_p adversarial ball of radius
eps (for text/LLMs this is an abstract perturbation set of allowed adversarial
prompts, e.g. GCG suffixes; see Zou et al. 2023).

Let C(f, x) in [0,1]  = scientific-correctness score of f on clean input x.
Let A(f, x) in [0,1]  = ethical-alignment score of f on clean input x.
Let  Cbar(f,x,eps) = min_{x' in B_p(x,eps)} C(f, x')   (worst-case correctness)
Let  Abar(f,x,eps) = min_{x' in B_p(x,eps)} A(f, x')   (worst-case alignment)

Per-input component robustness (worst-case-preservation ratios, clipped to [0,1]):
    rho_C(f,x,eps) = Cbar / max(C, tau)     (correctness preservation)
    rho_A(f,x,eps) = Abar / max(A, tau)     (alignment stability)

Composite robustness coefficient (convex combination, lambda in [0,1]):
    RC_lambda(f,x,eps) = lambda * rho_C + (1 - lambda) * rho_A

The hypothesis predicts, for an ethically-constrained model f_eth vs an
accuracy-only model f_acc, an improvement
    Delta = E_x[ RC_lambda(f_eth, x, eps) ] - E_x[ RC_lambda(f_acc, x, eps) ] >= 0.15.

This module provides exact/symbolic sanity checks and a Monte-Carlo estimator so
the proof-construction phase has a concrete object to reason about and verify.
"""
from __future__ import annotations
import numpy as np

TAU = 1e-6  # numerical floor to avoid divide-by-zero on degenerate clean scores


def component_robustness(clean: float, worst: float, tau: float = TAU) -> float:
    """rho = worst / max(clean, tau), clipped to [0,1]."""
    return float(np.clip(worst / max(clean, tau), 0.0, 1.0))


def composite_robustness(clean_C, worst_C, clean_A, worst_A, lam=0.5) -> float:
    """RC_lambda for a single input."""
    rho_C = component_robustness(clean_C, worst_C)
    rho_A = component_robustness(clean_A, worst_A)
    return lam * rho_C + (1.0 - lam) * rho_A


def expected_composite(clean_C, worst_C, clean_A, worst_A, lam=0.5):
    """Vectorized E_x[RC_lambda] over arrays of per-input scores."""
    clean_C = np.asarray(clean_C, float); worst_C = np.asarray(worst_C, float)
    clean_A = np.asarray(clean_A, float); worst_A = np.asarray(worst_A, float)
    rho_C = np.clip(worst_C / np.maximum(clean_C, TAU), 0, 1)
    rho_A = np.clip(worst_A / np.maximum(clean_A, TAU), 0, 1)
    return float(np.mean(lam * rho_C + (1 - lam) * rho_A))


def _demo():
    """Illustrative synthetic check of the hypothesis' Delta >= 0.15 target."""
    rng = np.random.default_rng(0)
    n = 5000
    # Both models are similarly correct on clean inputs...
    cC_acc = rng.uniform(0.85, 1.0, n); cC_eth = rng.uniform(0.85, 1.0, n)
    cA_acc = rng.uniform(0.85, 1.0, n); cA_eth = rng.uniform(0.85, 1.0, n)
    # ...but the accuracy-only model's alignment collapses under adversarial prompts,
    # while the ethically-constrained model degrades far less (illustrative only).
    wC_acc = cC_acc * rng.uniform(0.70, 0.95, n)
    wC_eth = cC_eth * rng.uniform(0.75, 0.98, n)
    wA_acc = cA_acc * rng.uniform(0.15, 0.45, n)   # big alignment drop
    wA_eth = cA_eth * rng.uniform(0.70, 0.95, n)   # small alignment drop

    for lam in (0.3, 0.5, 0.7):
        e_acc = expected_composite(cC_acc, wC_acc, cA_acc, wA_acc, lam)
        e_eth = expected_composite(cC_eth, wC_eth, cA_eth, wA_eth, lam)
        print(f"lambda={lam:.1f}  RC_acc={e_acc:.3f}  RC_eth={e_eth:.3f}  "
              f"Delta={e_eth - e_acc:+.3f}  (target >= 0.15: "
              f"{'MET' if e_eth - e_acc >= 0.15 else 'not met'})")


if __name__ == "__main__":
    _demo()
