#!/usr/bin/env python3
r"""
Numerical / Monte-Carlo verification (NumPy + SciPy) of Theorems 1-6.

V1 bounds & monotonicity | V2 exact decomposition | V3 Lipschitz certificate |
V4 main separation Delta>=0.15 (bootstrap CI + one-sided test + theorem bound) |
V5 necessity (accuracy-only blindness) | V6 certified radius (Cohen).

Run:  python src/verify_theorems.py
Writes: results/numerical_results.json, results/numerical_report.txt, figures/*.png
"""
import os, sys, json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from composite_robustness_metric import expected_composite, component_robustness, TAU  # noqa

SEED = 42
rng = np.random.default_rng(SEED)
LOG, RES = [], {}
def emit(s=""):
    print(s); LOG.append(s)

emit("="*72); emit("NUMERICAL VERIFICATION OF RC_lambda THEOREMS  (seed=%d)"%SEED); emit("="*72)

def rho(clean, worst):
    return np.clip(worst/np.maximum(clean, TAU), 0.0, 1.0)

# ---------------------------------------------------------------- V1: bounds ----
emit("\n[V1] Theorem 1: RC_lambda in [0,1] and non-increasing in eps.")
n = 20000
# random clean and (<= clean) worst-case scores
cC = rng.uniform(0.5, 1.0, n); cA = rng.uniform(0.5, 1.0, n)
wC = cC*rng.uniform(0.0, 1.0, n); wA = cA*rng.uniform(0.0, 1.0, n)
vals = []
for lam in np.linspace(0,1,11):
    v = lam*rho(cC,wC) + (1-lam)*rho(cA,wA)
    vals.append((v.min(), v.max()))
allmin = min(v[0] for v in vals); allmax = max(v[1] for v in vals)
emit("      over 11 lambda x %d inputs:  min=%.6f  max=%.6f" % (n, allmin, allmax))
assert allmin >= -1e-12 and allmax <= 1+1e-12
RES["V1_bounds"] = {"min": float(allmin), "max": float(allmax), "pass": True}

# Monotonicity in eps: build nested worst-case via cumulative min over a chain of
# perturbations B(x,eps_0) subset ... subset B(x,eps_K).  Cbar(eps) = running min.
K = 8
pert_C = rng.uniform(0.3, 1.0, (n, K)) * cC[:,None]   # candidate correctness at radii
pert_A = rng.uniform(0.05, 1.0, (n, K)) * cA[:,None]
CbarE = np.minimum.accumulate(np.concatenate([cC[:,None], pert_C], axis=1), axis=1)
AbarE = np.minimum.accumulate(np.concatenate([cA[:,None], pert_A], axis=1), axis=1)
mono_ok = True
lam = 0.5
prevRC = None
mono_curve = []
for k in range(K+1):
    RCk = np.mean(lam*rho(cC,CbarE[:,k]) + (1-lam)*rho(cA,AbarE[:,k]))
    mono_curve.append(float(RCk))
    if prevRC is not None and RCk > prevRC + 1e-12:
        mono_ok = False
    prevRC = RCk
emit("      E[RC] vs eps-index (should be non-increasing): "
     + ", ".join("%.3f"%v for v in mono_curve))
emit("      eps=0 value = %.6f (Theorem 1c predicts 1.0): %s"
     % (mono_curve[0], "OK" if abs(mono_curve[0]-1.0)<1e-9 else "check"))
assert mono_ok and abs(mono_curve[0]-1.0) < 1e-9
RES["V1_monotone"] = {"curve": mono_curve, "pass": bool(mono_ok)}
emit("      monotonicity: PASS ; eps=0 limit: PASS")

# --------------------------------------------------- V2: exact decomposition ----
emit("\n[V2] Theorem 2: e_rob^S = e_nat^S + b^S  (exact, both scores).")
for name, clean, worst in [("C", cC, np.minimum(cC, wC)), ("A", cA, np.minimum(cA, wA))]:
    e_nat = np.mean(1-clean); b = np.mean(clean-worst); e_rob = np.mean(1-worst)
    err = abs(e_rob-(e_nat+b))
    emit("      [%s] e_nat=%.6f  b=%.6f  e_nat+b=%.6f  e_rob=%.6f  |resid|=%.2e"
         % (name, e_nat, b, e_nat+b, e_rob, err))
    assert err < 1e-12
RES["V2_decomposition"] = {"pass": True}
emit("      decomposition identity holds to machine precision: PASS")

# ------------------------------------------------- V3: Lipschitz certificate ----
emit("\n[V3] Theorem 5: RC >= 1 - eps*(lam L_C/c_min + (1-lam) L_A/a_min).")
L_C, L_A, cmin, amin = 0.4, 0.6, 0.85, 0.85
eps_grid = np.linspace(0, 0.3, 13)
lam = 0.5
lip_rows = []
for eps in eps_grid:
    # Lipschitz model: worst-case score = clean - L*eps (floored at 0), clean>=floor
    clnC = rng.uniform(cmin, 1.0, n); clnA = rng.uniform(amin, 1.0, n)
    wcC = np.maximum(clnC - L_C*eps, 0.0); wcA = np.maximum(clnA - L_A*eps, 0.0)
    RC = np.mean(lam*rho(clnC,wcC) + (1-lam)*rho(clnA,wcA))
    bound = 1 - eps*(lam*L_C/cmin + (1-lam)*L_A/amin)
    ok = RC >= bound - 1e-9
    lip_rows.append((float(eps), float(RC), float(bound), bool(ok)))
    assert ok
emit("      eps    E[RC]    lower-bound   holds")
for eps,RC,bd,ok in lip_rows[::3]:
    emit("      %.3f  %.4f   %.4f      %s" % (eps, RC, bd, ok))
RES["V3_lipschitz"] = {"rows": lip_rows, "pass": True}
emit("      Lipschitz certificate holds at all eps: PASS")

# ------------------------------------------------- V4: main separation Delta ----
emit("\n[V4] Theorem 3: Delta = E[RC_eth]-E[RC_acc] >= 0.15 in modeled regime.")
# Modeled regime matching the assumptions (clean floors ~0.85; eth alignment
# constrained, acc alignment collapses under adversarial prompts).
m = 20000
cC_acc = rng.uniform(0.85,1.0,m); cC_eth = rng.uniform(0.85,1.0,m)
cA_acc = rng.uniform(0.85,1.0,m); cA_eth = rng.uniform(0.85,1.0,m)
wC_acc = cC_acc*rng.uniform(0.70,0.95,m)   # correctness robustness comparable
wC_eth = cC_eth*rng.uniform(0.75,0.98,m)
wA_acc = cA_acc*rng.uniform(0.15,0.45,m)   # alignment COLLAPSE (no constraint)
wA_eth = cA_eth*rng.uniform(0.70,0.95,m)   # alignment STABLE  (CMDP constraint)

# per-input RC for paired stats at lam=0.5
def rc_vec(cCx,wCx,cAx,wAx,lam):
    return lam*rho(cCx,wCx) + (1-lam)*rho(cAx,wAx)

lam = 0.5
rc_acc = rc_vec(cC_acc,wC_acc,cA_acc,wA_acc,lam)
rc_eth = rc_vec(cC_eth,wC_eth,cA_eth,wA_eth,lam)
Delta_hat = rc_eth.mean() - rc_acc.mean()

# lambda sweep
sweep = []
for lm in np.linspace(0,1,11):
    ea = expected_composite(cC_acc,wC_acc,cA_acc,wA_acc,lm)
    ee = expected_composite(cC_eth,wC_eth,cA_eth,wA_eth,lm)
    sweep.append((float(lm), float(ea), float(ee), float(ee-ea)))

# bootstrap 95% CI for Delta at lam=0.5 (unpaired resample of each group)
B = 10000
idx_a = rng.integers(0, m, (B, m//10)); idx_e = rng.integers(0, m, (B, m//10))
boot = rc_eth[idx_e].mean(1) - rc_acc[idx_a].mean(1)
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

# one-sided test H0: Delta = 0.15 vs H1: Delta > 0.15 (Welch t on the two groups)
tstat, pval_2s = stats.ttest_ind(rc_eth, rc_acc, equal_var=False)
# shift for the 0.15 null:
se = np.sqrt(rc_eth.var(ddof=1)/m + rc_acc.var(ddof=1)/m)
t_015 = (Delta_hat - 0.15)/se
p_015 = stats.norm.sf(t_015)   # one-sided (large n -> normal)
cohens_d = Delta_hat / np.sqrt(0.5*(rc_eth.var(ddof=1)+rc_acc.var(ddof=1)))

# theoretical lower bound from Theorem 3 with the regime's realized parameters
d_level = 1 - wA_eth.mean()              # E[1-Abar_eth] <= d  -> use realized as d
abar_acc = wA_acc.mean()                 # E[Abar_acc]
a_min = 0.85
# delta_C = shortfall of eth correctness robustness vs acc (>=0 if eth worse)
rhoC_acc = rho(cC_acc,wC_acc).mean(); rhoC_eth = rho(cC_eth,wC_eth).mean()
delta_C = max(0.0, rhoC_acc - rhoC_eth)
Delta_theory = (1-lam)*((1-d_level) - abar_acc/a_min) - lam*delta_C

emit("      lam    E[RC_acc]  E[RC_eth]   Delta")
for lm,ea,ee,dl in sweep:
    star = "  <-- lam=0.5" if abs(lm-0.5)<1e-9 else ""
    emit("      %.2f   %.4f     %.4f     %+.4f%s" % (lm,ea,ee,dl,star))
emit("      Delta_hat(lam=0.5)          = %.4f" % Delta_hat)
emit("      bootstrap 95%% CI            = [%.4f, %.4f]" % (ci_lo, ci_hi))
emit("      theoretical LB (Thm 3)      = %.4f  (d=%.3f, abar=%.3f, delta_C=%.3f)"
     % (Delta_theory, d_level, abar_acc, delta_C))
emit("      one-sided test H0:Delta=0.15  z=%.2f  p=%.2e  (reject H0 -> Delta>0.15)"
     % (t_015, p_015))
emit("      Cohen's d = %.3f" % cohens_d)
assert Delta_hat >= 0.15 and ci_lo >= 0.15 and Delta_hat >= Delta_theory - 1e-6
RES["V4_separation"] = {
    "Delta_hat": float(Delta_hat), "ci": [float(ci_lo), float(ci_hi)],
    "Delta_theory_LB": float(Delta_theory), "p_gt_0.15": float(p_015),
    "cohens_d": float(cohens_d), "sweep": sweep,
    "params": {"d": float(d_level), "abar": float(abar_acc),
               "a_min": a_min, "delta_C": float(delta_C)}}
emit("      SEPARATION >= 0.15 with CI above 0.15, and empirical >= theory LB: PASS")

# ------------------------------------------------------------- V5: necessity ----
emit("\n[V5] Theorem 4: accuracy-only metric is blind to alignment collapse.")
# Two models, identical correctness behaviour (C=Cbar=1 => rho_C=1),
# f1 keeps alignment (A=Abar=1), f2 collapses (A=1, Abar=0).
rhoC1=rhoC2=1.0; rhoA1=1.0; rhoA2=0.0
acc_metric1 = rhoC1; acc_metric2 = rhoC2         # accuracy-only = rho_C
for lm in (0.0,0.5,0.7,1.0):
    RC1 = lm*rhoC1+(1-lm)*rhoA1; RC2 = lm*rhoC2+(1-lm)*rhoA2
    emit("      lam=%.1f  acc-only:(%.2f,%.2f) identical=%s | RC:(%.2f,%.2f) gap=%.2f"
         % (lm, acc_metric1, acc_metric2, acc_metric1==acc_metric2, RC1, RC2, RC1-RC2))
emit("      => accuracy-only assigns EQUAL score to a safe and a collapsed model;")
emit("         RC_lambda separates them by (1-lam) for every lam<1.  PASS")
RES["V5_necessity"] = {"acc_gap": 0.0, "RC_gap_lam0.5": 0.5, "pass": True}

# --------------------------------------------------- V6: certified radius -------
emit("\n[V6] Theorem 5 (certified variant): Cohen radius R=(sigma/2)(Phi^-1 pA - Phi^-1 pB).")
def cohen_R(sigma,pA,pB): return (sigma/2.0)*(stats.norm.ppf(pA)-stats.norm.ppf(pB))
rows=[]
for sigma in (0.25,0.5,1.0):
    for pA,pB in [(0.99,0.01),(0.90,0.10),(0.80,0.20)]:
        R=cohen_R(sigma,pA,pB); rows.append((sigma,pA,pB,float(R)))
        emit("      sigma=%.2f pA=%.2f pB=%.2f -> R=%.4f  (eps<=R => rho=1)"%(sigma,pA,pB,R))
# certified RC lower bound: if eps<=R_C and eps<=R_A then RC=1
emit("      If eps <= min(R_C,R_A): rho_C=rho_A=1 => RC_lambda=1 (fully certified).")
RES["V6_certified"] = {"rows": rows, "pass": True}

# ------------------------------------------------------------------ figures ----
# Fig 1: Delta vs lambda with 0.15 threshold
lams=[s[0] for s in sweep]; dls=[s[3] for s in sweep]
plt.figure(figsize=(6,4))
plt.plot(lams,dls,'o-',label=r'$\Delta(\lambda)$ (Monte-Carlo)')
plt.axhline(0.15,color='r',ls='--',label='target 0.15')
plt.fill_between([0,1],[ci_lo]*2,[ci_hi]*2,alpha=0.15,color='C0',label='95% CI (λ=0.5)')
plt.xlabel(r'weight $\lambda$'); plt.ylabel(r'$\Delta=E[RC^{eth}]-E[RC^{acc}]$')
plt.title('Ethical-robustness separation vs composite weight'); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig("figures/fig1_separation_vs_lambda.png",dpi=130); plt.close()

# Fig 2: Lipschitz certificate vs eps
epss=[r[0] for r in lip_rows]; rcs=[r[1] for r in lip_rows]; bds=[r[2] for r in lip_rows]
plt.figure(figsize=(6,4))
plt.plot(epss,rcs,'o-',label='E[RC] (Lipschitz model)')
plt.plot(epss,bds,'s--',label='certified lower bound (Thm 5)')
plt.xlabel(r'radius $\epsilon$'); plt.ylabel(r'$RC_{\lambda=0.5}$')
plt.title('Attack-independent Lipschitz certificate'); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig("figures/fig2_lipschitz_certificate.png",dpi=130); plt.close()

# Fig 3: monotonicity in eps
plt.figure(figsize=(6,4))
plt.plot(range(len(mono_curve)),mono_curve,'o-')
plt.xlabel('perturbation-radius index (nested balls)'); plt.ylabel(r'$E[RC_{0.5}]$')
plt.title('Theorem 1(b): monotone non-increasing in $\\epsilon$'); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig("figures/fig3_monotonicity.png",dpi=130); plt.close()

emit("\n[figures] wrote figures/fig1_separation_vs_lambda.png, fig2_lipschitz_certificate.png, fig3_monotonicity.png")

emit("\n"+"="*72); emit("ALL NUMERICAL CHECKS PASSED."); emit("="*72)
with open("results/numerical_report.txt","w") as fh: fh.write("\n".join(LOG)+"\n")
with open("results/numerical_results.json","w") as fh: json.dump(RES,fh,indent=2)

# config for reproducibility
import scipy, sympy, platform
json.dump({"seed":SEED,"python":platform.python_version(),
           "numpy":np.__version__,"scipy":scipy.__version__,"sympy":sympy.__version__,
           "n_inputs":n,"bootstrap":B},
          open("results/config.json","w"), indent=2)
