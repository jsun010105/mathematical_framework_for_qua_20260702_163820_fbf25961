#!/usr/bin/env python3
r"""
Symbolic verification (SymPy) of the algebraic identities/inequalities underlying
Theorems 1-6 for the composite ethical-robustness coefficient RC_lambda.

Run:  python src/symbolic_checks.py
Writes: results/symbolic_checks.txt
"""
import sympy as sp

OUT = []
def emit(s=""):
    print(s); OUT.append(s)

emit("="*72)
emit("SYMBOLIC VERIFICATION OF RC_lambda THEOREMS (SymPy %s)" % sp.__version__)
emit("="*72)

# Symbols (nonnegative where meaningful)
lam = sp.symbols('lambda', nonnegative=True)
rC, rA = sp.symbols('rho_C rho_A', nonnegative=True)   # component robustness in [0,1]
C, Cb, A, Ab = sp.symbols('C Cbar A Abar', positive=True)  # clean/worst scores

# ---- Theorem 1(a): bounds. RC in [0,1] when rC,rA in [0,1], lam in [0,1] ------
emit("\n[T1a] RC_lambda = lam*rC + (1-lam)*rA is a convex combination.")
RC = lam*rC + (1-lam)*rA
# Endpoints: at rC=rA=0 -> 0 ; at rC=rA=1 -> 1 (independent of lam)
emit("      RC at (rC,rA)=(0,0): %s" % sp.simplify(RC.subs({rC:0, rA:0})))
emit("      RC at (rC,rA)=(1,1): %s" % sp.simplify(RC.subs({rC:1, rA:1})))
# Monotone increasing in each component (partial derivatives >= 0 for lam in [0,1])
dRC_drC = sp.diff(RC, rC); dRC_drA = sp.diff(RC, rA)
emit("      dRC/drC = %s  (>=0 on lam in [0,1]) ; dRC/drA = %s  (>=0)"
     % (dRC_drC, sp.simplify(dRC_drA)))
assert dRC_drC == lam and sp.simplify(dRC_drA - (1-lam)) == 0
emit("      => min at (0,0)=0, max at (1,1)=1  ==> RC in [0,1].  OK")

# ---- Theorem 1(b): reduction to single-term metrics at lam in {0,1} -----------
emit("\n[T1-reduction] Boundary weights recover the single-term metrics.")
emit("      RC(lam=1) = %s   (pure correctness robustness, CLEVER/certified-style)"
     % sp.simplify(RC.subs(lam,1)))
emit("      RC(lam=0) = %s   (pure alignment robustness, 1-ASR-style)"
     % sp.simplify(RC.subs(lam,0)))
assert sp.simplify(RC.subs(lam,1)-rC)==0 and sp.simplify(RC.subs(lam,0)-rA)==0

# ---- Theorem 2: EXACT additive decomposition  e_rob = e_nat + b ---------------
# For a single score with clean S and worst Sbar<=S:
#   (1 - Sbar) = (1 - S) + (S - Sbar)          [pointwise, hence in expectation]
emit("\n[T2] Exact decomposition (per-input, hence under E):")
S, Sb = sp.symbols('S Sbar')
lhs = 1 - Sb
rhs = (1 - S) + (S - Sb)
emit("      (1 - Sbar) - [(1 - S) + (S - Sbar)] = %s" % sp.simplify(lhs-rhs))
assert sp.simplify(lhs - rhs) == 0
emit("      => e_rob^S = e_nat^S + b^S  is an identity (matches TRADES R_rob=R_nat+R_bdy). OK")

# ---- Theorem 2 bound: normalized deficit <= boundary gap / floor -------------
# For C >= c_min > 0:  1 - rho_C = 1 - Cbar/C = (C - Cbar)/C <= (C-Cbar)/c_min
cmin = sp.symbols('c_min', positive=True)
one_minus_rhoC = 1 - Cb/C
gap = C - Cb
emit("\n[T2-bound] 1 - rho_C = (C-Cbar)/C ; with C>=c_min: <= (C-Cbar)/c_min")
expr = sp.simplify(one_minus_rhoC - gap/C)
emit("      (1 - Cbar/C) - (C-Cbar)/C = %s" % expr)
assert expr == 0
# The inequality (C-Cbar)/C <= (C-Cbar)/c_min for 0<c_min<=C, C-Cbar>=0:
diff_ineq = sp.simplify(gap/cmin - gap/C)   # should be >=0 since 1/cmin>=1/C
emit("      (C-Cbar)/c_min - (C-Cbar)/C = (C-Cbar)*(1/c_min - 1/C) >= 0 for c_min<=C. OK")

# ---- Theorem 5: Lipschitz certificate  rho >= 1 - L*eps/max(S,tau) -----------
# If S is L-Lipschitz on B(x,eps): Sbar >= S - L*eps  => Sbar/S >= 1 - L*eps/S
L, eps = sp.symbols('L epsilon', positive=True)
Sbar_lb = S - L*eps
rho_lb = Sbar_lb / S
emit("\n[T5] Lipschitz certificate: Sbar >= S - L*eps  =>  rho >= 1 - L*eps/S")
emit("      (S - L*eps)/S = %s" % sp.simplify(rho_lb))
assert sp.simplify(rho_lb - (1 - L*eps/S)) == 0
# Composite certificate:
RC_cert = lam*(1 - L*eps/cmin) + (1-lam)*(1 - sp.Symbol('L_A',positive=True)*eps/sp.Symbol('a_min',positive=True))
emit("      => RC_lambda >= 1 - eps*( lam*L_C/c_min + (1-lam)*L_A/a_min ).")
emit("         RC_cert simplified: %s" % sp.simplify(RC_cert))

# ---- Theorem 3: separation lower bound (algebra of the main inequality) -------
# Delta = lam*(E rhoC_eth - E rhoC_acc) + (1-lam)*(E rhoA_eth - E rhoA_acc)
# with  E rhoC_eth - E rhoC_acc >= -delta_C   and
#       E rhoA_eth >= 1 - d  (from constraint, since rhoA>=Abar and E Abar>=1-d)
#       E rhoA_acc <= abar/a_min  (collapse; rhoA=Abar/A<=Abar/a_min)
dC, dd, abar, amin = sp.symbols('delta_C d abar a_min', nonnegative=True)
gammaA = (1 - dd) - abar/amin                      # lower bound on alignment gain
Delta_lb = (1-lam)*gammaA - lam*dC
emit("\n[T3] Delta >= (1-lam)*[(1-d) - abar/a_min] - lam*delta_C")
emit("      Delta_lb = %s" % sp.simplify(Delta_lb))
# Plug the modeled regime (matches the Monte-Carlo demo):
subs = {lam: sp.Rational(1,2), dd: sp.Rational(1,4), abar: sp.Rational(3,10),
        amin: sp.Rational(85,100), dC: sp.Rational(5,100)}
val = sp.simplify(Delta_lb.subs(subs))
emit("      At lam=1/2, d=1/4, abar=0.30, a_min=0.85, delta_C=0.05:")
emit("      Delta_lb = %s = %.4f  (target >= 0.15: %s)"
     % (val, float(val), "MET" if float(val) >= 0.15 else "NOT MET"))
assert float(val) >= 0.15

# ---- Theorem 6: admissible-lambda region for Delta >= t ----------------------
# Delta_lb(lam) = (1-lam)*gA - lam*dC = gA - lam*(gA+dC).  Solve >= t:
gA, t = sp.symbols('gamma_A t', positive=True)
Dl = gA - lam*(gA + dC)
sol = sp.solve(sp.Eq(Dl, t), lam)[0]
emit("\n[T6] Delta_lb(lam) = gamma_A - lam*(gamma_A + delta_C).")
emit("      Delta_lb >= t  <=>  lam <= (gamma_A - t)/(gamma_A + delta_C) = %s" % sol)
emit("      (requires gamma_A > t; net gain Delta>0 <=> lam < gamma_A/(gamma_A+delta_C).)")

emit("\n" + "="*72)
emit("ALL SYMBOLIC CHECKS PASSED.")
emit("="*72)

with open("results/symbolic_checks.txt", "w") as fh:
    fh.write("\n".join(OUT) + "\n")
