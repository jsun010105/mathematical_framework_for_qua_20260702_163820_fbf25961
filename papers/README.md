# Downloaded Papers

Curated literature for **"Mathematical Framework for Quantifying Ethical Robustness
in Adversarial Astrophysics ML Systems."** Papers are grouped by the role they play
in supporting the hypothesis (that models trained with explicit ethical constraints
have measurably higher robustness under adversarial prompts, quantifiable via a
composite metric of scientific-correctness preservation + alignment stability).

## Group A — Formal robustness metrics (mathematical backbone)

1. **Theoretically Principled Trade-off between Robustness and Accuracy (TRADES)** — `1901.08573_zhang2019_trades_tradeoff.pdf`
   - Authors: Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric P. Xing, Laurent El Ghaoui, Michael I. Jordan — 2019
   - Why relevant: Decomposes robust error into **natural error + boundary error**. This decomposition is the template for a *composite* robustness metric (correctness term + stability term).

2. **Evaluating the Robustness of Neural Networks: An Extreme Value Theory Approach (CLEVER)** — `1801.10578_weng2018_clever_robustness_metric.pdf`
   - Authors: Tsui-Wei Weng, Huan Zhang, Pin-Yu Chen, et al. — 2018
   - Why relevant: An **attack-independent, scalar robustness metric** derived from the local Lipschitz constant via extreme value theory. Direct precedent for defining a "robustness coefficient."

3. **Certified Adversarial Robustness via Randomized Smoothing** — `1902.02918_cohen2019_randomized_smoothing_certified.pdf`
   - Authors: Jeremy Cohen, Elan Rosenfeld, J. Zico Kolter — 2019
   - Why relevant: Provides a **certified radius** R with a closed-form guarantee — a rigorously-quantified robustness value.

## Group B — Adversarial robustness foundations

4. **Towards Deep Learning Models Resistant to Adversarial Attacks (PGD/Madry)** — `1706.06083_madry2017_pgd_robust_optimization.pdf`
   - Authors: Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu — 2017
   - Why relevant: The **min-max (saddle-point) robust optimization** framework; the formal definition of adversarial robustness against an L_p perturbation set.

5. **Robustness May Be at Odds with Accuracy** — `1805.12152_tsipras2018_robustness_odds_accuracy.pdf`
   - Authors: Dimitris Tsipras, Shibani Santurkar, Logan Engstrom, Alexander Turner, Aleksander Madry — 2018
   - Why relevant: Proves (via an explicit distribution) a fundamental **tradeoff between standard and robust accuracy** — context for why a composite metric is needed.

6. **Explaining and Harnessing Adversarial Examples (FGSM)** — `1412.6572_goodfellow2014_explaining_adv_examples.pdf`
   - Authors: Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy — 2014
   - Why relevant: Foundational adversarial-example construction (FGSM); linear explanation of vulnerability.

## Group C — Ethical constraints during training (alignment)

7. **Constitutional AI: Harmlessness from AI Feedback** — `2212.08073_bai2022_constitutional_ai.pdf`
   - Authors: Yuntao Bai et al. (Anthropic) — 2022
   - Why relevant: Operationalizes **explicit ethical constraints ("a constitution") during training** via SL-CAI + RL-CAI (RLAIF).

8. **Training language models to follow instructions with human feedback (InstructGPT)** — `2203.02155_ouyang2022_instructgpt_rlhf.pdf`
   - Authors: Long Ouyang et al. (OpenAI) — 2022
   - Why relevant: Canonical **RLHF** pipeline (SFT → reward model → PPO with KL penalty) — the mechanism for injecting alignment objectives.

9. **Training a Helpful and Harmless Assistant with RLHF** — `2204.05862_bai2022_helpful_harmless_rlhf.pdf`
   - Authors: Yuntao Bai et al. (Anthropic) — 2022
   - Why relevant: Measures **helpfulness vs harmlessness** tradeoff (Elo/preference) — a two-objective analogue of correctness vs ethical alignment.

## Group D — Adversarial prompts & constrained training

10. **Universal and Transferable Adversarial Attacks on Aligned Language Models (GCG)** — `2307.15043_zou2023_universal_transferable_attacks.pdf`
    - Authors: Andy Zou, Zifan Wang, Nicholas Carlini, et al. — 2023
    - Why relevant: Defines **adversarial-prompt (suffix) optimization** against aligned LLMs — the "adversarial prompt" threat model, made concrete.

11. **JailbreakBench: An Open Robustness Benchmark for Jailbreaking LLMs** — `2404.01318_chao2024_jailbreakbench.pdf`
    - Authors: Patrick Chao, Edoardo Debenedetti, et al. — 2024
    - Why relevant: Standardized measurement of **attack success rate / robustness** to adversarial prompts.

12. **Constrained Policy Optimization (CPO)** — `1705.10528_achiam2017_constrained_policy_opt.pdf`
    - Authors: Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel — 2017
    - Why relevant: **Constrained-MDP** training (max reward s.t. expected cost ≤ d) — the formal template for "ethical constraints during training."

13. **The Modern Mathematics of Deep Learning** — `2105.04026_berner2021_modern_math_deep_learning.pdf`
    - Authors: Julius Berner, Philipp Grohs, Gitta Kutyniok, Philipp Petersen — 2021
    - Why relevant: General mathematical framing (approximation, generalization, robustness) for formal DL analysis.
