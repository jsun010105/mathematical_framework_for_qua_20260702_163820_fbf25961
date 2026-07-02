import time, os, requests
HDR = {'User-Agent': 'Mozilla/5.0 ResearchBot (chicagohailab@gmail.com)'}
# (arxiv_id, descriptive_filename, short reason)
PAPERS = [
    # --- Math backbone: adversarial robustness formalism & metrics ---
    ("1706.06083", "madry2017_pgd_robust_optimization",       "min-max robust optimization formulation"),
    ("1412.6572",  "goodfellow2014_explaining_adv_examples",  "foundational adversarial examples / FGSM"),
    ("1902.02918", "cohen2019_randomized_smoothing_certified","certified robustness radius (formal metric)"),
    ("1801.10578", "weng2018_clever_robustness_metric",       "CLEVER attack-independent robustness metric"),
    ("1901.08573", "zhang2019_trades_tradeoff",               "robust error decomposition (composite metric)"),
    ("1805.12152", "tsipras2018_robustness_odds_accuracy",    "robustness-accuracy tradeoff theory"),
    ("2105.04026", "berner2021_modern_math_deep_learning",    "general math-of-DL reference"),
    # --- LLM adversarial prompts / robustness of aligned models ---
    ("2307.15043", "zou2023_universal_transferable_attacks",  "adversarial prompts on aligned LLMs (GCG)"),
    ("2404.01318", "chao2024_jailbreakbench",                 "robustness benchmark for jailbreaks"),
    # --- Ethical constraints during training / alignment ---
    ("2212.08073", "bai2022_constitutional_ai",               "ethical constraints via AI feedback"),
    ("2203.02155", "ouyang2022_instructgpt_rlhf",             "RLHF alignment training"),
    ("2204.05862", "bai2022_helpful_harmless_rlhf",           "helpful+harmless RLHF assistant"),
    # --- Constrained optimization / safe training ---
    ("1705.10528", "achiam2017_constrained_policy_opt",       "constrained optimization during training"),
]
os.makedirs("papers", exist_ok=True)
ok, fail = [], []
for aid, name, reason in PAPERS:
    path = f"papers/{aid}_{name}.pdf"
    if os.path.exists(path) and os.path.getsize(path) > 20000:
        print(f"[skip] {path} exists"); ok.append((aid,name,reason)); continue
    url = f"https://arxiv.org/pdf/{aid}.pdf"
    try:
        r = requests.get(url, headers=HDR, timeout=90)
        if r.status_code == 200 and r.content[:4] == b'%PDF':
            open(path, "wb").write(r.content)
            print(f"[ok] {path}  ({len(r.content)//1024} KB)"); ok.append((aid,name,reason))
        else:
            print(f"[FAIL {r.status_code}] {aid} not PDF"); fail.append(aid)
    except Exception as e:
        print(f"[ERR] {aid}: {e}"); fail.append(aid)
    time.sleep(3.0)
print(f"\nDownloaded {len(ok)}, failed {len(fail)}: {fail}")
