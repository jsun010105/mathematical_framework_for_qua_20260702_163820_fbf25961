import time, json
from arxiv_search import search

QUERIES = {
    "certified_robustness": 'all:certified robustness randomized smoothing guarantee',
    "clever_lipschitz": 'all:CLEVER robustness metric Lipschitz lower bound neural network',
    "robustness_accuracy_tradeoff": 'all:robustness accuracy tradeoff adversarial training TRADES',
    "llm_adversarial_prompts": 'all:large language model adversarial prompt attack robustness jailbreak',
    "rlhf_alignment": 'all:reinforcement learning human feedback alignment language model',
    "constitutional_ai_safety": 'all:constitutional AI harmlessness safety training language model',
    "constrained_training": 'all:constrained optimization training neural network Lagrangian safety constraint',
    "alignment_evaluation": 'all:measuring alignment robustness evaluation language model safety benchmark',
    "ml_astronomy": 'all:machine learning astronomy astrophysics deep learning survey',
    "value_alignment_formal": 'all:formal framework value alignment AI ethics quantify',
}
results = {}
for k, q in QUERIES.items():
    try:
        results[k] = search(q, 6)
        print(f"[ok] {k}: {len(results[k])} papers", flush=True)
    except Exception as e:
        print(f"[ERR] {k}: {e}", flush=True)
        results[k] = []
    json.dump(results, open("search_results.json", "w"), indent=2)
    time.sleep(3.0)
print("done")
