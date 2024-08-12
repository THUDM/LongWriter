import json
import matplotlib.pyplot as plt
import numpy as np

def score(x, y):
    if y > x:
        return 100 * max(0, 1. - (y / x - 1) / 3)
    else:
        return 100 * max(0, 1. - (x / y - 1) / 2)

model = "LongWriter-glm4-9b"
prediction = [json.loads(line) for line in open(f'models/{model}/pred.jsonl', encoding='utf-8')]
x, y, scores = [], [], []
for pred in prediction:
    x.append(pred["length"])
    y.append(pred["response_length"])
    scores.append(score(pred["length"], pred["response_length"]))

print(np.mean(scores))

# set plt size 6x6
plt.figure(figsize=(6, 6))
lmt = 25000
# plot x, y
plt.scatter(x, y, s=100, c='r', alpha=0.3)
# plot x=y
plt.plot([0, lmt], [0, lmt], 'k--')
plt.xscale('log')
plt.yscale('log')
plt.xlim(50, lmt)
plt.ylim(50, lmt)
plt.xlabel('Required Length', fontsize=20, fontweight='bold')
plt.ylabel('Output Length', fontsize=20, fontweight='bold')
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.tight_layout()
plt.savefig(f'models/{model}/scatter.png')