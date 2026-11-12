import json
from pathlib import Path

import torch

from m1_unilab.evaluate import evaluate

root = Path(__file__).resolve().parents[1]
out = root / "outputs/m1_unilab/evaluations/iteration_24000"
torch.set_num_threads(1)
results = []
for level in (0, 2, 4):
    for terrain in ("flat", "uphill", "downhill", "rough", "stairs_up", "stairs_down"):
        row = evaluate(out / "checkpoint.pt", terrain, steps=1000, count=8,
                       speed=0.5, level=level, profile="b2w-full")
        results.append(row)
        print(json.dumps(row), flush=True)
        (out / "metrics.json").write_text(json.dumps({
            "checkpoint": json.loads((out / "checkpoint.json").read_text()),
            "conditions": "nominal dynamics, no sensor noise or pushes, zero motor delay, upright reset with XY +-0.1 m; seed 123; tilt >0.8 rad fails; each row stops scoring at 4 m or failure, maximum 20 s",
            "results": results,
        }, ensure_ascii=False, indent=2) + "\n")
