import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
data=json.loads((root/"data/results/retrieval_ladder_metrics.json").read_text(encoding="utf-8"))
print(json.dumps(data["adaptive_corrective"], indent=2))
