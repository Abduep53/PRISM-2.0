
---

# ✅ 4) Документация (корректная, “как в paper”)

## `docs/UCF101_BENCHMARK_PROTOCOL.md`
```md
# UCF101 Benchmark Protocol (Evidence-Grade)

## Proof definition
A benchmark is considered proven if:
- `preds.npz` contains `y_true` and `y_pred` for the official test list
- `ids.txt` lists sample IDs aligned with those arrays
- `metrics.json` matches recomputation from `preds.npz`
- `evidence.json` contains SHA256 hashes for anti-tamper verification
- `scripts/verify_benchmark_results.py` passes

## Dataset
- UCF101 (101 action classes)
- Official split lists must be placed under `data/ucf101/splits/`:
  - classInd.txt
  - trainlist01.txt / testlist01.txt (and optionally 02/03)

## Processed features
This repo expects **processed features** (e.g., pose/skeleton sequences) extracted from UCF101 videos.

File location:
`data/ucf101/processed/<ClassName>/<VideoName>.npz`

Required key:
- `x`: float32 array of shape [T, V, C]

Where:
- T = frames
- V = number of joints
- C = channels (e.g., 2 for x,y)

The dataloader center-crops or pads sequences to a fixed `clip_len`.

## Manifest
To freeze which processed files were used:
- `data/ucf101/manifest_sha256.txt` stores SHA256 for every `.npz` under `data/ucf101/processed/`.

## Differential Privacy
DP training uses DP-SGD (Opacus):
- `noise_multiplier` and `max_grad_norm` define DP noise/clipping
- `delta` is stored
- Achieved epsilon may be recorded if an accountant is used

The benchmark table reports the configured privacy budget ε.
