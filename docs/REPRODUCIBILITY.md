# Reproducibility

## Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
Provide official UCF101 split files
Place into data/ucf101/splits/:

classInd.txt

trainlist01.txt, testlist01.txt

Provide processed features
Place extracted features into:
data/ucf101/processed/<ClassName>/<VideoName>.npz
Each file must contain x of shape [T, V, C].

Freeze processed dataset fingerprint
bash
Copy code
python scripts/make_manifest_ucf101.py
Run benchmarks (creates artifacts)
bash
Copy code
python prism/cli_run_benchmark.py --config configs/ucf101/stgcn_nodp.yaml --split 1
python prism/cli_run_benchmark.py --config configs/ucf101/stgcn_dp_eps0.1.yaml --split 1
python prism/cli_run_benchmark.py --config configs/ucf101/stgcn_dp_eps1.0.yaml --split 1
python prism/cli_run_benchmark.py --config configs/ucf101/stgcn_dp_eps10.0.yaml --split 1
python prism/cli_run_benchmark.py --config configs/ucf101/lstm_baseline.yaml --split 1
Build markdown table from artifacts
bash
Copy code
python prism/make_table.py --results_root results/benchmarks/ucf101_split1 --out results/benchmarks/ucf101_split1/table.md
Verify (no-questions proof)
bash
Copy code
python scripts/verify_benchmark_results.py
