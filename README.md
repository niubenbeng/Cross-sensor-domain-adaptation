# Cross-Sensor Domain Adaptation for Mechanical Fault Diagnosis

Cross-sensor transfer learning for mechanical fault diagnosis, fusing JMMSD (RBF + squared) and JVDR (Student-t + centered squared) joint distribution alignment on top of a 1D Swin Transformer backbone.

## Project Structure

```
Cross-sensor-domain-adaptation/
├── main_framework/
│   ├── main.py                  # Training entry point
│   ├── losses.py                # MMSD / JMMSD / VDR / JVDR / JVM losses
│   ├── swin_transformer_1d.py   # 1D Swin Transformer backbone + PatchMerging
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── README.md
```

## Environment
- PyTorch 2.5.1 + CUDA 12.4
- Install dependencies: `pip install -r requirements.txt`

## Data Layout

Data Availability:
The CWRU dataset: https://engineering.case.edu/bearingdatacenter/download-data-file.
The XJTU Spurgear: https://github.com/HazeDT/PHMGNNBenchmark/tree/main. 

Datasets are `.npy` files of shape `(N, L)` for signals and `(N, C)` one-hot for labels.

### Sensor Positions

Cross-sensor transfer tasks are built by selecting one sensor position as the source domain and another as the target domain. Each position's signals are stored as separate `.npy` files.

**CWRU** — 3 sensor positions:

| Position | Code |
|---|---|
| Fan End | `FE` |
| Drive End | `DE` |
| Basement | `BA` |

Example transfer tasks: `FE -> DE`, `DE -> BA`, `BA -> FE`, etc.

**XJTU Spurgear** — 12 sensor positions:

| Position | Code |
|---|---|
| Channel 0 | `channel_0` |
| Channel 1 | `channel_1` |
| ... | ... |
| Channel 11 | `channel_11` |

Example transfer tasks: `channel_0 -> channel_1`, `channel_3 -> channel_7`, etc.

To run a cross-sensor task, point `--src_data/--src_label` to one position's files and `--dst_data/--dst_label` to another position's files.

## Quick Start

Run a single transfer task (src -> dst):

```run script
python main_framework/main.py `
  --src_data "data_path_of_src_domain" `
  --src_label "label_path_of_src_domain" `
  --dst_data "data_path_of_target_domain" `
  --dst_label "label_path_of_target_domain" `
  --save_path "./"
```

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--src_data` / `--src_label` | — | source domain `.npy` paths |
| `--dst_data` / `--dst_label` | — | target domain `.npy` paths |
| `--src_name` / `--dst_name` | source / target | short labels used in logs and summaries |
| `--batch_size` | 16 | batch size |
| `--nepoch` | 200 | max epochs |
| `--num_classes` | 5 | number of fault classes |
| `--w_c` | 1.0 | classification loss weight |
| `--w_d` | 1.0 | domain loss weight |
| `--w_v` | 1.0 | JVDR weight inside JVM |
| `--w_m` | 1.0 | JMMSD weight inside JVM |
| `--save_path` | ./ | output directory |

## Output

Each run writes to `--save_path`:

| File | Content |
|---|---|
| `train.log` | per-epoch metrics (loss, acc, P/R/F1, best so far) |
| `hparams.json` | full hyperparameter record |
| `summary.json` | final best/final accuracy, P/R/F1, total time |

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

