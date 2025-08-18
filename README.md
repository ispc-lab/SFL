# Stretching the Limits: A Generalizable Strategy for Out-of-Distribution Molecular Property Prediction

## Overview
**Stretching Features and Labels (SFL)** is a general-purpose, architecture-agnostic training strategy designed to improve OOD generalization across molecular modeling tasks. SFL generates pseudo-OOD samples by extrapolating latent molecular representations and proportionally adjusting associated property labels along semantically meaningful directions, thereby expanding both structural and property coverage without violating chemical validity. It integrates seamlessly into existing pipelines for graph-based, transformer-based, and other architectures, with minimal computational overhead.

## News
### December 2024
🏆 **Championship Award**: SFL achieved 1st place (out of 226 teams) in [The Second Global AI Drug Development Algorithm Competition](https://aistudio.baidu.com/competition/detail/1214/0/leaderboard)!


## Getting Started

### Environment Setup
Create and activate Conda environment:
```bash
conda env create -f environment.yml
conda activate SFL
```

### Model Training
#### Base Models
**Molecular Property Prediction**:
```bash
python train_MP.py --use_SFL 0 --arch_type unimol --dataset esol \
                   --save_dir ./ckpt_MP/base --device cuda:0
```
*Datasets*: `esol`, `freesolv`, `lipo`, `qm7`, `qm9_homo`, `qm9_lumo`, `qm9_gap`  
*Models*: `attentive_fp`, `schnet`, `egnn`, `dimenet++`, `visnet`, `gem`, `unimol`, `unimol2_84M`

**Protein-Ligand Binding Affinity Prediction**:
```bash
python train_LBA.py --use_SFL 0 --arch_type atom3d_gnn --dataset LBA_30 \
                    --save_dir ./ckpt_LBA30/base --device cuda:0
```
*Datasets*: `LBA_30`, `LBA_60`  
*Models*: `deepdta`, `moltrans`, `atom3d_cnn3d`, `atom3d_gnn`, `comenet`, `visnet`

**Protein-Protein Interaction Prediction (PPB)**:
```bash
python train_PPB.py --use_SFL 0 --arch_type dg_model --dataset PPB \
                    --save_dir ./ckpt_PPB/base --device cuda:0
```
*Datasets*: `PPB`  
*Models*: `dg_model`

#### Models with SFL
**Molecular Property Prediction**:
```bash
python train_MP_SFL.py --use_SFL 1 --arch_type unimol --dataset esol \
                       --save_dir ./ckpt_MP/SFL --device cuda:0
```

**Protein-Ligand Binding Affinity Prediction**:
```bash
python train_LBA_SFL.py --use_SFL 1 --arch_type atom3d_gnn --dataset LBA_30 \
                        --save_dir ./ckpt_LBA30/SFL --device cuda:0
```

**Protein-Protein Interaction Prediction**:
```bash
python train_PPB_SFL.py --use_SFL 1 --arch_type dg_model --dataset PPB \
                        --save_dir ./ckpt_PPB/SFL --device cuda:0
```

### Key Parameters
| Parameter    | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `--dataset`  | Target dataset (e.g., `esol`, `LBA_30`, `PPB`)                             |
| `--save_dir` | Output directory for model checkpoints                                      |
| `--device`   | Training device (`cuda:0`, `cpu`, etc.)                                     |
| `--use_SFL`  | Enable SFL strategy (`0` = base model, `1` = SFL-enhanced)                 |
| `--arch_type`| Model architecture (see dataset-specific options above)                    |

### Evaluation Metrics
All models report:
- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)

SFL-enhanced models additionally provide separate **in-distribution (ID)** and **out-of-distribution (OOD)** evaluations.