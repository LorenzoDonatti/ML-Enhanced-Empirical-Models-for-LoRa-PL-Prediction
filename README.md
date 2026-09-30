# Machine Learning Enhanced Empirical Models for LoRa Path-Loss Prediction

**Companion implementation for the research manuscript by:**

Lorenzo Moreira Donatti¹, Tiago Bandeira Marchesan¹, José Batista Ferreira Neto²,
Daliane Lanzarin², Diogo Damasceno do Espirito Santo², and Carlos Henrique Barriquello¹.

¹ Federal University of Santa Maria (UFSM), Santa Maria, Brazil.  
² Santo Antônio Energia (SAE), Porto Velho, Brazil.

**Corresponding author:** Carlos Henrique Barriquello — carlos.barriquello@ufsm.br.

This repository provides a compact implementation of the main numerical experiments
in *Machine Learning Enhanced Empirical Models for LoRa Path-Loss Prediction*.
It compares classical propagation models, direct machine learning, and hybrid
residual learning on public urban LoRa measurements from Bonn, Germany.

The implementation consists of **four Python files**. It downloads the public data,
trains the models, and exports numerical results as CSV files. It is intended as an
accessible research baseline for studying analytical models combined with machine learning.
Numerical diagnostics include MAE, R², a training-mean baseline, paired cluster-bootstrap
intervals, and distance-bin errors; no plotting dependencies are needed.

## Research overview

An empirical path-loss model describes a large-scale propagation trend. A residual
learner estimates the difference between that prediction and the measured loss:

```text
training residual = measured path loss − classical prediction
hybrid prediction = classical prediction + predicted residual
```

Three learners—Random Forest (RF), XGBoost, and distance-weighted k-nearest
neighbors (KNN)—estimate this correction independently. Their predictions are
combined with equal weights. Direct ensembles use the same learner families to
predict path loss directly.

The main question is whether the benefit of residual learning persists when the
models receive comparable receiver information and are tested farther from their
training locations. The experiments compare gateway-height inputs, geographic
exclusion, training sample counts, and transfer to an unseen gateway.

The manuscript finds very similar direct and residual errors under local
interpolation once height is available to both. Larger residual advantages appear
under spatial exclusion and receiver holdout. These findings concern the measured
Bonn deployment; they do not establish transfer to another city or frequency band.

## Getting started

Use Python 3.12 or newer. Run the commands from this repository's root directory.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

Start with one split and one seed:

```bash
python run.py --study controls --protocols random --seeds 0 --output results/quickstart
```

This evaluates a training-mean baseline and trains all seven controlled-input configurations on the **complete dataset**,
using one random partition. It is a smaller experiment, not a subsample of the data.
The first run downloads `samples.csv` and `gateways.csv` into `data/`.

Run all implemented experiments with the default five seeds:

```bash
python run.py
```

The complete run trains hundreds of models and may take hours depending on the
machine. Four model threads are requested by default. `--jobs` changes that setting.
The required packages are pinned in `requirements.txt`; numerical differences
across platforms or dependency versions remain possible.

## What does a run produce?

### 1. Progress and results in the terminal

For the quick-start command, each progress line reports the **ensemble's test RMSE
in dB** for one configuration. Lower is better. Example values from seed zero:

```text
131,057 receptions; 9 gateways
random seed=0  direct_basic: 6.4349 dB
random seed=0  direct_height: 6.1214 dB
random seed=0  direct_anchor: 6.1638 dB
random seed=0  oh_basic: 6.2923 dB
random seed=0  oh_height: 6.1222 dB
random seed=0  ldpl_basic: 6.4256 dB
random seed=0  oh_continuous: 6.2908 dB
```

After fitting, the terminal also prints an aggregate table containing the baseline,
each individual learner, and its ensemble, followed by the output directory.
This command produces 29 metric rows (one constant baseline plus 28 learned-model
results) and six paired bootstrap comparisons.
These seed-zero values are not the five-seed means reported in the manuscript.

### 2. CSV files

```text
results/quickstart/
├── runs.csv
├── summary.csv
├── comparisons.csv
└── distance_errors.csv
```

Gateway experiments additionally generate `gateways.csv`.

| File | Contents | Typical use |
|---|---|---|
| `runs.csv` | RMSE, MAE, and R² per component or ensemble, per configuration and partition; also classical/linear baselines when requested | Compare individual fits and inspect selected KNN settings |
| `summary.csv` | Mean, sample standard deviation, and valid-run count for RMSE, MAE, and R² per protocol/configuration/model | Compare repeated-partition results |
| `comparisons.csv` | Seed-zero paired cluster-bootstrap differences in RMSE and conditional 95% intervals | Assess uncertainty in the paper comparisons |
| `distance_errors.csv` | Seed-zero MAE, RMSE, and reception counts in 0.5-km distance intervals | Inspect distance-dependent errors without plots |
| `gateways.csv` | Macro and pooled test RMSE per configuration/model across held-out receivers | Compare receiver transfer |

For example, an ensemble row in `runs.csv` identifies `protocol=random`, `seed=0`,
`config=oh_height`, `model=ensemble`, and test RMSE approximately **6.1222 dB**.
That partition contains **98,292 training**, **13,106 validation**, and **19,659 test**
receptions. The same configuration has separate `rf`, `xgboost`, and `knn` rows.

| Column in `runs.csv` | Meaning |
|---|---|
| `protocol` | `random`, `packet`, `spatial_500m`, `buffer250`, `size250`, `gateway`, etc. |
| `seed` | Random seed for the partition and learner fitting |
| `gateway` | Held-out receiver identifier; empty for other experiments |
| `config` | Feature/target configuration, listed below; for classical baselines, the anchor name |
| `model` | `rf`, `xgboost`, `knn`, `ensemble`, `linear`, `classical`, or `constant` |
| `train_rows`, `validation_rows`, `test_rows` | Number of receptions in each partition |
| `validation_rmse_db` | Component validation RMSE; not emitted for ensembles or classical/linear rows |
| `test_rmse_db` | Test RMSE of the final path-loss predictions, including the anchor for residual models |
| `test_mae_db` | Mean absolute test error in dB |
| `test_r2` | Test coefficient of determination; undefined for fewer than two observations or a constant target |
| `selected_k` | Validation-selected neighbor count for KNN; empty for other models |
| `status` | `infeasible` for a partition with fewer than 100 training or validation rows; otherwise empty |

Infeasible rows use `config=infeasible` and `model=none`, with no error estimate.
They remain visible in `runs.csv` and are excluded from the summary.

`summary.csv` retains `protocol,config,model,mean,std,count` for RMSE and adds
`mae_db_mean`, `mae_db_std`, `mae_db_count`, `r2_mean`, `r2_std`, and `r2_count`.
Each metric is aggregated across runs; this is not a pooled R². Undefined R² values
are excluded from its mean and count. For the quick-start
run, `count=1` and `std` is empty (`NaN` in the terminal): one observation has no
sample standard deviation. For the default repeated-split study, count is normally
five, or four for the feasible 1,000-m runs. Gateway summaries average nine receivers;
their standard deviation describes differences between receivers, not between seeds.

In `gateways.csv`, **macro RMSE** averages the receiver-specific RMSE values with
equal weight per gateway. **Pooled RMSE** weights squared errors by the number of
test receptions before taking the square root.

The scripts save aggregate errors, not per-reception predictions or trained models.
CSV files can be opened in Excel, LibreOffice, or pandas. For example:

```python
import pandas as pd

summary = pd.read_csv("results/quickstart/summary.csv")
print(summary.loc[summary["model"].eq("ensemble")])
```

### 3. Statistical and distance diagnostics

Every feasible partition includes `config=training_mean`, `model=constant`, which
predicts the training-target mean for all test receptions. It never uses test or
validation targets to set the prediction. Gateway experiments also report
uncorrected OH alongside the learned models.

`comparisons.csv` uses **2,000 paired cluster-bootstrap replicates**, with RNG seed
zero, for fits trained on split seed zero. Random-row and packet-grouped tests
resample proxy packet groups; spatial and buffered tests resample held-out 500-m
tiles defined using the full campaign grid. Entire clusters are sampled with
replacement, and both models receive exactly the same sampled clusters. No models
are refitted. The 2.5th and 97.5th percentiles give the conditional 95% interval.

The comparison columns identify configurations/models A and B. `point_db` is
**RMSE(A) − RMSE(B)**; negative values favor A. `lower_db` and `upper_db` are the
interval endpoints. `clusters` and `repetitions` document the resampling size.
The implemented comparisons, when both fits are present, are:

- Initial OH ensemble versus each of its RF, XGBoost, and KNN components.
- Height-augmented OH ensemble versus the height-augmented direct ensemble.
- Initial OH ensemble versus the height-augmented direct ensemble.
- Initial OH ensemble versus the basic direct ensemble.

These intervals describe test-cluster variation conditional on fitted models and
chosen hyperparameters. They do not replace variation across training seeds,
include no multiple-comparison adjustment, and are not equivalence tests.
No bootstrap is emitted for sample-count controls or gateway holdouts.
If seed zero or a required pair is absent, the corresponding comparisons are absent;
a file with only its header means no eligible comparisons were run.

`distance_errors.csv` covers every evaluated model for seed zero, including gateway
holdouts. Bins are half-open `[lower_km, upper_km)` intervals of width 0.5 km.
`count` is the number of test receptions; `mae_db` and `rmse_db` describe errors
within that bin. Empty bins are omitted, and sparse bins are retained so their
limited support remains visible. For the paper's distance analysis, select
`protocol=packet`, `config=oh_basic`, and bins with `count >= 100`.

## Experiment options

| `--study` | Experiments |
|---|---|
| `all` (default) | All experiments below, with duplicate configurations fitted once per main partition |
| `controls` | Seven configurations comparing direct/residual targets, gateway height, an OH input feature, and distance resolution |
| `anchors` | Five classical baselines, linear controls, and residual RF/XGBoost/KNN ensembles for each anchor |
| `reference` | Classical/linear baselines, the initial OH ensemble, and its distance-only ablation |
| `buffers` | Fixed spatial test tiles with 250-, 500-, and 1,000-m exclusion radii |
| `size` | Random subsets of the original development sets matching buffered training/validation counts |
| `gateway` | Nine receiver holdouts comparing direct+height, residual+height, and the initial OH ensemble |

```bash
python run.py --study anchors --output results/anchors
python run.py --study reference --output results/reference
python run.py --study buffers --radii 250 500 1000 --output results/buffers
python run.py --study size --output results/size
python run.py --study gateway --output results/gateway
```

The main protocols are:

- **`random`:** individual gateway receptions are assigned independently.
- **`packet`:** receptions with the same date/packet/position key remain together.
- **`spatial_500m`:** complete transmitter-position tiles remain together. A
  500-m tile does not guarantee a 500-m distance from the nearest training point.

Buffered experiments preserve each seed's original spatial test set and remove
nearby training and validation observations. Size controls match the retained row
counts without geographic exclusion. Seed two at 1,000 m has only 87 validation
receptions and is reported as infeasible.

`--seeds 0 1 2 3 4` and `--protocols random packet spatial_500m` select the main
partitions. Buffered and size studies always use the spatial partition;
`--protocols` does not alter them. Gateway holdouts always use seed zero and a
packet-grouped validation subset of the remaining receivers, independent of
`--seeds` and `--protocols`.

Other options: `--jobs 4`, `--data data`, and `--output results`.
See `python run.py --help` for the complete command-line interface.

**Each execution overwrites its output files.** Use separate output directories
for separate experiments. `runs.csv`, `comparisons.csv`, and `distance_errors.csv` are written progressively; summaries are
created after a successful run. There is no automatic resume mechanism. When
reusing a directory, remove older outputs first so stale summaries are not mistaken
for the results of an interrupted run.

## Model configurations

The basic inputs are transmitter latitude, longitude, and link distance. Trees use
distance rounded to 10 m; KNN and linear regression use log-distance. KNN and
linear regression use training-fitted standardization.

| Configuration | Target | Learner inputs / anchor |
|---|---|---|
| `direct_basic` | Measured path loss | Basic inputs |
| `direct_height` | Measured path loss | Basic inputs + gateway height |
| `direct_anchor` | Measured path loss | Basic inputs + OH prediction |
| `oh_basic` | OH residual | Basic inputs; OH added back at prediction |
| `oh_height` | OH residual | Basic inputs + gateway height; OH added back |
| `ldpl_basic` | Bonn LDPL residual | Basic inputs; training-fitted LDPL added back |
| `oh_continuous` | OH residual | Tree distance at 1-m resolution; KNN unchanged |
| `oh_distance_only` | OH residual | Distance/log-distance only; OH anchor unchanged |
| `fspl_residual` | FSPL residual | Basic inputs; FSPL added back |
| `oulu_residual` | Oulu LDPL residual | Basic inputs; Oulu LDPL added back |
| `winner_residual` | Winner+ residual | Basic inputs; Winner+ added back |

RF and XGBoost use 100 trees. KNN selects k from {5, 10, 20, 50, 100} using
validation RMSE. Ensemble predictions are averaged with weights of one third
**before** RMSE is calculated. The nominal main split proportions are 75/10/15;
grouped partitions can have different reception counts. Bonn LDPL is fitted on
training distance-bin medians only. SVR is not part of this implementation.

## Reference results from the manuscript

The following values are comparison targets, not hard-coded model outputs.
RMSE is in dB. Split results are means over five seeds, except four feasible runs
at 1,000 m; receiver results use nine holdouts.

| Evaluation | Direct + height | OH residual + height |
|---|---:|---:|
| Random receptions | 6.19 | 6.19 |
| Packet groups | 6.19 | 6.19 |
| Spatial tiles, no buffer | 7.07 | 7.04 |
| 250-m exclusion | 7.93 | 7.83 |
| 500-m exclusion | 8.76 | 8.46 |
| 1,000-m exclusion | 11.77 | 10.19 |
| Unseen gateway, macro | 9.84 | 9.34 |
| Unseen gateway, pooled | 10.63 | 9.72 |

The original OH ensemble without explicit height input to its learners yields
approximately 6.35, 6.35, and 7.52 dB under random, packet and spatial partitions.

This compact companion implements the main RMSE comparisons, MAE/R² metrics,
training-mean baseline, conditional paired bootstrap intervals, and numerical
distance-bin errors. Publication figures and plotting are outside its scope. The reference values are rounded; seed selection and the
execution environment matter when comparing results.

## Repository structure

```text
.
├── data.py              # Download, preprocessing, classical models, partitions
├── models.py            # Learners and equal direct/residual ensembles
├── experiments.py       # Experimental configurations and validation studies
├── run.py               # Command-line entry point and numerical output
├── requirements.txt
└── README.md
```

`data/` and `results/` are created at runtime. No manuscript files, image assets,
archived results, or dataset copies are required in the repository.

## Dataset and attribution

The public [Bonn LoRa survey](https://github.com/mclab-hbrs/lora-bonn) contains
131,057 received-link records at 868 MHz from nine gateways. The downloader uses
upstream revision `74d9e29a55eae6ba0909738e9f623f6e7219e902`. For offline execution,
place `samples.csv` and `gateways.csv` in the directory supplied through `--data`.

The path-loss target follows the source processing:

```text
received packet power (dBm) = RSSI + min(SNR, 0)
path loss (dB)              = 15 − received packet power
```

Only successful receptions are available. The timestamp has date precision,
so date/packet/position groups are proxy transmission identifiers. The dataset
supports received-link prediction, not direct estimation of packet delivery ratio.

**Original dataset publication:** Michael Rademacher, Hendrik Linka, Thorsten
Horstmann, and Martin Henze, “Path Loss in Urban LoRa Networks: A Large-Scale
Measurement Study,” IEEE VTC2021-Fall,
[DOI: 10.1109/VTC2021-Fall52928.2021.9625531](https://doi.org/10.1109/VTC2021-Fall52928.2021.9625531).

The data are distributed under
[Data licence Germany – attribution – version 2.0](https://www.govdata.de/dl-de/by-2-0).
Retain the original dataset attribution when using or redistributing the data.
This data license does not assign a license to the companion implementation.

## Citing this work

Please cite the companion manuscript and the original dataset publication when
using these experiments. The entry below identifies the manuscript; it does not
claim journal publication or assign a DOI. It will be Replaced with the final bibliographic
record when available.

```bibtex
@misc{donatti_lora_pathloss,
  author = {Donatti, Lorenzo Moreira and Marchesan, Tiago Bandeira and
            Ferreira Neto, José Batista and Lanzarin, Daliane and
            do Espirito Santo, Diogo Damasceno and Barriquello, Carlos Henrique},
  title = {Machine Learning Enhanced Empirical Models for {LoRa} Path-Loss Prediction},
  note = {Research manuscript; companion numerical implementation}
}
```

## Acknowledgments

This work was supported by the Coordenação de Aperfeiçoamento de Pessoal de Nível
Superior (CAPES), Brazil — Finance Code 001; the Federal University of Santa Maria;
and Santo Antônio Energia through ANEEL R&D project PD-06683-0123.

The authors acknowledge the Bonn survey authors for making their measurement
data publicly available.
