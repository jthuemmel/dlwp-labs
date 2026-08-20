# Deep Learning for Weather Prediction: labs

The lab notebooks of the course and the package the labs fill in.
The first notebook asks a line of questions and names the objects the answers produce; the later ones state the pieces to build and check each one.
The code you write accumulates in `utils/`, so that by the end the repository holds a working forecasting pipeline.

## Structure

```
01_data_and_verification.ipynb   ERA5 in xarray, climatologies, tendencies, the persistence forecast, and its verification
02_dl_pipeline.ipynb             the training pipeline: Dataset, loss, Vision Transformer, LightningModule, and configuration
03_inductive_biases.ipynb        checkpoints and forecasts as xarray, then loss weights, roll-out training, per-variable embeddings, and the time of day
04_ensembles.ipynb               the empirical score, a noise input, and the verification of an ensemble
experiment.py                    lab 5: one training run of the project, from the command line, with its scores and figures
prepare_eval_data.py             lab 5: the evaluation store, the climatology, and the training statistics
utils/                           the package the labs fill in; empty modules for now
configs/                         the configuration files the notebooks run from
environment.yaml                 the conda environment every notebook runs in
```

Notebooks live in the top level and import from `utils`, so start Jupyter from the repository root.
The later notebooks appear here as the course reaches them.

## Environment

```
conda env create -f environment.yaml
conda activate dlwp
jupyter lab
```

The environment carries every lab from the start: xarray with zarr, dask, and gcsfs for the cloud stores, torch, lightning, and einops for the models, and matplotlib and cartopy for the figures.
Nothing is added per lab.
Solving it takes a few minutes; `mamba env create -f environment.yaml` is faster if you have mamba.

## Data

All data are read from public Google Cloud buckets with anonymous access, and every notebook lists the stores it uses:

- WeatherBench 2 (`gs://weatherbench2/`): ERA5 at 5.625 and 1.5 degrees, the 1990 to 2019 climatology, and the HRES forecasts.
- ARCO-ERA5 (`gs://gcp-public-data-arco-era5/`): ERA5 at its native 0.25 degrees, updated to about a week ago.

Opening a store is lazy and costs nothing; downloading follows the store's chunks.
Load a subset once, cache it with `to_netcdf` or `to_zarr`, and work from the cache.

## The project: experiment.py

`experiment.py` trains one configuration, tests it on 2017 to 2019, and writes the scores and the three figures the project brief asks for.

```
python experiment.py --config configs/baseline.yaml
python experiment.py --config configs/baseline.yaml configs/yours.yaml
python experiment.py --config configs/yours.yaml --seed 1
```

One configuration file is one model version.
`--seed` overrides the `seed` the file carries.
`Experiment("configs/yours.yaml", seed=1).run()` starts the same run from a notebook cell.

### The reporting years

The script trains on the years the configuration names, validates on 2016, and tests on 2017, 2018, and 2019.
The evaluation years come from a store of their own and are standardised with the statistics of your training years, so a test year enters no training statistic.
The run stops on a configuration that trains past 2015, drops Z500 or T850, or takes its statistics from the evaluation store.

### The data

`prepare_eval_data.py` writes the evaluation store for 2016 to 2019, the WeatherBench 2 climatology, and the statistics of your training period.
The first experiment calls it, and it also runs on its own:

```
python prepare_eval_data.py --config configs/baseline.yaml
```

It takes from your own store whatever it already covers and downloads the rest.
Your training data stay yours: the setup cell of lab 2 writes them, and nothing here downloads them.

The sixteen variables of `configs/baseline.yaml` cost about 3 GB of ERA5 on a first run.
The nine of labs 2 to 4 and the pair the report requires are already in your store, so only their climatology is fetched.
A climatology variable takes a few minutes on a slow line, because a chunk carries the full column, and it is fetched once for all your runs.

A forecast holds `predict_steps` states per initialisation, so the test epoch runs a year at a time.
At twice-daily initialisations and sixteen variables a run writes about 5 GB of forecasts, where the three test years of data are 570 MB.

### What a run writes

In `runs/<name>/seed<k>/`:

```
config.yaml              the configuration as it ran, seed included: --config on this file repeats the run
metrics.csv              the training and validation loss, step by step
scores.csv               the test scores: run, seed, variable, lead_hours, metric, value
run.json                 seed, device, parameter count, and the runtime the report states
model.ckpt               the trained weights
forecasts.zarr           the test forecasts, in the forecast schema of lab 1
loss_curves.png, forecast_t850_24h.png
```

`runs/rmse_z500_t850.png` holds every run under `runs/`, against persistence and the climatology.
The metrics are `rmse`, `rmse_climatology`, `rmse_persistence`, `skill`, and `acc`.
`Experiment.evaluation_plots` is empty and takes your own two figures.

### Seeds

A seed fixes the initialisation, the shuffling, and the noise draws.
A run repeats closely on the same machine and library versions, and not bit for bit across devices.
Several seeds are a loop, and the long form of `scores.csv` averages over them in one groupby:

```python
df = pd.concat(map(pd.read_csv, Path("runs").glob("*/*/scores.csv")))
df.groupby(["run", "variable", "lead_hours", "metric"]).value.agg(["mean", "std", "count"])
```

### What the script imports from utils

`Config`, `WeatherDataset`, `ForecastModule`, `forecasts_to_xarray`, `plot_metrics`, `count_parameters`, and from `utils/metrics.py` the functions `scores`, `rmse`, `acc`, and `truth_at`.
All of them are yours from labs 1 to 4.
If you named one of them differently, change the import at the top of the file.
