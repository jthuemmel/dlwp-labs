import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

xr.set_options(use_bottleneck=False)

ANON = {"token": "anon"}
ERA5 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
CLIM = "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr"
YEARS = ("2016", "2019")

SURFACE = {
    "T2M": "2m_temperature",
    "U10M": "10m_u_component_of_wind",
    "V10M": "10m_v_component_of_wind",
    "TP6h": "total_precipitation_6hr",
}
COLUMN = {
    "Z": "geopotential",
    "T": "temperature",
    "Q": "specific_humidity",
    "U": "u_component_of_wind",
    "V": "v_component_of_wind",
}


### HELPER FUNCTIONS
def wb2_name(var: str) -> tuple:
    # "T2M" -> ("2m_temperature", None), "Z500" -> ("geopotential", 500)
    if var in SURFACE:
        return SURFACE[var], None
    return COLUMN[var[0]], int(var[1:])


def select(store: xr.Dataset, var: str) -> xr.DataArray:
    name, level = wb2_name(var)
    da = store[name]
    return da.sel(level=level, drop=True) if level is not None else da


def covers(ds: xr.Dataset) -> bool:
    t = ds.time.values
    return t[0] <= np.datetime64(f"{YEARS[0]}-01-01") and t[-1] >= np.datetime64(f"{YEARS[1]}-12-31")


def write(data: xr.Dataset, path: Path, mode: str) -> None:
    # the cloud stores are zarr 2 and their codecs do not carry into a zarr 3 store
    for var in data.variables.values():
        var.encoding.clear()
    data.to_zarr(path, mode=mode)


def missing_variables(path: Path, variables: list) -> tuple:
    # a store that stops short of 2019 is rebuilt rather than patched
    have = xr.open_zarr(path) if path.exists() else None
    if have is not None and "time" in have.dims and not covers(have):
        have = None
    missing = list(variables) if have is None else [v for v in variables if v not in have.data_vars]
    return have, missing


### STORE BUILDERS
def eval_store(path, variables: list, source: str = None) -> Path:
    path = Path(path)
    have, missing = missing_variables(path, variables)
    if not missing:
        return path

    fields = {}
    local = xr.open_zarr(source) if source is not None and Path(source).exists() else None
    if local is not None and covers(local):
        for var in [v for v in missing if v in local.data_vars]:
            fields[var] = local[var].sel(time=slice(*YEARS))
    remote = [v for v in missing if v not in fields]

    if remote:
        store = xr.open_zarr(ERA5, storage_options=ANON)
        size = sum(select(store, v).sel(time=slice(*YEARS)).nbytes for v in remote) / 1e9
        print(f"downloading {', '.join(remote)} for {YEARS[0]} to {YEARS[1]}: {size:.1f} GB in memory, "
              f"more over the wire, since a level chunk carries the full column")
        for var in remote:
            fields[var] = select(store, var).sel(time=slice(*YEARS))

    data = xr.Dataset(fields).transpose("time", "latitude", "longitude").chunk({"time": 100})
    write(data, path, mode="a" if have is not None else "w")
    print(f"evaluation store {path}: {', '.join(variables)}")
    return path


def climatology_store(path, variables: list) -> Path:
    path = Path(path)
    have = xr.open_zarr(path) if path.exists() else None
    missing = list(variables) if have is None else [v for v in variables if v not in have.data_vars]
    if not missing:
        return path

    print(f"downloading the WeatherBench 2 1990 to 2019 climatology: {', '.join(missing)}")
    store = xr.open_zarr(CLIM, storage_options=ANON)
    data = xr.Dataset({var: select(store, var) for var in missing})
    data = data.transpose("hour", "dayofyear", "latitude", "longitude")
    write(data, path, mode="a" if have is not None else "w")
    print(f"climatology store {path}: {', '.join(variables)}")
    return path


def stats_store(path, source: str, variables: list, time_slice: dict) -> Path:
    # the statistics of the training period; a store written for another period is rebuilt
    path = Path(path)
    start, stop = (time_slice or {}).get("start"), (time_slice or {}).get("stop")
    period = f"{start}:{stop}"
    if path.exists():
        have = xr.open_zarr(path)
        if all(v in have.data_vars for v in variables) and have.attrs.get("period") == period:
            return path

    data = xr.open_zarr(source)[list(variables)].sel(time=slice(start, stop))
    stats = xr.concat([data.mean(), data.std()], dim=xr.DataArray(["mean", "std"], dims="statistic", name="statistic")).compute()
    stats.attrs["period"] = period
    write(stats, path, mode="w")
    print(f"statistics store {path}: {data.sizes['time']} states of {period}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Build the evaluation stores a configuration names")
    parser.add_argument("--config", type=str, required=True, help="path to the config file")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    settings, dataset = cfg.get("experiment", {}), cfg["dataset"]
    eval_store(settings["eval_path"], dataset["variables"], source=dataset["path"])
    climatology_store(settings["climatology_path"], dataset["variables"])
    stats_store(dataset["stats_path"], dataset["path"], dataset["variables"], dataset.get("time_slice"))


if __name__ == "__main__":
    main()
