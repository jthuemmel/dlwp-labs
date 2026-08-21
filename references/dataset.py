import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from utils.config import DatasetConfig


class WeatherData(Dataset):
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.seq_len = config.sequence_length
        self.variables = list(config.variables)

        ds = xr.open_zarr(config.path)

        self._parse_slices(config)
        ds = ds[self.variables].sel(time=self.time_slice, latitude=self.lat_slice, longitude=self.lon_slice)
        assert list(ds.to_array(dim="variable").coords["variable"].values) == self.variables, "Variable order mismatch"
        self.dataset = ds

        self.tensor_data = self._preprocess(self.dataset)
        self.time_hours = torch.from_numpy(((ds.time.dt.dayofyear.values - 1) * 24 + ds.time.dt.hour.values).astype(np.float32))

    @staticmethod
    def _slice_from_cfg(s):
        if isinstance(s, dict):
            return slice(s.get("start"), s.get("stop"), s.get("step"))
        elif isinstance(s, slice):
            return s
        else:
            return slice(None)

    def _parse_slices(self, config: DatasetConfig) -> None:
        self.time_slice = self._slice_from_cfg(config.time_slice)
        self.lat_slice = self._slice_from_cfg(config.lat_slice)
        self.lon_slice = self._slice_from_cfg(config.lon_slice)

    def _preprocess(self, data: xr.Dataset) -> torch.Tensor:
        self.compute_stats(data)
        self.compute_length(data)
        return self._to_tensor(self._standardize(data))

    def compute_stats(self, data: xr.Dataset) -> None:
        if self.config.stats_path is None:
            means = [data[key].mean().values for key in self.variables]
            stds = [data[key].std().values for key in self.variables]
        else:
            stats = xr.open_zarr(self.config.stats_path)
            means = [stats[key].sel(statistic="mean").values for key in self.variables]
            stds = [stats[key].sel(statistic="std").values for key in self.variables]
        self._means = xr.DataArray(np.float32(means), dims=["variable"], coords={"variable": self.variables}, name="mean")
        self._stds = xr.DataArray(np.float32(stds), dims=["variable"], coords={"variable": self.variables}, name="std")

    def compute_length(self, data: xr.Dataset) -> None:
        assert "time" in data.dims, "Dataset must have a time dimension"
        assert data.sizes["time"] > self.seq_len, "Dataset must have more time steps than the sequence length"
        self._length = data.sizes["time"] - self.seq_len

    def _standardize(self, data: xr.Dataset) -> xr.Dataset:
        return data.assign({key: (data[key] - self._means.sel(variable=key, drop=True)) / self._stds.sel(variable=key, drop=True)
                            for key in self.variables})

    @staticmethod
    def _to_tensor(data: xr.Dataset) -> torch.Tensor:
        data = data.fillna(0.0)
        array = data.to_array(dim="variable").transpose("variable", "time", "latitude", "longitude").values
        return torch.from_numpy(array).float().share_memory_()

    def to_xarray(self, x: torch.Tensor, **coords) -> xr.Dataset:
        x = np.asarray(x.detach().cpu())
        dims = ("variable", *coords.keys(), "latitude", "longitude")
        assert x.ndim == len(dims), f"expected the axes {dims}, got {x.ndim} axes"
        da = xr.DataArray(x, dims=dims, coords={
            "variable": self.variables, **{k: np.atleast_1d(v) for k, v in coords.items()},
            "latitude": self.dataset.latitude.values, "longitude": self.dataset.longitude.values})
        return (da * self._stds + self._means).to_dataset(dim="variable")

    def __len__(self):
        return self._length

    def __getitem__(self, idx: int) -> tuple:
        return self.tensor_data[:, idx: idx + self.seq_len], self.time_hours[idx]
