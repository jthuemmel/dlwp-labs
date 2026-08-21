import numpy as np
import xarray as xr

SPACE = ("latitude", "longitude")


def latitude_weights(field: xr.Dataset) -> xr.DataArray:
    w = np.cos(np.deg2rad(field.latitude))
    return w / w.mean()


def truth_at(era5: xr.Dataset, forecast: xr.Dataset) -> xr.Dataset:
    valid = forecast.time + forecast.prediction_timedelta
    return era5[list(forecast.data_vars)].rename(time="valid_time").sel(valid_time=valid)


def climatology_at(climatology: xr.Dataset, forecast: xr.Dataset) -> xr.Dataset:
    valid = forecast.time + forecast.prediction_timedelta
    return climatology[list(forecast.data_vars)].sel(hour=valid.dt.hour, dayofyear=valid.dt.dayofyear)


def ensemble_mean(forecast: xr.Dataset, member: str = "number") -> xr.Dataset:
    return forecast.mean(member) if member in forecast.dims else forecast


def rmse(forecast: xr.Dataset, truth: xr.Dataset) -> xr.Dataset:
    weights = latitude_weights(forecast)
    return np.sqrt(((forecast - truth) ** 2).weighted(weights).mean(SPACE))


def mae(forecast: xr.Dataset, truth: xr.Dataset) -> xr.Dataset:
    weights = latitude_weights(forecast)
    return np.abs(forecast - truth).weighted(weights).mean(SPACE)


def bias(forecast: xr.Dataset, truth: xr.Dataset) -> xr.Dataset:
    weights = latitude_weights(forecast)
    return (forecast - truth).weighted(weights).mean(SPACE)


def activity(field: xr.Dataset, climatology: xr.Dataset) -> xr.Dataset:
    weights = latitude_weights(field)
    return np.sqrt(((field - climatology) ** 2).weighted(weights).mean(SPACE))


def acc(forecast: xr.Dataset, truth: xr.Dataset, climatology: xr.Dataset) -> xr.Dataset:
    weights = latitude_weights(forecast)
    f, t = forecast - climatology, truth - climatology
    covariance = (f * t).weighted(weights).mean(SPACE)
    return covariance / np.sqrt((f ** 2).weighted(weights).mean(SPACE) * (t ** 2).weighted(weights).mean(SPACE))


def skill_score(score: xr.Dataset, reference: xr.Dataset) -> xr.Dataset:
    return 1 - score / reference


def information_noise(forecast: xr.Dataset, truth: xr.Dataset, climatology: xr.Dataset) -> xr.Dataset:
    forecast_activity = activity(forecast, climatology)
    true_activity = activity(truth, climatology)
    correlation = acc(forecast, truth, climatology)
    error = rmse(forecast, truth)
    information = forecast_activity * correlation
    information_error = np.abs(true_activity - information)
    noise_error = np.sqrt(np.maximum(error ** 2 - information_error ** 2, 0))
    names = ["activity", "true_activity", "acc", "rmse", "information", "information_error", "noise_error"]
    quantities = [forecast_activity, true_activity, correlation, error, information, information_error, noise_error]
    return xr.concat([q.reset_coords(drop=True) for q in quantities],
                     dim=xr.DataArray(names, dims="quantity", name="quantity"))


def crps(ensemble: xr.Dataset, truth: xr.Dataset, member: str = "number", fair: bool = True) -> xr.Dataset:
    m = ensemble.sizes[member]
    coefficient = 1 / (m * (m - 1)) if fair else 1 / m ** 2
    absolute_error = np.abs(ensemble - truth).mean(member)
    pairs = np.abs(ensemble - ensemble.rename({member: f"{member}_"})).sum((member, f"{member}_")) / 2
    weights = latitude_weights(ensemble)
    return (absolute_error - coefficient * pairs).weighted(weights).mean(SPACE)


def spread_skill_ratio(ensemble: xr.Dataset, truth: xr.Dataset, member: str = "number") -> xr.Dataset:
    weights = latitude_weights(ensemble)
    m = ensemble.sizes[member]
    spread = np.sqrt(ensemble.var(member, ddof=1).weighted(weights).mean(SPACE))
    skill = np.sqrt(((truth - ensemble.mean(member)) ** 2).weighted(weights).mean(SPACE))
    return np.sqrt((m + 1) / m) * spread / skill


def rank_histogram(ensemble: xr.DataArray, truth: xr.DataArray, member: str = "number") -> np.ndarray:
    m = ensemble.sizes[member]
    members = ensemble.transpose(..., member).values.reshape(-1, m)
    observation = truth.values.reshape(-1, 1)
    valid = ~np.isnan(members).any(-1) & ~np.isnan(observation[:, 0])
    members, observation = members[valid], observation[valid]
    return np.bincount(np.sum(members < observation, axis=-1), minlength=m + 1) / len(members)
