import numpy as np

en_lat_bottom = -5
en_lat_top = 5
en_lon_left = 360 - 170
en_lon_right = 360 - 120


def get_area_mean(tas, lat_bottom, lat_top, lon_left, lon_right):
    """The array of mean temperatures in a region at all time points."""
    return tas.loc[:, lat_bottom:lat_top, lon_left:lon_right].mean(
        dim=('lat', 'lon'))


def get_enso_mean(tas):
    """The array of mean temperatures in the El Nino 3.4 region.

    At all time point.
    """
    return get_area_mean(
        tas, en_lat_bottom, en_lat_top, en_lon_left, en_lon_right)


class FeatureExtractor(object):

    def __init__(self):
        pass

    def transform(self, X_ds):
        """Anomaly and running mean.

        Compute the El Nino running mean at time t - (12 - X_ds.n_lookahead),
        corresponding the month to be predicted and the anomaly at time t
        (the difference between the temperature and the monthly running mean).

        The code is short but inefficient.
        """
        n_lookahead = X_ds.n_lookahead
        n_burn_in = X_ds.n_burn_in
        enso = get_enso_mean(X_ds['tas'])
        # The running monthly mean at time t (for every month).
        # It is using xarray's groupby at every t,
        # so its running time is O(T^2). In principle it can be computed in
        # O(T).
        running_monthly_means = np.array([
            enso.isel(time=slice(None, t)).groupby('time.month').mean(
                dim='time')
            for t in range(X_ds.n_burn_in, len(X_ds['tas']))])
        # The running monthly mean at time t
        # (corresponding to the current month).
        running_monthly_means_t0 = np.array(
            [running_monthly_means[t, t % 12]
             for t in range(len(running_monthly_means))])
        # The running monthly mean at time t
        # (corresponding to the month to be predicted).
        running_monthly_means_lookahead = np.array(
            [running_monthly_means[t, (t + n_lookahead) % 12]
             for t in range(len(running_monthly_means))])
        # The temperature anomaly at t0
        anomalys = enso.values[n_burn_in:] - running_monthly_means_t0
        X_array = np.array([running_monthly_means_lookahead, anomalys]).T
        return X_array