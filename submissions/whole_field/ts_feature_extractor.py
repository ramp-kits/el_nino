import numpy as np


class FeatureExtractor(object):

    def __init__(self):
        pass

    def transform(self, X_ds):
        """Compute the whole field at time t."""
        # This is the range for which features should be provided. Strip
        # the burn-in from the beginning.
        valid_range = np.arange(X_ds.n_burn_in, len(X_ds['time']))
        # Take the whole temperature field.
        all = X_ds['tas'].values
        # Vectorize it to obtain a single feature vector at time t.
        vectorized = all.reshape(len(all), -1)
        # Strip burn-in.
        X_array = vectorized[valid_range]
        return X_array
