import numpy as np


def get_random_data(nb_samples, timesteps, x_low=1, x_high=12, y_low=0,
                    y_high=5):
    x = np.random.randint(low=x_low, high=x_high,
                          size=nb_samples * timesteps)
    x = x.reshape((nb_samples, timesteps))
    # x[0, -4:] = 0  # right padding
    # x[1, :5] = 0  # left padding, currently not supported by crf layer

    y = np.random.randint(low=y_low, high=y_high, size=nb_samples * timesteps)
    y = y.reshape((nb_samples, timesteps))

    return x, y
