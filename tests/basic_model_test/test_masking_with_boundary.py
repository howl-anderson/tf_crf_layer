from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Embedding

from layer import CRF
from loss import crf_loss
from tests.common import get_random_data


def test_masking_with_boundary():
    nb_samples = 2
    timesteps = 10
    embedding_dim = 4
    output_dim = 5
    embedding_num = 12

    x, y = get_random_data(nb_samples, timesteps, x_high=embedding_num,y_high=output_dim)
    # right padding; left padding is not supported due to the tf.contrib.crf
    x[0, -4:] = 0

    # test with masking, fix length
    model = Sequential()
    model.add(Embedding(embedding_num, embedding_dim, input_length=timesteps,
                        mask_zero=True))
    model.add(CRF(output_dim, use_boundary=True))
    model.compile(optimizer='adam', loss=crf_loss)
    model.summary()
    model.fit(x, y, epochs=1, batch_size=1)
    model.fit(x, y, epochs=1, batch_size=2)
    model.fit(x, y, epochs=1, batch_size=3)
    model.fit(x, y, epochs=1)
