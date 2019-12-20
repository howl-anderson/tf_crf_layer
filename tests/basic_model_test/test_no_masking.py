import os

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.models import load_model

from tf_crf_layer.layer import CRF
from tf_crf_layer.loss import crf_loss, ConditionalRandomFieldLoss


def test_no_masking(get_random_data):
    nb_samples = 2
    timesteps = 10
    embedding_dim = 4
    output_dim = 5
    embedding_num = 12

    crf_loss_instance = ConditionalRandomFieldLoss()

    x, y = get_random_data(nb_samples, timesteps, x_high=embedding_num, y_high=output_dim)

    # test with no masking, fix length
    model = Sequential()
    model.add(Embedding(embedding_num, embedding_dim))
    model.add(CRF(output_dim, name="crf_layer"))
    model.compile(optimizer='adam', loss={"crf_layer": crf_loss_instance})

    model.fit(x, y, epochs=1, batch_size=1)
    model.fit(x, y, epochs=1, batch_size=2)
    model.fit(x, y, epochs=1, batch_size=3)
    model.fit(x, y, epochs=1)

    # test saving and loading model
    MODEL_PERSISTENCE_PATH = './test_saving_crf_model.h5'
    model.save(MODEL_PERSISTENCE_PATH)
    load_model(MODEL_PERSISTENCE_PATH,
               custom_objects={'CRF': CRF,
                               'crf_loss': crf_loss})

    try:
        os.remove(MODEL_PERSISTENCE_PATH)
    except OSError:
        pass
