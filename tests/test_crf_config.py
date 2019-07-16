from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Embedding

from layer import CRF
from loss import crf_loss
from tests.common import get_random_data


def test_crf_config():
    nb_samples = 2
    timesteps = 10
    embedding_dim = 4
    output_dim = 5
    embedding_num = 12

    x, y = get_random_data(nb_samples, timesteps, x_high=embedding_num,
                           y_high=output_dim)
    # right padding; left padding is not supported due to the tf.contrib.crf
    x[0, -4:] = 0

    # test with masking, fix length
    model = Sequential()
    model.add(Embedding(embedding_num, embedding_dim, input_length=timesteps,
                        mask_zero=True))
    model.add(CRF(output_dim))
    model.compile(optimizer='rmsprop', loss=crf_loss)
    model.summary()
    model.fit(x, y, epochs=1, batch_size=10)

    # test config
    result = model.get_config()

    print(repr(result))

    expected = {
        'name': 'sequential',
        'layers': [
            {'class_name': 'Embedding',
             'config': {
                 'name': 'embedding',
                 'trainable': True,
                 'batch_input_shape': (
                     None, 10),
                 'dtype': 'float32',
                 'input_dim': 12,
                 'output_dim': 4,
                 'embeddings_initializer': {
                     'class_name': 'RandomUniform',
                     'config': {
                         'minval': -0.05,
                         'maxval': 0.05,
                         'seed': None,
                         'dtype': 'float32'}},
                 'embeddings_regularizer': None,
                 'activity_regularizer': None,
                 'embeddings_constraint': None,
                 'mask_zero': True,
                 'input_length': 10}},
            {'class_name': 'CRF',
             'config': {'name': 'crf',
                        'trainable': True,
                        'dtype': 'float32',
                        'units': 5,
                        'learn_mode': 'join',
                        'test_mode': 'viterbi',
                        'use_boundary': False,
                        'use_bias': True,
                        'sparse_target': False,
                        'kernel_initializer': {
                            'class_name': 'GlorotUniform',
                            'config': {
                                'seed': None,
                                'dtype': 'float32'}},
                        'chain_initializer': {
                            'class_name': 'Orthogonal',
                            'config': {
                                'gain': 1.0,
                                'seed': None,
                                'dtype': 'float32'}},
                        'boundary_initializer': {
                            'class_name': 'Zeros',
                            'config': {
                                'dtype': 'float32'}},
                        'bias_initializer': {
                            'class_name': 'Zeros',
                            'config': {
                                'dtype': 'float32'}},
                        'activation': 'linear',
                        'kernel_regularizer': None,
                        'chain_regularizer': None,
                        'boundary_regularizer': None,
                        'bias_regularizer': None,
                        'kernel_constraint': None,
                        'chain_constraint': None,
                        'boundary_constraint': None,
                        'bias_constraint': None,
                        'input_dim': 4,
                        'unroll': False}
             }
        ]
    }


if __name__ == "__main__":
    test_crf_config()
