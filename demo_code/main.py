import tensorflow as tf
from tensorflow.python.keras.layers import Embedding, Bidirectional, LSTM
from tensorflow.python.keras.models import Sequential

from demo_code.input_data import get_input_data
from tf_crf_layer.layer import CRF
from tf_crf_layer.loss import crf_loss
from tf_crf_layer.metrics import crf_accuracy

config, (train_x, train_y), (test_x, test_y), tag_lookup, vocabulary_lookup = get_input_data()

EPOCHS = 10
EMBED_DIM = 64
BiRNN_UNITS = 200

vacab_size = vocabulary_lookup.size()
tag_size = tag_lookup.size()

model = Sequential()
model.add(Embedding(vacab_size, EMBED_DIM, mask_zero=True))
model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
model.add(CRF(tag_size))

# print model summary
model.summary()

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=config['model_dir'])

model.compile('adam', loss=crf_loss, metrics=[crf_accuracy])
model.fit(
    train_x, train_y,
    epochs=EPOCHS,
    validation_data=[test_x, test_y],
    callbacks=[tensorboard_callback]
)

tf.keras.experimental.export_saved_model(model, config['saved_model_dir'])
