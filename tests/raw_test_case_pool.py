# test with masking, sparse target, dynamic length;
# test crf_viterbi_accuracy, crf_marginal_accuracy
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Embedding

from layer import CRF
from loss import crf_loss
from metrics import crf_viterbi_accuracy, crf_marginal_accuracy

nb_samples = 2
timesteps = 10
embedding_dim = 4
output_dim = 5
embedding_num = 12

x = np.random.randint(1, embedding_num, nb_samples * timesteps)
x = x.reshape((nb_samples, timesteps))
x[0, -4:] = 0  # right padding
x[1, :5] = 0  # left padding
y = np.random.randint(0, output_dim, nb_samples * timesteps)
y = y.reshape((nb_samples, timesteps))
# y_onehot = np.eye(output_dim)[y]
# y = np.expand_dims(y, 2)  # .astype('float32')

model = Sequential()
model.add(Embedding(embedding_num, embedding_dim, mask_zero=True))
crf = CRF(output_dim, sparse_target=True)
model.add(crf)
# model.compile(optimizer='rmsprop', loss=crf_loss,
#               metrics=[crf_viterbi_accuracy, crf_marginal_accuracy])
model.compile(optimizer='adam', loss=crf_loss)
model.fit(x, y, epochs=1, batch_size=10)

# check mask
y_pred = model.predict(x).argmax(-1)
assert (y_pred[0, -4:] == 0).all()  # right padding
assert (y_pred[1, :5] == 0).all()  # left padding

# test viterbi_acc
_, v_acc, _ = model.evaluate(x, y)
np_acc = (y_pred[x > 0] == y[:, :, 0][x > 0]).astype('float32').mean()
print(v_acc, np_acc)
assert np.abs(v_acc - np_acc) < 1e-4

# test config
model.get_config()
