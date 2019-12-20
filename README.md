# tf_crf_layer
一个用于 TensorFlow 1.x 版本的 CRF keras layer

NOTE: tensorflow-addons 包含适用于 TensorFlow 2.0 版本的 CRF keras layer

## Functions
### Vanilla CRF

Ordinal liner chain CRF function.

* Support START/END transfer probability learning.
    * Which TensorFlow's `tf.contrib.crf` do not support

```python
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding, Bidirectional, LSTM

from tf_crf_layer.layer import CRF
from tf_crf_layer.loss import crf_loss
from tf_crf_layer.metrics import crf_accuracy

vocab = 3000
EMBED_DIM = 300
BiRNN_UNITS = 48
class_labels_number = 9
EPOCHS = 10

train_x, train_y = read_train_data()
test_x, test_y = read_test_data()

model = Sequential()
model.add(Embedding(len(vocab), EMBED_DIM, mask_zero=True)) 
model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
model.add(CRF(class_labels_number))

model.compile('adam', loss=crf_loss, metrics=[crf_accuracy])
model.fit(train_x, train_y, epochs=EPOCHS, validation_data=[test_x, test_y])
```

see [conll2000_chunking_crf](examples/conll2000_chunking_crf.py) for a real application of vanilla CRF

### CRF with static transfer constraint

User can pass a static transfer constraint to limit hidden states transfer probability.
Mainly used to greatly (not absolutely) reduce the probability of illegal hidden states sequence.
Such like  B -> B in BILUO tag schema.

Static transfer constraint is first introduced by AllenNLP. I also learned this technology from it.

```python
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding, Bidirectional, LSTM
from tf_crf_layer.crf_static_constraint_helper import allowed_transitions

from tf_crf_layer.layer import CRF
from tf_crf_layer.loss import crf_loss
from tf_crf_layer.metrics import crf_accuracy

vocab = 3000
EMBED_DIM = 300
BiRNN_UNITS = 48
class_labels_number = 9
EPOCHS = 10

tag_decoded_labels = get_tag_decoded_labels()
train_x, train_y = read_train_data()
test_x, test_y = read_test_data()

constraints = allowed_transitions("BIO", tag_decoded_labels)

model = Sequential()
model.add(Embedding(len(vocab), EMBED_DIM, mask_zero=True)) 
model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
model.add(CRF(class_labels_number, transition_constraint=constraints))

model.compile('adam', loss=crf_loss, metrics=[crf_accuracy])
model.fit(train_x, train_y, epochs=EPOCHS, validation_data=[test_x, test_y])
```

### CRF with dynamic transfer constraint

Dynamic transfer constraint is different from static transfer constraint that business logical may require apply some transfer constraint at running time.
For example, user have a intent or domain classifier which work pretty well.
User need implement a CRF based NER extractor. But not every entity of NER are illegal for that domain.
For example, location entity is illegal for music domain.
But user do not know such information at compile time, such information can only be get at running time from other component.

```python
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Embedding, Bidirectional, LSTM, Input
from tf_crf_layer.crf_dynamic_constraint_helper import generate_constraint_table

from tf_crf_layer.layer import CRF
from tf_crf_layer.loss import crf_loss
from tf_crf_layer.metrics import crf_accuracy

vocab = 3000
EMBED_DIM = 300
BiRNN_UNITS = 48
class_labels_number = 9
MAX_LEN = 24
intent_number = 2
EPOCHS = 10

tag_decoded_labels = get_tag_decoded_labels()
train_x, train_y = read_train_data()
test_x, test_y = read_test_data()
constraint_mapping = get_constraint_mapping()  # maping from intent to entity

constraint_table = generate_constraint_table(constraint_mapping, tag_decoded_labels)

raw_input = Input(shape=(MAX_LEN,))
embedding_layer = Embedding(vocab, EMBED_DIM, mask_zero=True)(raw_input)
bilstm_layer = Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True))(embedding_layer)

crf_layer = CRF(
    units=class_labels_number,
    transition_constraint_matrix=constraint_table
)

dynamic_constraint_input = Input(shape=(intent_number,))

output_layer = crf_layer([bilstm_layer, dynamic_constraint_input])

model = Model([raw_input, dynamic_constraint_input], output_layer)

# print model summary
model.summary()

model.compile('adam', loss=crf_loss, metrics=[crf_accuracy])
model.fit(train_x, train_y, epochs=EPOCHS, validation_data=[test_x, test_y])
```

# TODO
* Add more metric according to http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/