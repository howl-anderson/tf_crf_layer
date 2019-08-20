# tf_crf_layer
一个用于 TensorFlow 1.x 版本的 CRF keras layer

NOTE: tensorflow-addons 包含适用于 TensorFlow 2.0 版本的 CRF keras layer

## Functions
### Vanilla CRF

Ordinal liner chain CRF function.

* Support START/END transfer probability learning.
    * Which TensorFlow's `tf.contrib.crf` do not support

```python
from tf_crf_layer.layer import CRF
from tf_crf_layer.loss import crf_loss
from tf_crf_layer.metrics import crf_accuracy

model = Sequential()
model.add(Embedding(len(vocab), EMBED_DIM, mask_zero=True)) 
model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
model.add(CRF(class_labels_number))

model.compile('adam', loss=crf_loss, metrics=[crf_accuracy])
model.fit(train_x, train_y, epochs=EPOCHS, validation_data=[test_x, test_y])
```

### CRF with static transfer constraint

User can pass a static transfer constraint to limit hidden states transfer probability.
Mainly used to greatly (not absolutely) reduce the probability of illegal hidden states sequence.
Such like  B -> B in BILUO tag schema.

Static transfer constraint is first introduced by AllenNLP. I also learned this technology from it.

```python

```

### CRF with dynamic transfer constraint

Dynamic transfer constraint is different from static transfer constraint that business logical may require apply some transfer constraint at running time.
For example, user have a intent or domain classifier which work pretty well.
User need implement a CRF based NER extractor. But not every entity of NER are illegal for that domain.
For example, location entity is illegal for music domain.
But user do not know such information at compile time, such information can only be get at running time from other component.
