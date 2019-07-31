# tf_crf_layer
一个用于 TensorFlow 1.x 版本的 CRF keras layer

NOTE: tensorflow-addons 包含适用于 TensorFlow 2.0 版本的 CRF keras layer

# 使用
## 一个 BiLSTM + CRF 模型
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