from keras.engine import InputSpec
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import Layer
import tensorflow as tf

from tf_crf_layer import keras_utils


@keras_utils.register_keras_custom_object
class MockMasking(Layer):
    def __init__(self, mask_shape, mask_value, **kwargs):
        super(MockMasking, self).__init__(**kwargs)

        # setup mask supporting flag, used by base class (the Layer)
        self.supports_masking = True

        self.mask_shape = mask_shape
        self.mask_value = mask_value

    def build(self, input_shape):
        input_shape = tuple(tf.TensorShape(input_shape).as_list())
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[-1]

        self.mask = self.add_weight(
            shape=self.mask_shape,
            name='transition_constraint_mask',
            initializer=initializers.Constant(self.mask_value),
            trainable=False
        )

        # or directly call self.built = True
        super(MockMasking, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        return inputs

    def get_config(self):
        # will be used for loading model from disk,
        # see https://github.com/keras-team/keras/issues/4871#issuecomment-269714512

        config = {
            'mask_shape': self.mask_shape,
            'mask_value': self.mask_value
        }
        base_config = super(MockMasking, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, input, mask=None):
        return self.mask
