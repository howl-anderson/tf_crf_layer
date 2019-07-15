import tensorflow as tf
from tensorflow.python.keras import initializers, regularizers, constraints, \
    activations
from tensorflow.python.keras.layers import InputSpec, Layer


class CRF(Layer):
    def __init__(self, units,
                 learn_mode='join',
                 test_mode=None,
                 sparse_target=False,
                 use_boundary=False,
                 use_bias=True,
                 activation='linear',
                 kernel_initializer='glorot_uniform',
                 chain_initializer='orthogonal',
                 bias_initializer='zeros',
                 boundary_initializer='zeros',
                 kernel_regularizer=None,
                 chain_regularizer=None,
                 boundary_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 chain_constraint=None,
                 boundary_constraint=None,
                 bias_constraint=None,
                 input_dim=None,
                 unroll=False,
                 **kwargs):
        super(CRF, self).__init__(**kwargs)

        # setup mask supporting flag, used by base class (the Layer)
        self.supports_masking = True

        self.units = units  # numbers of tags

        self.learn_mode = learn_mode
        assert self.learn_mode in ['join', 'marginal']

        self.test_mode = test_mode
        if self.test_mode is None:
            self.test_mode = 'viterbi' if self.learn_mode == 'join' else 'marginal'
        else:
            assert self.test_mode in ['viterbi', 'marginal']
        self.sparse_target = sparse_target
        self.use_boundary = use_boundary
        self.use_bias = use_bias

        self.activation = activations.get(activation)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.chain_initializer = initializers.get(chain_initializer)
        self.boundary_initializer = initializers.get(boundary_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.chain_regularizer = regularizers.get(chain_regularizer)
        self.boundary_regularizer = regularizers.get(boundary_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.chain_constraint = constraints.get(chain_constraint)
        self.boundary_constraint = constraints.get(boundary_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.unroll = unroll

        # value remembered for loss/metrics function
        self.logits = None
        self.nwords = None
        self.mask = None

    def build(self, input_shape):
        input_shape = tuple(tf.TensorShape(input_shape).as_list())
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[-1]

        # weights that mapping arbitrary tensor to tensor who have correct shape
        self.kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        # weights that work as transfer probility of each tags
        self.chain_kernel = self.add_weight(shape=(self.units, self.units),
                                            name='chain_kernel',
                                            initializer=self.chain_initializer,
                                            regularizer=self.chain_regularizer,
                                            constraint=self.chain_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = 0

        # if self.use_boundary:
        #     self.left_boundary = self.add_weight(shape=(self.units,),
        #                                          name='left_boundary',
        #                                          initializer=self.boundary_initializer,
        #                                          regularizer=self.boundary_regularizer,
        #                                          constraint=self.boundary_constraint)
        #     self.right_boundary = self.add_weight(shape=(self.units,),
        #                                           name='right_boundary',
        #                                           initializer=self.boundary_initializer,
        #                                           regularizer=self.boundary_regularizer,
        #                                           constraint=self.boundary_constraint)

    def call(self, input, mask=None, **kwargs):
        """

        :param input:
        :param mask: Tensor("embedding/NotEqual:0", shape=(?, ?), dtype=bool) or None
        :param kwargs:
        :return:
        """
        print("mask: {}".format(mask))

        # remember this value for later use
        self.mask = mask

        if mask is not None:
            assert tf.keras.backend.ndim(
                mask) == 2, 'Input mask to CRF must have dim 2 if not None'

        logits = self._dense_layer(input)

        # remember this value for later use
        self.logits = logits

        nwords = self._get_nwords(input, mask)
        print("nwords: {}".format(nwords))

        # remember this value for later use
        self.nwords = nwords

        # if self.test_mode == 'viterbi':
        test_output = self.get_viterbi_decoding(logits, nwords)
        # else:
        #     # TODO: not finished yet
        #     test_output = self.get_marginal_prob(input, mask)

        # self.uses_learning_phase = True
        # if self.learn_mode == 'join':
        # train_output = tf.keras.backend.zeros_like(tf.keras.backend.dot(input, self.kernel))
        # out = tf.keras.backend.in_train_phase(train_output, test_output)
        test_output = tf.cast(test_output, tf.float32)
        # out = tf.keras.backend.in_train_phase(logits, test_output)
        out = test_output
        # else:
        #     # TODO: not finished yet
        #     if self.test_mode == 'viterbi':
        #         train_output = self.get_marginal_prob(input, mask)
        #         out = tf.keras.backend.in_train_phase(train_output,
        #                                               test_output)
        #     else:
        #         out = test_output
        return out

    def _get_nwords(self, input, mask):
        if mask is not None:
            int_mask = tf.cast(mask, tf.int8)
            nwords = self.mask_to_nwords(int_mask)
        else:
            # make a mask tensor from input, then used to generate nwords
            input_energy_shape = tf.shape(input)
            raw_input_shape = tf.slice(input_energy_shape, [0], [2])
            alt_mask = tf.ones(raw_input_shape)

            nwords = self.mask_to_nwords(alt_mask)

        print("nwords: {}".format(nwords))
        return nwords

    def mask_to_nwords(self, mask):
        nwords = tf.cast(tf.reduce_sum(mask, 1), tf.int64)
        return nwords

    def get_viterbi_decoding(self, input_energy, nwords):
        # if self.use_boundary:
        #     input_energy = self.add_boundary_energy(
        #         input_energy, mask, self.left_boundary, self.right_boundary)

        pred_ids, _ = tf.contrib.crf.crf_decode(input_energy,
                                                self.chain_kernel, nwords)

        return pred_ids

    def get_config(self):
        # will be used for loading model from disk,
        # see https://github.com/keras-team/keras/issues/4871#issuecomment-269714512

        config = {
            'units': self.units,
            'learn_mode': self.learn_mode,
            'test_mode': self.test_mode,
            'use_boundary': self.use_boundary,
            'use_bias': self.use_bias,
            'sparse_target': self.sparse_target,
            'kernel_initializer': initializers.serialize(
                self.kernel_initializer),
            'chain_initializer': initializers.serialize(
                self.chain_initializer),
            'boundary_initializer': initializers.serialize(
                self.boundary_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'activation': activations.serialize(self.activation),
            'kernel_regularizer': regularizers.serialize(
                self.kernel_regularizer),
            'chain_regularizer': regularizers.serialize(
                self.chain_regularizer),
            'boundary_regularizer': regularizers.serialize(
                self.boundary_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'chain_constraint': constraints.serialize(self.chain_constraint),
            'boundary_constraint': constraints.serialize(
                self.boundary_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'input_dim': self.input_dim,
            'unroll': self.unroll}
        base_config = super(CRF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        # output_shape = input_shape[:2] + (self.units,)
        output_shape = input_shape[:2]
        return output_shape

    def compute_mask(self, input, mask=None):
        # if mask is not None and self.learn_mode == 'join':
        #     # new_mask = tf.keras.backend.any(mask, axis=1)
        #     return new_mask

        # must be None if this is the last layer
        new_mask = None
        return new_mask

    def get_decode_result(self, logits, mask):
        nwords = tf.cast(tf.reduce_sum(mask, 1), tf.int64)

        pred_ids, _ = tf.contrib.crf.crf_decode(logits, self.chain_kernel,
                                                nwords)

        return pred_ids

    # def get_negative_log_likelihood(self, y_preds, y_true, mask):
    def get_negative_log_likelihood(self, y_true):
        y_preds = self.logits

        # nwords = tf.cast(tf.reduce_sum(mask, 1), tf.int64)
        nwords = self.nwords

        y_preds = tf.cast(y_preds, tf.float32)
        y_true = tf.cast(y_true, tf.int64)
        nwords = tf.cast(nwords, tf.int32)
        self.chain_kernel = tf.cast(self.chain_kernel, tf.float32)

        # y_true = tf.squeeze(y_true, [-1])

        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(y_preds, y_true,
                                                              nwords,
                                                              self.chain_kernel)

        loss = tf.reduce_mean(-log_likelihood)

        return loss

    def get_accuracy(self, y_true, y_pred):
        judge = tf.keras.backend.cast(tf.keras.backend.equal(y_pred, y_true),
                                      tf.keras.backend.floatx())
        if self.mask is None:
            return tf.keras.backend.mean(judge)
        else:
            mask = tf.keras.backend.cast(self.mask, tf.keras.backend.floatx())
            return tf.keras.backend.sum(judge * mask) / tf.keras.backend.sum(
                mask)

    def _dense_layer(self, input_):
        # TODO: can simply just use tf.keras.layers.dense ?
        return self.activation(
            tf.keras.backend.dot(input_, self.kernel) + self.bias)
