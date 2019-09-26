import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import initializers, regularizers, constraints, activations
from tensorflow.python.keras.layers import InputSpec, Layer
from tensorflow.contrib.crf import crf_log_likelihood
from tf_crf_layer.crf import crf_decode

from tf_crf_layer import keras_utils

"""
TODO

* learn_mode is not supported
* test_mode is not supported
* sparse_target is not supported
* use_boundary need test
* input_dim is not know how to use
* unroll is not supported

* left padding of mask is not supported

* not test yet if CRF is the first layer
"""


# for future reference:
# B: batch size
# M: intent number
# n: where n is the tag set number
# n+2: n plus start and end tag
# N: n * n
# T: sequence length
# F: feature number


def add_pooling_strategies(x):
    # x: shape (B, M, n+2, n+2)
    # return (B, n+2, n+2)
    return K.sum(x, axis=1, keepdims=False)


pooling_strategies = {"add": add_pooling_strategies}


@keras_utils.register_keras_custom_object
class CRF(Layer):
    def __init__(
        self,
        units,
        learn_mode="join",
        test_mode=None,
        sparse_target=False,
        use_boundary=False,
        use_bias=True,
        use_kernel=True,
        activation="linear",
        kernel_initializer="glorot_uniform",
        chain_initializer="orthogonal",
        bias_initializer="zeros",
        left_boundary_initializer="zeros",
        right_boundary_initializer="zeros",
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
        transition_constraint=None,
        transition_constraint_matrix=None,  # shape: (M, n+2, n+2)
        pooling_strategy=add_pooling_strategies,
        **kwargs
    ):
        super(CRF, self).__init__(**kwargs)

        # setup mask supporting flag, used by base class (the Layer)
        self.supports_masking = True

        self.units = units  # numbers of tags

        self.learn_mode = learn_mode
        assert self.learn_mode in ["join", "marginal"]

        self.test_mode = test_mode
        if self.test_mode is None:
            self.test_mode = "viterbi" if self.learn_mode == "join" else "marginal"
        else:
            assert self.test_mode in ["viterbi", "marginal"]
        self.sparse_target = sparse_target
        self.use_boundary = use_boundary
        self.use_bias = use_bias
        self.use_kernel = use_kernel

        self.activation = activations.get(activation)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.chain_initializer = initializers.get(chain_initializer)
        self.left_boundary_initializer = initializers.get(left_boundary_initializer)
        self.right_boundary_initializer = initializers.get(right_boundary_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.chain_regularizer = regularizers.get(chain_regularizer)
        self.boundary_regularizer = regularizers.get(boundary_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.chain_constraint = constraints.get(chain_constraint)
        self.boundary_constraint = constraints.get(boundary_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.input_dim = input_dim
        self.unroll = unroll
        self.transition_constraint = transition_constraint
        self.transition_constraint_matrix = transition_constraint_matrix
        self.pooling_strategy = pooling_strategy

        # value remembered for loss/metrics function
        self.logits = None
        self.nwords = None
        self.mask = None

        # global variable
        self.kernel = None
        self.chain_kernel = None  # shape: (self.units, self.units)
        self.bias = None
        self.left_boundary = None
        self.right_boundary = None
        self.transition_constraint_mask = None
        self.transition_constraint_matrix_mask = None
        self.dynamic_transition_constraint = None
        self.is_training = False

        # for debug
        self.dynamic_transition_constraint_input = None
        self.inputs = None
        self.actual_left_boundary = None
        self.actual_right_boundary = None

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        input_shape = tuple(tf.TensorShape(input_shape).as_list())
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[-1]

        self.transition_constraint_mask = self.add_weight(
            shape=(self.units + 2, self.units + 2),
            name="transition_constraint_mask",
            initializer=initializers.Constant(self.get_transition_constraint_mask()),
            trainable=False,
        )

        if self.transition_constraint_matrix is not None:
            self.transition_constraint_matrix_mask = self.add_weight(
                shape=self.transition_constraint_matrix.shape,
                name="transition_constraint_mask",
                initializer=initializers.Constant(self.transition_constraint_matrix),
                trainable=False,
            )

        if self.use_kernel:
            # weights that mapping arbitrary tensor to correct shape
            self.kernel = self.add_weight(
                shape=(self.input_dim, self.units),
                name="kernel",
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
            )

        # weights that work as transfer probability of each tags
        self.chain_kernel = self.add_weight(
            shape=(self.units, self.units),
            name="chain_kernel",
            initializer=self.chain_initializer,
            regularizer=self.chain_regularizer,
            constraint=self.chain_constraint,
        )

        # bias that works with self.kernel
        if self.use_kernel and self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                name="bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = 0

        # weight of <START> to tag probability and tag to <END> probability
        if self.use_boundary:
            self.left_boundary = self.add_weight(
                shape=(self.units,),
                name="left_boundary",
                initializer=self.left_boundary_initializer,
                regularizer=self.boundary_regularizer,
                constraint=self.boundary_constraint,
            )
            self.right_boundary = self.add_weight(
                shape=(self.units,),
                name="right_boundary",
                initializer=self.right_boundary_initializer,
                regularizer=self.boundary_regularizer,
                constraint=self.boundary_constraint,
            )

        # or directly call self.built = True
        super(CRF, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        # inputs: Tensor(shape(B, T, F)) or (Tensor(B, T, F), Tensor(B, M))
        # mask: Tensor(shape=(B, T), dtype=bool) or None

        training = kwargs.get("training", None)
        if training is None:
            training = K.learning_phase()

        # session = K.get_session()
        #
        # print("training_flag:", session.run(training))

        # # DEBUG: output training value
        # training = tf.Print(training, [training])

        self.is_training = training

        dynamic_transition_constraint_indicator = None

        if isinstance(inputs, list):
            assert len(inputs) == 2, "Input must have two input tensors"

            dynamic_transition_constraint_indicator = inputs[1]
            self.dynamic_transition_constraint_input = (
                dynamic_transition_constraint_indicator
            )

            inputs = inputs[0]
            self.inputs = inputs

        if isinstance(mask, list):
            print(mask)
            mask = mask[0]

        if mask is not None:
            assert K.ndim(mask) == 2, "Input mask to CRF must have dim 2 if not None"

        # remember this value for later use
        self.mask = mask

        if dynamic_transition_constraint_indicator is not None:
            # reshape from (B, M) to (B, M, 1, 1)
            constraint_indicator = K.expand_dims(
                K.expand_dims(dynamic_transition_constraint_indicator)
            )

            # shape: (B, M, n+2, n+2)
            raw_constrain_pool = (
                self.transition_constraint_matrix * constraint_indicator
            )

            # shape: (B, n+2, n+2)
            unrolled_dynamic_transition_constraint = self.pooling_strategy(
                raw_constrain_pool
            )

            # shape: (B, n+2, n+2), +2 for start and end tag
            self.dynamic_transition_constraint = unrolled_dynamic_transition_constraint

        logits = self._dense_layer(inputs)
        print("logits:", logits)

        # appending boundary probability info
        if self.use_boundary:
            logits = self.add_boundary_energy(logits, mask)

        print("logits:", logits)

        # remember this value for later use
        self.logits = logits

        nwords = self._get_nwords(inputs, mask)
        print("nwords: {}".format(nwords))

        # remember this value for later use
        self.nwords = nwords

        if self.test_mode == "viterbi":
            test_output = self.get_viterbi_decoding(logits, nwords)
        else:
            # TODO: not supported yet
            pass
            # test_output = self.get_marginal_prob(input, mask)

        if self.learn_mode == "join":
            # WHY: don't remove this line, useless but remote it will cause bug
            test_output = tf.cast(test_output, tf.float32)
            out = test_output
        else:
            # TODO: not supported yet
            pass
            # if self.test_mode == 'viterbi':
            #     train_output = self.get_marginal_prob(input, mask)
            #     out = K.in_train_phase(train_output,
            #                                           test_output)
            # else:
            #     out = test_output

        return out

    def _get_nwords(self, input, mask):
        if mask is not None:
            int_mask = K.cast(mask, tf.int8)
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
        nwords = K.cast(K.sum(mask, 1), tf.int64)
        return nwords

    @staticmethod
    def shift_left(x, offset=1):
        assert offset > 0
        return K.concatenate([x[:, offset:], K.zeros_like(x[:, :offset])], axis=1)

    @staticmethod
    def shift_right(x, offset=1):
        assert offset > 0
        return K.concatenate([K.zeros_like(x[:, :offset]), x[:, :-offset]], axis=1)

    def compute_energy(self, start, end, energy, mask):
        self.actual_left_boundary = start
        self.actual_right_boundary = end

        if mask is None:
            energy = K.concatenate([energy[:, :1, :] + start, energy[:, 1:, :]], axis=1)

            energy = K.concatenate([energy[:, :-1, :], energy[:, -1:, :] + end], axis=1)
        else:
            mask = K.expand_dims(K.cast(mask, K.floatx()), axis=-1)
            start_mask = K.cast(K.greater(mask, self.shift_right(mask)), K.floatx())

            # original code:
            # end_mask = K.cast(K.greater(self.shift_left(mask), mask), K.floatx())
            # TODO: is this a bug? should be K.greater(mask, self.shift_left(mask)) ?
            # patch applied
            end_mask = K.cast(K.greater(mask, self.shift_left(mask)), K.floatx())
            energy = energy + start_mask * start
            energy = energy + end_mask * end

        return energy

    def compute_boundary_with_constraint(self):
        if self.transition_constraint:
            start_tag = self.units
            end_tag = self.units + 1

            left_boundary = self.left_boundary * self.transition_constraint_mask[
                start_tag, : self.units
            ] + -10000.0 * (
                1 - self.transition_constraint_mask[start_tag, : self.units]
            )
            right_boundary = self.right_boundary * self.transition_constraint_mask[
                : self.units, end_tag
            ] + -10000.0 * (1 - self.transition_constraint_mask[: self.units, end_tag])

            return left_boundary, right_boundary

        if self.transition_constraint_matrix is not None:
            start_tag = self.units
            end_tag = self.units + 1

            # shape: (B, 1, n)
            # left_dynamic_constraint = self.dynamic_transition_constraint[:, start_tag, :self.units]  # shape: (B, n)
            left_dynamic_constraint = tf.slice(
                self.dynamic_transition_constraint,
                [0, start_tag, 0],
                [-1, 1, self.units],
            )  # shape: (B, 1, n)

            # shape: (B, 1, n)
            left_boundary = self.left_boundary * left_dynamic_constraint + -10000.0 * (
                1 - left_dynamic_constraint
            )

            # shape: (B, 1, n)
            # right_dynamic_constraint = self.dynamic_transition_constraint[:, :self.units, end_tag]  # shape: (B, n)
            raw_right_dynamic_constraint = tf.slice(
                self.dynamic_transition_constraint, [0, 0, end_tag], [-1, self.units, 1]
            )  # shape: (B, n, 1)
            right_dynamic_constraint = tf.transpose(
                raw_right_dynamic_constraint, perm=[0, 2, 1]
            )  # shape: (B, 1, n)

            # shape: (B, 1, n)
            right_boundary = (
                self.right_boundary * right_dynamic_constraint
                + -10000.0 * (1 - right_dynamic_constraint)
            )

            return left_boundary, right_boundary

        # no constraint at all, return ordinary energy
        return self.compute_boundary_ordinary()

    def compute_boundary_without_constraint(self):
        return self.compute_boundary_ordinary()

    def compute_boundary_ordinary(self):
        # no any constraint at all

        def expend_scalar_to_3d(x):
            # expend tensor from shape (x, ) to (1, 1, x)
            return K.expand_dims(K.expand_dims(x, 0), 0)

        # shape: (1, 1, n)
        left_boundary = expend_scalar_to_3d(self.left_boundary)

        # shape: (1, 1, n)
        right_boundary = expend_scalar_to_3d(self.right_boundary)

        return left_boundary, right_boundary

    def add_boundary_energy(self, energy, mask):
        """

        :param energy: Tensor(shape=(B, T, F))
        :param mask: Tensor(shape=(B, T), dtype=bool) or None
        :return:
        """

        def compute_constrained_energy():
            left_boundary, right_boundary = self.compute_boundary_with_constraint()

            return self.compute_energy(left_boundary, right_boundary, energy, mask)

        def compute_unconstrained_energy():
            left_boundary, right_boundary = self.compute_boundary_without_constraint()

            return self.compute_energy(left_boundary, right_boundary, energy, mask)

        computed_energy = tf_utils.smart_cond(
            self.is_training, compute_unconstrained_energy, compute_constrained_energy
        )

        return computed_energy

    def compute_static_constrained_chain_kernel(self):
        if self.transition_constraint:
            chain_kernel_mask = self.transition_constraint_mask[
                : self.units, : self.units
            ]
            constrained_transitions = (
                self.chain_kernel * chain_kernel_mask
                + -10000.0 * (1 - chain_kernel_mask)
            )

            return constrained_transitions

    def compute_dynamic_constrained_chain_kernel(self):
        if self.transition_constraint_matrix is not None:
            dynamic_chain_kernel_mask = self.dynamic_transition_constraint[
                :, : self.units, : self.units
            ]
            dynamic_constrained_transitions = (
                self.chain_kernel * dynamic_chain_kernel_mask
                + -10000.0 * (1 - dynamic_chain_kernel_mask)
            )

            return dynamic_constrained_transitions

    def get_viterbi_decoding(self, input_energy, nwords):
        def compute_constrained_chain_kernel():
            dynamic_constrained_transitions = None

            if self.transition_constraint:
                dynamic_constrained_transitions = (
                    self.compute_static_constrained_chain_kernel()
                )

            if self.transition_constraint_matrix is not None:
                dynamic_constrained_transitions = (
                    self.compute_dynamic_constrained_chain_kernel()
                )

            if dynamic_constrained_transitions is None:
                # in case there are no constraint at all
                dynamic_constrained_transitions = self.chain_kernel

            return crf_decode(input_energy, dynamic_constrained_transitions, nwords)

        def compute_unconstrained_chain_kernel():
            return crf_decode(input_energy, self.chain_kernel, nwords)

        pred_ids, _ = tf_utils.smart_cond(
            self.is_training,
            compute_unconstrained_chain_kernel,
            compute_constrained_chain_kernel,
        )

        return pred_ids

    def get_config(self):
        # will be used for loading model from disk,
        # see https://github.com/keras-team/keras/issues/4871#issuecomment-269714512

        config = {
            "units": self.units,
            "learn_mode": self.learn_mode,
            "test_mode": self.test_mode,
            "use_boundary": self.use_boundary,
            "use_bias": self.use_bias,
            "use_kernel": self.use_kernel,
            "sparse_target": self.sparse_target,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "chain_initializer": initializers.serialize(self.chain_initializer),
            "left_boundary_initializer": initializers.serialize(
                self.left_boundary_initializer
            ),
            "right_boundary_initializer": initializers.serialize(
                self.right_boundary_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "activation": activations.serialize(self.activation),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "chain_regularizer": regularizers.serialize(self.chain_regularizer),
            "boundary_regularizer": regularizers.serialize(self.boundary_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "chain_constraint": constraints.serialize(self.chain_constraint),
            "boundary_constraint": constraints.serialize(self.boundary_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "input_dim": self.input_dim,
            "unroll": self.unroll,
            "transition_constraint": self.transition_constraint,
            "transition_constraint_matrix": self.transition_constraint_matrix,
            "pooling_strategy": self.pooling_strategy,
        }
        base_config = super(CRF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[:2]
        return output_shape

    # def compute_mask(self, input, mask=None):
    #     if isinstance(mask, list):
    #         mask = mask[0]
    #
    #     if mask is not None and self.learn_mode == "join":
    #         # transform mask from shape (?, ?) to (?, )
    #         new_mask = K.any(mask, axis=1)
    #         return new_mask
    #
    #     return mask

    # def get_decode_result(self, logits, mask):
    #     nwords = K.cast(K.sum(mask, 1), tf.int64)
    #
    #     pred_ids, _ = crf_decode(logits, self.chain_kernel,
    #                                             nwords)
    #
    #     return pred_ids

    def get_negative_log_likelihood(self, y_true):
        y_preds = self.logits

        nwords = self.nwords

        y_preds = K.cast(y_preds, tf.float32)
        y_true = K.cast(y_true, tf.int64)
        nwords = K.cast(nwords, tf.int32)
        self.chain_kernel = K.cast(self.chain_kernel, tf.float32)

        # print_op = tf.print([tf.shape(y_preds), tf.shape(y_true), tf.shape(nwords)])
        # with tf.control_dependencies([print_op]):
        #     y_preds = tf.identity(y_preds)

        # print(K.get_session().graph.get_name_scope())

        log_likelihood, _ = crf_log_likelihood(
            y_preds, y_true, nwords, self.chain_kernel
        )

        return -log_likelihood

    def get_accuracy(self, y_true, y_pred):
        judge = K.cast(K.equal(y_pred, y_true), K.floatx())
        if self.mask is None:
            result = K.mean(judge)
            return result
        else:
            mask = K.cast(self.mask, K.floatx())
            result = K.sum(judge * mask) / K.sum(mask)
            return result

    def _dense_layer(self, input_):
        """
        mapping any number of features to votes for each tags

        :param input_: Tensor(shape=(B, T, F))
        :return: Tensor(shape=(B, T, F))
        """
        # TODO: can simply just use tf.keras.layers.dense ?
        if self.use_kernel:
            return self.activation(K.dot(input_, self.kernel) + self.bias)

        return input_

    def get_transition_constraint_mask(self):
        if not self.transition_constraint:
            # All transitions are valid.
            constraint_mask = np.ones([self.units + 2, self.units + 2])
        else:
            constraint_mask = np.zeros([self.units + 2, self.units + 2])
            for i, j in self.transition_constraint:
                constraint_mask[i, j] = 1.0

        return constraint_mask

    def compute_effective_boundary(self, energy):
        """

        :return: Tensor(shape=(B, 1, n)) or Tensor(shape=(1, 1, n))
        """

        def compute_constrained_boundary():
            if self.transition_constraint:
                start_tag = self.units
                end_tag = self.units + 1

                left_boundary = self.left_boundary * self.transition_constraint_mask[
                    start_tag, : self.units
                ] + -10000.0 * (
                    1 - self.transition_constraint_mask[start_tag, : self.units]
                )
                right_boundary = self.right_boundary * self.transition_constraint_mask[
                    : self.units, end_tag
                ] + -10000.0 * (
                    1 - self.transition_constraint_mask[: self.units, end_tag]
                )

                return left_boundary, right_boundary

            if self.transition_constraint_matrix is not None:
                start_tag = self.units
                end_tag = self.units + 1

                # shape: (B, 1, n)
                # left_dynamic_constraint = self.dynamic_transition_constraint[:, start_tag, :self.units]  # shape: (B, n)
                left_dynamic_constraint = tf.slice(
                    self.dynamic_transition_constraint,
                    [0, start_tag, 0],
                    [-1, 1, self.units],
                )  # shape: (B, 1, n)

                # shape: (B, 1, n)
                left_boundary = (
                    self.left_boundary * left_dynamic_constraint
                    + -10000.0 * (1 - left_dynamic_constraint)
                )

                # shape: (B, 1, n)
                # right_dynamic_constraint = self.dynamic_transition_constraint[:, :self.units, end_tag]  # shape: (B, n)
                raw_right_dynamic_constraint = tf.slice(
                    self.dynamic_transition_constraint,
                    [0, 0, end_tag],
                    [-1, self.units, 1],
                )  # shape: (B, n, 1)
                right_dynamic_constraint = tf.transpose(
                    raw_right_dynamic_constraint, perm=[0, 2, 1]
                )  # shape: (B, 1, n)

                # shape: (B, 1, n)
                right_boundary = (
                    self.right_boundary * right_dynamic_constraint
                    + -10000.0 * (1 - right_dynamic_constraint)
                )

                return left_boundary, right_boundary

            return compute_unconstrained_boundary()

        def compute_unconstrained_boundary():
            # no any constraint at all

            def expend_scalar_to_3d(x):
                # expend tensor from shape (x, ) to (1, 1, x)
                return K.expand_dims(K.expand_dims(x, 0), 0)

            # shape: (1, 1, n)
            left_boundary = expend_scalar_to_3d(self.left_boundary)

            # shape: (1, 1, n)
            right_boundary = expend_scalar_to_3d(self.right_boundary)

            return left_boundary, right_boundary

        left_boundary, right_boundary = tf_utils.smart_cond(
            self.is_training,
            compute_unconstrained_boundary,
            compute_constrained_boundary,
        )

        return left_boundary, right_boundary
