from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_tf
tf = try_import_tf()


class LSTMFCNet(RecurrentTFModelV2):
    """Example of using the Keras functional API to define a RNN model."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 hiddens_size=256,
                 cell_size=64):
        super(MyKerasRNN, self).__init__(obs_space, action_space, num_outputs,
                                         model_config, name)
        self.cell_size = cell_size

        # Define input layers
        input_layer = tf.keras.layers.Input(
            shape=(None, obs_space.shape[0]), name="inputs")
        state_in_h = tf.keras.layers.Input(shape=(cell_size, ), name="h")
        state_in_c = tf.keras.layers.Input(shape=(cell_size, ), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        # Preprocess observation with a hidden layer and send to LSTM cell
        dense1 = tf.keras.layers.Dense(
            hiddens_size, activation=tf.nn.relu, name="dense1")(input_layer)
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            cell_size, return_sequences=True, return_state=True, name="lstm")(
                inputs=dense1,
                mask=tf.sequence_mask(seq_in),
                initial_state=[state_in_h, state_in_c])

        # Postprocess LSTM output with another hidden layer and compute values
        logits = tf.keras.layers.Dense(
            self.num_outputs,
            activation=tf.keras.activations.linear,
            name="logits")(lstm_out)
        values = tf.keras.layers.Dense(
            1, activation=None, name="values")(lstm_out)

        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[input_layer, seq_in, state_in_h, state_in_c],
            outputs=[logits, values, state_h, state_c])
        self.register_variables(self.rnn_model.variables)
        self.rnn_model.summary()

    @override(RecurrentTFModelV2)
    def forward_rnn(self, inputs, state, seq_lens):
        model_out, self._value_out, h, c = self.rnn_model([inputs, seq_lens] +
                                                          state)
        return model_out, [h, c]

    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])
