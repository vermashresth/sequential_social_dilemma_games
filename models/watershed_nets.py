import numpy as np

from ray.rllib.models.tf.misc import normc_initializer, flatten
from ray.rllib.models.model import Model
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_tf
tf = try_import_tf()

NUM_AGENTS = 4
shared_layers = []
##FC
# Input -16 -16 out
# Value Input 16-16-out

# LSTMFC
#Input -16 -16 LSTM(64) - out
#Value Input -16 -16 LSTM(64) - out

for i in range(NUM_AGENTS):
    shared_layers.append(tf.keras.layers.Dense(
        16,
        name="my_layer1"+str(i),
        activation=tf.nn.relu,
        kernel_initializer=normc_initializer(1.0)))

class FCNet(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(FCNet, self).__init__(obs_space, action_space,
                                           num_outputs, model_config, name)
        try:
            share_comm_layer = model_config["custom_options"]["share_comm_layer"]
            id = model_config["custom_options"]["id"]

            self.id = model_config["custom_options"]["id"]
            if self.id>=NUM_AGENTS:
                self.causal=True
            else:
                self.causal = False
        except:
            share_comm_layer = False


        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations")
        layer_0 = tf.keras.layers.Dense(
            16,
            name="my_layer0",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(self.inputs)

        if share_comm_layer:
            new_layer = shared_layers[id]
        else:
            new_layer = tf.keras.layers.Dense(
                16,
                name="my_layer1",
                activation=tf.nn.relu,
                kernel_initializer=normc_initializer(1.0))
        layer_1 = new_layer(layer_0)

        # layer_2 = tf.keras.layers.Dense(
        #     4,
        #     name="my_layer2",
        #     activation=tf.nn.relu,
        #     kernel_initializer=normc_initializer(1.0))(layer_1)
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(layer_1)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(layer_1)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.register_variables(self.base_model.variables)
        print("FCNET", self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

class LSTMFCNet(RecurrentTFModelV2):
    """Example of using the Keras functional API to define a RNN model."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 hiddens_size=16,
                 cell_size=128):
        super(LSTMFCNet, self).__init__(obs_space, action_space, num_outputs,
                                         model_config, name)
        self.cell_size = cell_size
        try:
            share_comm_layer = model_config["custom_options"]["share_comm_layer"]
            id = model_config["custom_options"]["id"]
            self.id = model_config["custom_options"]["id"]
            if self.id>=NUM_AGENTS:
                self.causal=True
            else:
                self.causal = False
        except Exception as e:
            print(e)
            share_comm_layer = False
            self.causal = False
        # Define input layers
        input_layer = tf.keras.layers.Input(
            shape=(None, obs_space.shape[0]), name="inputs")
        state_in_h = tf.keras.layers.Input(shape=(cell_size, ), name="h")
        state_in_c = tf.keras.layers.Input(shape=(cell_size, ), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        # Preprocess observation with a hidden layer and send to LSTM cell
        dense0 = tf.keras.layers.Dense(
            16, activation=tf.nn.relu, name="dense0")(input_layer)

        if share_comm_layer:
            new_layer = shared_layers[id]
        else:
            new_layer = tf.keras.layers.Dense(
                16,
                name="my_layer1",
                activation=tf.nn.relu,
                kernel_initializer=normc_initializer(1.0))
        dense1 = new_layer(dense0)

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
        print("LSTMFC", self.rnn_model.variables)

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
