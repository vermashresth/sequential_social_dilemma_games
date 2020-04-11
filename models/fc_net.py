# Model taken from https://arxiv.org/pdf/1810.08647.pdf,
# INTRINSIC SOCIAL MOTIVATION VIA CAUSAL
# INFLUENCE IN MULTI-AGENT RL


# model is a single convolutional layer with a kernel of size 3, stride of size 1, and 6 output
# channels. This is connected to two fully connected layers of size 32 each

# import tensorflow as tf

from ray.rllib.models.tf.misc import normc_initializer, flatten
from ray.rllib.models.model import Model
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

from ray.rllib.utils import try_import_tf
tf = try_import_tf()

class FCNet(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(FCNet, self).__init__(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations")
        layer_1 = tf.keras.layers.Dense(
            4,
            name="my_layer1",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(self.inputs)
        layer_2 = tf.keras.layers.Dense(
            4,
            name="my_layer2",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(layer_1)
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(layer_2)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(layer_1)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

class FCNetOld(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):

        # The last row is other agent actions
        if options["custom_options"]["return_agent_actions"]:
            inputs = input_dict["obs"][:-1, :]
        else:
            inputs = input_dict["obs"]

        hiddens = [32, 32]
        with tf.name_scope("custom_net"):
            # inputs = slim.conv2d(
            #     inputs,
            #     6,
            #     [3, 3],
            #     1,
            #     activation_fn=tf.nn.relu,
            #     scope="conv")
            last_layer = flatten(inputs)
            i = 1
            for size in hiddens:
                label = "fc{}".format(i)
                last_layer = tf.keras.layers.Dense(
                    size,
                    kernel_initializer=normc_initializer(1.0),
                    activation=tf.nn.relu,
                    name=label)(last_layer)
                i += 1
            output = tf.keras.layers.Dense(
                size,
                kernel_initializer=normc_initializer(0.01),
                activation=tf.nn.relu,
                name=label+'out')(last_layer)
            return output, last_layer
