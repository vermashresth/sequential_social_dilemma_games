import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune.registry import register_env
import tensorflow as tf

from social_dilemmas.envs.watershedComm import  WatershedSeqCommEnv

# from models.conv_to_fc_net import ConvToFCNet
# from models.conv_to_fcnet_v2 import ConvToFCNetv2
from models.fc_net import FCNet
from models.lstm_fc_net import LSTMFCNet
from models.watershed_nets import LSTMFCNet, FCNet

N_AGENTS = 4
FLAGS = tf.compat.v1.flags.FLAGS

tf.compat.v1.flags.DEFINE_string(
    'exp_name', None,
    'Name of the ray_results experiment directory where results are stored.')
tf.compat.v1.flags.DEFINE_string(
    'env', 'cleanup',
    'Name of the environment to rollout. Can be cleanup or harvest.')
tf.compat.v1.flags.DEFINE_string(
    'algorithm', 'A3C',
    'Name of the rllib algorithm to use.')
tf.compat.v1.flags.DEFINE_integer(
    'num_agents', 4,
    'Number of agent policies')
tf.compat.v1.flags.DEFINE_integer(
    'train_batch_size', 30000,
    'Size of the total dataset over which one epoch is computed.')
tf.compat.v1.flags.DEFINE_integer(
    'checkpoint_frequency', 20,
    'Number of steps before a checkpoint is saved.')
tf.compat.v1.flags.DEFINE_integer(
    'training_iterations', 10000,
    'Total number of steps to train for')
tf.compat.v1.flags.DEFINE_integer(
    'num_cpus', 2,
    'Number of available CPUs')
tf.compat.v1.flags.DEFINE_integer(
    'num_gpus', 1,
    'Number of available GPUs')
tf.compat.v1.flags.DEFINE_boolean(
    'use_gpus_for_workers', False,
    'Set to true to run workers on GPUs rather than CPUs')
tf.compat.v1.flags.DEFINE_boolean(
    'share_comm_layer', False,
    'Set to true to have shared layers for communication')
tf.compat.v1.flags.DEFINE_boolean(
    'use_gpu_for_driver', False,
    'Set to true to run driver on GPU rather than CPU.')
tf.compat.v1.flags.DEFINE_float(
    'num_workers_per_device', 1,
    'Number of workers to place on a single device (CPU or GPU)')
tf.compat.v1.flags.DEFINE_boolean(
    'return_agent_actions', 0,
    'If true we return the previous actions of all the agents')

harvest_default_params = {
    'lr_init': 0.00136,
    'lr_final': 0.000028,
    'entropy_coeff': .000687}

cleanup_default_params = {
    'lr_init': 0.00126,
    'lr_final': 0.000012,
    'entropy_coeff': .00176}

watershed_default_params = {
    'lr_init': 0.01,
    'lr_final': 0.0001,
    'entropy_coeff': 0.001
}

watershed_seq_comm_default_params = {
    'lr_init': 0.01,
    'lr_final': 0.0001,
    'entropy_coeff': 0.001
}


def setup(env, hparams, algorithm, train_batch_size, num_cpus, num_gpus,
          num_agents, use_gpus_for_workers=False, use_gpu_for_driver=False,
          num_workers_per_device=1, return_agent_actions=False, share_comm_layer=True):


    if env == 'watershed_seq_comm':
        def env_creator(_):
            return WatershedSeqCommEnv()
        single_env = WatershedSeqCommEnv()
    else:
        print("No other env is supported")

    env_name = env + "_env"
    register_env(env_name, env_creator)

    obs_space = single_env.observation_space
    act_space = single_env.action_space

    obs_comm_space = single_env.get_observation_space(2)
    act_comm_space = single_env.get_action_space(2)

    # Each policy can have a different configuration (including custom model)
    def gen_policy(i):
        if i<N_AGENTS:
            config = {
            "model": {"custom_model": "comm_fc_net", "use_lstm": False,
                    "custom_options": {"id": i%N_AGENTS, "share_comm_layer": share_comm_layer, "return_agent_actions": return_agent_actions, "cell_size": 128},
                      }}
            return (None, obs_comm_space, act_comm_space, config)
        else:
            config = {
            "model": {"custom_model": "lstm_fc_net", "use_lstm": False,
                    "custom_options": {"id": i%N_AGENTS, "share_comm_layer": share_comm_layer, "return_agent_actions": return_agent_actions, "cell_size": 128},
                      }}
            return (None, obs_space, act_space, config)

    # Setup PPO with an ensemble of `num_policies` different policy graphs
    policy_graphs = {}
    for i in range(2*num_agents):
        policy_graphs['agent-' + str(i)] = gen_policy(i)

    def policy_mapping_fn(agent_id):
        return agent_id

    # register the custom model
    # model_name = "conv_to_fc_net"
    # ModelCatalog.register_custom_model(model_name, ConvToFCNet)

    # model_name = "fc_net"
    # ModelCatalog.register_custom_model(model_name, FCNet)

    model_name = "lstm_fc_net"
    ModelCatalog.register_custom_model(model_name, LSTMFCNet)

    model_name = "comm_fc_net"
    ModelCatalog.register_custom_model(model_name, FCNet)

    agent_cls = get_agent_class(algorithm)
    config = agent_cls._default_config.copy()

    # information for replay
    config['env_config']['func_create'] = tune.function(env_creator)
    config['env_config']['env_name'] = env_name
    config['env_config']['run'] = algorithm

    # Calculate device configurations
    gpus_for_driver = int(use_gpu_for_driver)
    cpus_for_driver = 1 - gpus_for_driver
    if use_gpus_for_workers:
        spare_gpus = (num_gpus - gpus_for_driver)
        num_workers = int(spare_gpus * num_workers_per_device)
        num_gpus_per_worker = spare_gpus / num_workers
        num_cpus_per_worker = 0
    else:
        spare_cpus = (num_cpus - cpus_for_driver)
        num_workers = int(spare_cpus * num_workers_per_device)
        num_gpus_per_worker = 0
        num_cpus_per_worker = spare_cpus / num_workers

    # hyperparams
    config.update({
        # TODO(@evinitsky) why is this 3000
                "train_batch_size": train_batch_size,
                "horizon": 1000,
                "gamma": 0.99,
                "lr_schedule":
                [[0, hparams['lr_init']],
                    [20000000, hparams['lr_final']]],
                "num_workers": num_workers,
                "num_gpus": gpus_for_driver,  # The number of GPUs for the driver
                "num_cpus_for_driver": cpus_for_driver,
                "num_gpus_per_worker": num_gpus_per_worker,   # Can be a fraction
                "num_cpus_per_worker": num_cpus_per_worker,   # Can be a fraction
                "entropy_coeff": hparams['entropy_coeff'],
                "multiagent": {
                    "policies": policy_graphs,
                    "policy_mapping_fn": tune.function(policy_mapping_fn),
                },

    })
    return algorithm, env_name, config


def main(unused_argv):
    ray.init()
    if FLAGS.env == 'watershed_seq_comm':
        hparams = watershed_seq_comm_default_params
    else:
        print("Only watershed seq comm supported")
    alg_run, env_name, config = setup(FLAGS.env, hparams, FLAGS.algorithm,
                                      FLAGS.train_batch_size,
                                      FLAGS.num_cpus,
                                      FLAGS.num_gpus, FLAGS.num_agents,
                                      FLAGS.use_gpus_for_workers,
                                      FLAGS.use_gpu_for_driver,
                                      FLAGS.num_workers_per_device,
                                      FLAGS.return_agent_actions,
                                      FLAGS.share_comm_layer)

    if FLAGS.exp_name is None:
        exp_name = FLAGS.env + '_' + FLAGS.algorithm
    else:
        exp_name = FLAGS.exp_name
    print('Commencing experiment', exp_name)

    run_experiments({
        exp_name: {
            "run": alg_run,
            "env": env_name,
            "stop": {
                "training_iteration": FLAGS.training_iterations
            },
            'checkpoint_freq': FLAGS.checkpoint_frequency,
            "config": config,
        }
    })


if __name__ == '__main__':
    tf.compat.v1.app.run(main)
