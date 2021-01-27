import argparse
import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune.registry import register_env
import tensorflow as tf

from social_dilemmas.envs.watershedOrderedComm import  WatershedSeqCommEnv

# from models.conv_to_fc_net import ConvToFCNet
# from models.conv_to_fcnet_v2 import ConvToFCNetv2

# from models.fc_net import FCNet
# from models.lstm_fc_net import LSTMFCNet
from models.watershed_nets import LSTMFCNet, FCNet

from social_dilemmas.envs.watershedLogging import on_episode_start, on_episode_step_comm, on_episode_end

N_AGENTS = 4
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='baseline', help='Name experiment will be stored under')
parser.add_argument('--env', type=str, default='cleanup', help='Name of the environment to rollout. Can be ')
parser.add_argument('--algorithm', type=str, default='PPO', help='Name of the rllib algorithm to use.')
parser.add_argument('--num_agents', type=int, default=4, help='Number of agent policies')
parser.add_argument('--train_batch_size', type=int, default=2600,
                    help='Size of the total dataset over which one epoch is computed.')
parser.add_argument('--checkpoint_frequency', type=int, default=10,
                    help='Number of steps before a checkpoint is saved.')
parser.add_argument('--training_iterations', type=int, default=100, help='Total number of steps to train for')
parser.add_argument('--num_cpus', type=int, default=2, help='Number of available CPUs')
parser.add_argument('--num_gpus', type=int, default=0, help='Number of available GPUs')
parser.add_argument('--use_gpus_for_workers', action='store_true', default=False,
                    help='Set to true to run workers on GPUs rather than CPUs')
parser.add_argument('--use_gpu_for_driver', action='store_true', default=False,
                    help='Set to true to run driver on GPU rather than CPU.')
parser.add_argument('--num_workers_per_device', type=float, default=1,
                    help='Number of workers to place on a single device (CPU or GPU)')

parser.add_argument('--return_agent_actions', action='store_true', default=False,
                    help='If true we only use local observation')

parser.add_argument('--share_comm_layer', action='store_true', default=False,
                    help='If true we share layers')
parser.add_argument('--local_obs', action='store_true', default=False,
                    help='If true we only use local observation')
parser.add_argument('--local_rew', action='store_true', default=False,
                    help='If true we return indicidual rewards to agents')


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
    'lr_init': 0.0001,
    'lr_final': 0.00001,
    'entropy_coeff': 0.001
}


def setup(env, hparams, algorithm, train_batch_size, num_cpus, num_gpus,
          num_agents, use_gpus_for_workers=False, use_gpu_for_driver=False,
          num_workers_per_device=1, return_agent_actions=False, local_rew=False, local_obs=False, share_comm_layer=True):


    if env == 'watershed_seq_comm':
        def env_creator(_):
            return WatershedSeqCommEnv(return_agent_actions=return_agent_actions, local_rew=local_rew, local_obs=local_obs)
        single_env = WatershedSeqCommEnv(return_agent_actions=return_agent_actions, local_rew=local_rew, local_obs=local_obs)
    else:
        print("only watershed_seq_comm supported")

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
            "model": {"custom_model": "lstm_fc_net", "use_lstm": False,
                    "custom_options": {"id": i%N_AGENTS, "share_comm_layer": share_comm_layer, "return_agent_actions": return_agent_actions},
                      }}
            return (None, obs_comm_space, act_comm_space, config)
        else:
            config = {
            "model": {"custom_model": "lstm_fc_net", "use_lstm": False,
                    "custom_options": {"id": i%N_AGENTS, "share_comm_layer": share_comm_layer, "return_agent_actions": return_agent_actions},
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

    # model_name = "comm_fc_net"
    # ModelCatalog.register_custom_model(model_name, FCNet)

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
                "callbacks": {
                    "on_episode_start": on_episode_start,
                    "on_episode_step": on_episode_step_comm,
                    "on_episode_end": on_episode_end,
                },

    })

    if args.algorithm == "PPO":
        config.update({"num_sgd_iter": 10,
                       "train_batch_size": train_batch_size,
                       "sgd_minibatch_size": 128,
                       "vf_loss_coeff": 1e-4
                       })
    elif args.algorithm == "A3C":
        config.update({"sample_batch_size": 50,
                       "vf_loss_coeff": 0.1
                       })
    elif args.algorithm == "IMPALA":
        config.update({"train_batch_size": train_batch_size,
                       "sample_batch_size": 50,
                       "vf_loss_coeff": 0.1
                       })
    else:
        sys.exit("The only available algorithms are A3C and PPO")

    return algorithm, env_name, config


if __name__=='__main__':
    args = parser.parse_args()
    ray.init()
    if args.env == 'watershed_seq_comm':
        hparams = watershed_seq_comm_default_params
    else:
        print("Only watershed seq comm supported")
    alg_run, env_name, config = setup(args.env, hparams, args.algorithm,
                                      args.train_batch_size,
                                      args.num_cpus,
                                      args.num_gpus, args.num_agents,
                                      args.use_gpus_for_workers,
                                      args.use_gpu_for_driver,
                                      args.num_workers_per_device,
                                      args.return_agent_actions,
                                      args.local_rew,
                                      args.local_obs,
                                      args.share_comm_layer)

    # if args.exp_name is None:
    #     exp_name = args.env + '_' + args.algorithm
    # else:
    #     exp_name = args.exp_name

    exp_name = "{}-{}-{}-local_rew-{}-local_obs-{}".format(args.env, args.exp_name, args.algorithm, args.local_rew, args.local_obs)

    print('Commencing experiment', exp_name)

    config['env'] = env_name
    exp_dict = {
            'name': exp_name,
            'run_or_experiment': alg_run,
            "stop": {
                "training_iteration":  args.training_iterations
            },
            'checkpoint_freq': args.checkpoint_frequency,
            "config": config,
            # "local_dir": "/content/gdrive/My Drive/watershed_exps"
            "local_dir": "logs/"
        }



    tune.run(**exp_dict, queue_trials=True)
