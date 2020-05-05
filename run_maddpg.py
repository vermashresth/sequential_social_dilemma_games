import ray
from ray.tune import run_experiments
from ray.tune.registry import register_trainable, register_env
from env import MultiAgentParticleEnv
import ray.rllib.contrib.maddpg.maddpg as maddpg
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import argparse
import gym
import os
from env import WatershedSeqEnv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
n_agents = 4
class myEnv(MultiAgentEnv):
  def __init__(self):
    self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(2,))
    self.action_space = gym.spaces.Box(low=-10, high=10, shape=(2,))

  def reset(self):
    obs={}
    self.counter = 0
    for i in range(1):
      obs[i] = [2,2]
    return obs

  def step(self, action_dict):
    obs, rew, done = {}, {}, {}
    # print(sum(action_dict[0]), sum(action_dict[1]))
    # print(action_dict)
    self.counter+=1
    for i in range(n_agents):
      obs[i] = [2,2]
      rew[i] = i**2
      if self.counter>10:
        done[i] = True
        done["__all__"] = True
      else:
        done[i] = False
        done["__all__"] = False
    return obs, rew, done, {}

class CustomStdOut(object):
    def _log_result(self, result):
        if result["training_iteration"] % 50 == 0:
            try:
                print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                    result["timesteps_total"],
                    result["episodes_total"],
                    result["episode_reward_mean"],
                    result["policy_reward_mean"],
                    round(result["time_total_s"] - self.cur_time, 3)
                ))
            except:
                pass

            self.cur_time = result["time_total_s"]


def parse_args():
    parser = argparse.ArgumentParser("MADDPG with OpenAI MPE")

    # Environment
    parser.add_argument("--scenario", type=str, default="simple",
                        choices=['simple', 'simple_speaker_listener',
                                 'simple_crypto', 'simple_push',
                                 'simple_tag', 'simple_spread', 'simple_adversary'],
                        help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100,
                        help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000,
                        help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0,
                        help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg",
                        help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg",
                        help="policy of adversaries")

    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="discount factor")
    # NOTE: 1 iteration = sample_batch_size * num_workers timesteps * num_envs_per_worker
    parser.add_argument("--sample-batch-size", type=int, default=128,
                        help="number of data points sampled /update /worker")
    parser.add_argument("--train-batch-size", type=int, default=10240,
                        help="number of data points /update")
    parser.add_argument("--n-step", type=int, default=1,
                        help="length of multistep value backup")
    parser.add_argument("--num-units", type=int, default=64,
                        help="number of units in the mlp")
    parser.add_argument("--replay-buffer", type=int, default=10000,
                        help="size of replay buffer in training")

    # Checkpoint
    parser.add_argument("--checkpoint-freq", type=int, default=7500,
                        help="save model once every time this many iterations are completed")
    parser.add_argument("--local-dir", type=str, default="./ray_results",
                        help="path to save checkpoints")
    parser.add_argument("--restore", type=str, default=None,
                        help="directory in which training state and model are loaded")

    # Parallelism
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-envs-per-worker", type=int, default=n_agents)
    parser.add_argument("--num-gpus", type=int, default=0)

    return parser.parse_args()

# from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
def main(args):
    ray.init()
    MADDPGAgent = maddpg.MADDPGTrainer.with_updates(
        mixins=[CustomStdOut]
    )
    register_trainable("MADDPG", MADDPGAgent)

    def env_creater(mpe_args):
        return WatershedSeqEnv(return_agent_actions = False, local_rew = True, local_obs = True)

    register_env("mpe", env_creater)

    env = env_creater({
        "scenario_name": args.scenario,
    })

    def gen_policy(i):
        use_local_critic = [
            args.adv_policy == "ddpg" if i < args.num_adversaries else
            args.good_policy == "ddpg" for i in range(n_agents)
        ]
        return (
            None,
            env.observation_space,
            env.action_space,
            {
                "agent_id": i,
                "use_local_critic": True,
                "obs_space_dict": {i:env.observation_space for i in range(n_agents)},
                "act_space_dict": {i:env.action_space for i in range(n_agents)},
            }
        )

    policies = {"agent-%d" %i: gen_policy(i) for i in range(n_agents)}
    policy_ids = list(policies.keys())

    run_experiments({
        "MADDPG_RLLib": {
            "run": "contrib/MADDPG",
            "env": "mpe",
            "stop": {
                "episodes_total": args.num_episodes,
            },
            "checkpoint_freq": args.checkpoint_freq,
            "local_dir": args.local_dir,
            "restore": args.restore,
            "config": {
                # === Log ===
                "log_level": "WARN",

                # === Environment ===
                "env_config": {
                    "num_agents": n_agents,
                },
                "num_envs_per_worker": args.num_envs_per_worker,
                "horizon": args.max_episode_len,

                # === Policy Config ===
                # --- Model ---
                "good_policy": args.good_policy,
                "adv_policy": args.adv_policy,
                "actor_hiddens": [args.num_units] * 2,
                "actor_hidden_activation": "relu",
                "critic_hiddens": [args.num_units] * 2,
                "critic_hidden_activation": "relu",
                "n_step": args.n_step,
                "gamma": args.gamma,

                # --- Exploration ---
                "tau": 0.01,

                # --- Replay buffer ---
                "buffer_size": args.replay_buffer,

                # --- Optimization ---
                "actor_lr": args.lr,
                "critic_lr": args.lr,
                "learning_starts": args.train_batch_size * args.max_episode_len,
                "sample_batch_size": args.sample_batch_size,
                "train_batch_size": args.train_batch_size,
                "batch_mode": "truncate_episodes",

                # --- Parallelism ---
                "num_workers": args.num_workers,
                "num_gpus": args.num_gpus,
                "num_gpus_per_worker": 0,

                # === Multi-agent setting ===
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": ray.tune.function(
                        lambda i: policy_ids[i]
                    )
                },
            },
        },
    }, verbose=2)


if __name__ == '__main__':
    args = parse_args()
    main(args)
