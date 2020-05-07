import numpy as np
import math
def on_episode_start(info):
    # print(dir(info['episode']))
    info['episode'].user_data["viol"] = []
    info['episode'].user_data["rew"] = []

def on_episode_step(info):
    if info['episode'].last_info_for('agent-0') is not None:
      if "viol" in info['episode'].last_info_for('agent-0').keys():
        viol = info['episode'].last_info_for('agent-0')["viol"]
        info['episode'].user_data["viol"].append(viol)
    if info['episode'].last_info_for('agent-3') is not None:
        if "true_end" in info['episode'].last_info_for('agent-3').keys():
            if info['episode'].last_info_for('agent-3')["true_end"]:
                info['episode'].user_data["total_rews"] = info['episode'].last_info_for('agent-3')["running_rew"][:]

def on_episode_step_comm(info):
    if info['episode'].last_info_for('agent-0') is not None:
      if "viol" in info['episode'].last_info_for('agent-0').keys():
        viol = info['episode'].last_info_for('agent-0')["viol"]
        info['episode'].user_data["viol"].append(viol)
    if info['episode'].last_info_for('agent-7') is not None:
        if "true_end" in info['episode'].last_info_for('agent-7').keys():
            if info['episode'].last_info_for('agent-7')["true_end"]:
                info['episode'].user_data["total_rews"] = info['episode'].last_info_for('agent-7')["running_rew"][:]

def on_episode_end(info):
  info['episode'].custom_metrics["violations"] = np.mean(info['episode'].user_data["viol"])
  total_rews = info['episode'].user_data["total_rews"]
  num = 0
  for i in total_rews:
      for j in total_rews:
          num+=math.fabs(i-j)
  den = 2*sum(total_rews)
  info['episode'].custom_metrics["Equality"] = 1-num/den
  info['episode'].custom_metrics["Utility"] = den

# def on_train_result(info):
#     utility = 0
    # for i in info["result"]["policy_reward_mean"]:
    #   utility+=info["result"]["policy_reward_mean"][i]
    # # # you can mutate the result dict to add new fields to return
    # # info["result"]["callback_ok"] = True
    # info["result"]["utility"] = utility
