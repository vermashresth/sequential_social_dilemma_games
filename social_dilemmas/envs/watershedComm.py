from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import gym
from collections import defaultdict

a = [0, -.2, -.06, -.29, -.13, -.056, -.15]
b = [0, 6, 2.5, 6.28, 6, 3.74, 7.6]
c = [0, -5, 0, -3, -6, -23, -15]


from itertools import product
# B = [[0], range(8, 40, 8), range(0, 40, 8), range(8, 40, 8), range(6, 40, 8), [15], [10] ]
# all_al = list(product(*B))
# B = [[0], range(8, 24, 8), range(8, 40, 8), range(8, 20, 8), range(8, 24, 8), [15], [10] ]
# all_al = list(product(*B))
B = [[0], range(8, 24, 8), range(8, 30, 8), [8], range(8, 24, 8), [15], range(8, 30, 8) ]
all_al = list(product(*B))
# all_al = [[0, 12, 10, 8, 6, 15, 10],
#           [10, 40, 0, 80, 1, 45, 30],
#           [20, 42, 10, 30, 60, 5, 0],
#           [30, 2, 0, 5, 10, 20, 0],
#           [40, 22, 60, 28, 36, 45, 40]]
# al = [0, 12, 10, 8, 6, 15, 10]

big_req = [24*40, 30*40, 24*40, 30*40]

class WatershedEnv(MultiAgentEnv):
    def __init__(self, return_agent_actions = False, local_rew = False, local_obs = False):

        self.local_obs = local_obs
        self.local_rew = local_rew
        self.return_agent_actions = return_agent_actions

        self.num_agents = 4
        self.n_flows = 3

        if self.local_obs:
            self.n_reqs = 1
        else:
            self.n_reqs = self.num_agents

        self.dones = set()
        self.agentID2real = {0:1, 1:2, 2:4, 3:6}
        if self.return_agent_actions:
            # We will append on some extra values to represent the actions of other agents
            self.observation_space = gym.spaces.Dict({"curr_obs": gym.spaces.Box(low=-200, high=200, shape=(self.n_flows + self.n_reqs,)),
                         "other_agent_actions": gym.spaces.Box(low=0, high=10, shape=(self.num_agents - 1, ), dtype=np.int32,),
                         "visible_agents": gym.spaces.Box(low=0, high=self.num_agents, shape=(self.num_agents - 1,), dtype=np.int32)})
        else:
            self.observation_space = gym.spaces.Box(low=-200, high=200, shape=(self.n_flows + self.n_reqs,))
        self.action_space = gym.spaces.Discrete(10)



        if self.return_agent_actions:
            self.prev_actions = defaultdict(lambda: [0] * self.num_agents)

    def i2id(self, i):
        return 'agent-'+str(i)

    def find_visible_agents(self, agent_id):
        visible_agents = [1 for i in range(self.num_agents-1)]
        return np.array(visible_agents)

    def get_new_state(self, id=-1):
        global al
        self.seed = np.random.choice(range(108))
        # self.seed = 0
        ss = self.seed%3
        sa = self.seed//3
        self.Q1 = [160, 115, 80][ss]
        self.Q2 = [65, 50, 35][ss]
        self.S = [15, 12, 10][ss]
        al = all_al[sa]

        st = [self.Q1, self.Q2, self.S]

        if id==-1:
            st.extend(al[1:5])
        else:
            st.append(al[self.agentID2real[id]])
        return st

    def reset(self):
        global al
        self.dones = set()
        if not self.local_obs:
            st = self.get_new_state()
        obs = {}
        prev_actions = np.array([0 for _ in range(self.num_agents - 1)]).astype(np.int64)
        for i in range(self.num_agents):
            if self.local_obs:
                st = self.get_new_state(i)
            if self.return_agent_actions:
                # No previous actions so just pass in zeros

                obs[self.i2id(i)] = {"curr_obs": np.array(st), "other_agent_actions": prev_actions,
                                                "visible_agents": self.find_visible_agents(self.i2id(i))}
            else:
                obs[self.i2id(i)] = np.array(st)
        return obs

    def get_seed(self):
        return self.seed

    def cal_violations(self, x):
        x1,x2,x3,x4,x5,x6 = x
        v = [0,0,0,0,0,0,0,0,0]

        v[0] = al[1]-x1
        v[1] = al[2]-self.Q1+x1
        v[2] = x2 - self.S -self.Q1 +x1
        v[3] = al[4] - x3
        v[4] = al[3] - x4
        v[5] = al[4] - self.Q2 + x4
        v[6] = al[6] - x5
        v[7] = al[5] - x6
        v[8] = al[6] - x2 -x3 + x6

        pen = 0
        n_viol = 0
        for i in range(9):
            if v[i]>0:
                n_viol+=1
                pen += (v[i]+1)*100
        return pen, n_viol

    def get_observation_space(self, t=0):
        if t==0:
            return self.observation_space
        else:
            return self.observation_space_comm

    def get_action_space(self, t=0):
        if t==0:
            return self.action_space
        else:
            return self.action_space_comm

    def cal_rewards(self, action_dict):
        actions = list(action_dict.values())
        x1, x2, x4, x6 = list(np.array(actions)*0.1)
        x1 = al[1]+(self.Q1-al[2]-al[1])*x1
        x4 = al[3]+(self.Q2-al[4]-al[3])*x4
        x2 = (self.S+self.Q1-al[1])*x2
        x6 = al[5]+(self.S+self.Q1+self.Q2-al[1]-al[3]-al[6]-al[5])*x6

        x3 = self.Q2 - x4
        x5 = x3 + x3 - x6

        x = [x1,x2,x3,x4,x5,x6]

        f_rew = []
        for j in range(6):
            f_rew.append(a[j+1]*x[j]**2+b[j+1]*x[j]+c[j+1])
        pen, n_viol = self.cal_violations(x)

        return x, f_rew, pen, n_viol

    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}

        _, f_rew, pen, n_viol = self.cal_rewards(action_dict)

        if not self.local_obs:
            st = self.get_new_state()
        for i in range(self.num_agents):
            if self.local_obs:
                st = self.get_new_state(i)
            if self.local_rew:
                rewnow = f_rew[i] - pen
            else:
                rewnow = sum(f_rew) - pen
            if self.return_agent_actions:
                prev_actions = np.array([action_dict[key] for key in sorted(action_dict.keys())
                                         if key != self.i2id(i)]).astype(np.int64)
                obs[self.i2id(i)] = {"curr_obs": np.array(st), "other_agent_actions": prev_actions,
                                                "visible_agents": self.find_visible_agents(self.i2id(i))}
                rew[self.i2id(i)], done[self.i2id(i)], info[self.i2id(i)] =  rewnow, True, {"viol":n_viol}
            else:
                obs[self.i2id(i)], rew[self.i2id(i)], done[self.i2id(i)], info[self.i2id(i)] = np.array(st), rewnow, True, {"viol":n_viol}
            self.dones.add(i)
        done["__all__"] = True

        return obs, rew, done, info


class WatershedSeqEnv(WatershedEnv):
    def __init__(self, *args, **kwargs):
        super(WatershedSeqEnv, self).__init__(*args, **kwargs)
        self.end_episode = False
        self.internal_step = 0;
        self.max_steps = 10;
        self.mybigreq = [24*10, 40*10, 24*10, 10*10]
        self.current_sums = [0,0,0,0]

    def reset(self, full_reset = True):
        global al
        self.dones = set()

        obs = {}
        if not self.local_obs:
            st = self.get_new_state()

        self.internal_step = 0
        self.current_sums = [0,0,0,0]
        self.end_episode = False
        prev_actions = np.array([0 for _ in range(self.num_agents - 1)]).astype(np.int64)

        for i in range(self.num_agents):
            if self.local_obs:
                st = self.get_new_state(i)
            if self.return_agent_actions:
                # No previous actions so just pass in zeros

                obs[self.i2id(i)] = {"curr_obs": np.array(st), "other_agent_actions": prev_actions,
                                                "visible_agents": self.find_visible_agents(self.i2id(i))}
            else:
                obs[self.i2id(i)] = np.array(st)
        return obs

    def step(self, action_dict):

        if self.internal_step>=self.max_steps:
            self.end_episode = True

        obs, rew, done, info = {}, {}, {}, {}
        x, f_rew, pen, n_viol = self.cal_rewards(action_dict)
        x1,x2,x3,x4,x5,x6 = x

        self.current_sums = list(np.array(self.current_sums) + np.array([x1, x2, x4, x6]))

        if not self.local_obs:
            new_st = self.get_new_state()

        for i in range(self.num_agents):

            if self.local_obs:
                new_st = self.get_new_state(i)

            temp = []
            if self.end_episode:
                for j in range(self.num_agents):
                    temp.append(self.current_sums[j]/self.mybigreq[j]*100)

            if self.local_rew:
                rewnow = f_rew[i] - pen
                if self.end_episode:
                    rewnow += temp[i]
            else:
                rewnow = sum(f_rew) - pen
                if self.end_episode:
                    rewnow+=sum(temp)

            if self.return_agent_actions:
                prev_actions = np.array([action_dict[key] for key in sorted(action_dict.keys())
                                         if key != self.i2id(i)]).astype(np.int64)
                obs[self.i2id(i)] = {"curr_obs": np.array(new_st), "other_agent_actions": prev_actions,"visible_agents": self.find_visible_agents(self.i2id(i))}
                rew[self.i2id(i)], done[self.i2id(i)], info[self.i2id(i)] = rewnow, self.end_episode, {"viol":n_viol, "temp":sum(temp), "acts":action_dict, 'end':self.end_episode}
            else:
                obs[self.i2id(i)], rew[self.i2id(i)], done[self.i2id(i)], info[self.i2id(i)] = np.array(new_st), rewnow, self.end_episode, {"viol":n_viol, "temp":sum(temp), "acts":action_dict, 'end':self.end_episode}
            if self.end_episode:
                self.dones.add(i)

        done["__all__"] = self.end_episode
        self.internal_step+=1
        return obs, rew, done, info


class WatershedSeqCommEnv(WatershedSeqEnv):
    def __init__(self, *args, **kwargs):
        super(WatershedSeqCommEnv, self).__init__(*args, **kwargs)

        self.comm_agents = 4
        self.action_agents = 4
        self.num_agents = self.comm_agents + self.action_agents

        if self.return_agent_actions:
            # We will append on some extra values to represent the actions of other agents
            self.observation_space = gym.spaces.Dict({"curr_obs": gym.spaces.Box(low=-200, high=200, shape=(self.n_flows+self.n_reqs+ self.comm_agents,)),
                         "other_agent_actions": gym.spaces.Box(low=0, high=10, shape=(self.action_agents - 1, ), dtype=np.int32,),
                         "visible_agents": gym.spaces.Box(low=0, high=self.num_agents, shape=(self.action_agents - 1,), dtype=np.int32)})
        else:
            self.observation_space = gym.spaces.Box(low=-200, high=200, shape=(self.n_flows+self.n_reqs+ self.comm_agents,))


        self.observation_space_comm = gym.spaces.Box(low=-200, high=200, shape=(self.n_flows+self.n_reqs,))
        self.action_space_comm = gym.spaces.Discrete(3)

        if self.local_obs:
            self.prev_obs = [None]*self.comm_agents

    def find_visible_agents(self, agent_id):
        visible_agents = [1 for i in range(self.action_agents-1)]
        return np.array(visible_agents)

    def reset(self, full_reset = True):
        global al
        self.dones = set()

        obs = {}
        if not self.local_obs:
            st = self.get_new_state()
            self.prev_obs = st[:]

        self.internal_step = 0
        self.current_sums = [0,0,0,0]
        self.end_episode = False
        prev_actions = np.array([0 for _ in range(self.action_agents - 1)]).astype(np.int64)

        for i in range(self.num_agents):
            if self.local_obs:
                st = self.get_new_state(i%self.comm_agents)
                self.prev_obs[i%self.comm_agents] = st[:]
            if self.return_agent_actions:
                # No previous actions so just pass in zeros
                if i<self.comm_agents:
                    #comm
                    obs[self.i2id(i)] = np.array(st)
                else:
                    new_st = st[:]
                    new_st.extend([-1 for i in range(self.comm_agents)])
                    obs[self.i2id(i)] = {"curr_obs": np.array(new_st), "other_agent_actions": prev_actions,
                                                "visible_agents": self.find_visible_agents(self.i2id(i))}
            else:
                if i<self.comm_agents:
                    #comm
                    obs[self.i2id(i)] = np.array(st)
                else:
                    new_st = st[:]
                    new_st.extend([-1 for i in range(self.comm_agents)])
                    obs[self.i2id(i)] = np.array(new_st)
        return obs

    def step(self, action_dict):
        if self.internal_step>=self.max_steps:
            self.end_episode = True
        obs, rew, done, info = {}, {}, {}, {}
        only_actions = {i:action_dict[i] for i in action_dict if int(i[-1])>=self.action_agents}
        x, f_rew, pen, n_viol = self.cal_rewards(only_actions)
        x1,x2,x3,x4,x5,x6 = x

        self.current_sums = list(np.array(self.current_sums) + np.array([x1, x2, x4, x6]))

        if not self.local_obs:
            new_st = self.get_new_state()
            old_st = self.prev_obs.copy()
            self.prev_obs = new_st.copy()

        comm_actions = []
        for i in range(self.num_agents//2):
            comm_actions.append(action_dict['agent-{}'.format(4+i)])

        for i in range(self.num_agents):
            temp = []

            if self.local_obs:
                new_st = self.get_new_state(i%self.comm_agents)
                old_st = self.prev_obs[i%self.comm_agents].copy()
                self.prev_obs[i%self.comm_agents] = new_st.copy()

            if self.end_episode:
                for j in range(self.comm_agents):
                    temp.append(self.current_sums[j]/self.mybigreq[j]*100)

            if self.local_rew:
                rewnow = f_rew[i%self.comm_agents] - pen
                if self.end_episode:
                    rewnow += temp[i%self.comm_agents]
            else:
                rewnow = sum(f_rew) - pen
                if self.end_episode:

                    rewnow+=sum(temp)

            if self.return_agent_actions:
                if i < self.comm_agents:
                    obs[self.i2id(i)] = np.array(new_st)
                else:
                    aug_state = old_st[:]
                    aug_state.extend(comm_actions)
                    prev_actions = np.array([only_actions[key] for key in sorted(only_actions.keys())
                                             if key != self.i2id(i)]).astype(np.int64)
                    obs[self.i2id(i)] = {"curr_obs": np.array(aug_state), "other_agent_actions": prev_actions,"visible_agents": self.find_visible_agents(self.i2id(i))}
            else:
                if i < self.comm_agents:
                    obs[self.i2id(i)] = np.array(new_st)
                else:
                    aug_state = old_st[:]
                    aug_state.extend(comm_actions)
                    obs[self.i2id(i)] = np.array(aug_state)
            rew[self.i2id(i)], done[self.i2id(i)], info[self.i2id(i)] = rewnow, self.end_episode, {"viol":n_viol, "temp":sum(temp), "acts":action_dict, 'end':self.end_episode}
            if self.end_episode:
                self.dones.add(i)

        done["__all__"] = self.end_episode
        self.internal_step+=1
        return obs, rew, done, info
