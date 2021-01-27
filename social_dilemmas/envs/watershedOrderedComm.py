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
        self.my_flow = 1
        if self.local_obs:
            self.n_reqs = 1
        else:
            self.n_reqs = self.num_agents

        self.dones = set()
        self.agentID2real = {0:1, 1:2, 2:4, 3:6}
        if self.return_agent_actions:
            # We will append on some extra values to represent the actions of other agents
            self.observation_space = gym.spaces.Dict({"curr_obs": gym.spaces.Box(low=-250, high=250, shape=(self.n_flows + self.n_reqs + self.my_flow,)),
                         "other_agent_actions": gym.spaces.Box(low=0, high=10, shape=(self.num_agents - 1, ), dtype=np.int32,),
                         "visible_agents": gym.spaces.Box(low=0, high=self.num_agents, shape=(self.num_agents - 1,), dtype=np.int32)})
        else:
            self.observation_space = gym.spaces.Box(low=-250, high=250, shape=(self.n_flows + self.n_reqs + self.my_flow,))
        self.action_space = gym.spaces.Box(low=0, high = 1, shape=(1,))

        self.total_phases = 5
        self.current_phase = 0
        self.agent_in_phases = [[0], [1], [2], [3], [0,1,2,3]]

        self.incoming_flows = {}


    def i2id(self, i):
        return 'agent-'+str(i)

    def find_visible_agents(self, agent_id):
        visible_agents = [1 for i in range(self.num_agents-1)]
        return np.array(visible_agents)

    def set_new_season(self):
        global al
        self.seed = np.random.choice(range(108))
        # self.seed = 0
        ss = self.seed%3
        sa = self.seed//3
        self.Q1 = [160, 115, 80][ss]
        self.Q2 = [65, 50, 35][ss]
        self.S = [15, 12, 10][ss]
        self.al = all_al[sa]

    def get_state(self, id=-1):
        st = [self.Q1, self.Q2, self.S]
        al = self.al[:]
        if id==-1:
            st.extend(al[1:5])
        else:
            st.append(al[self.agentID2real[id]])
        return st

    def get_personal_state(self, id, action_dict, st):
        local_st = st[:]
        if id==0:
            self.incoming_flows[id] = self.Q1
        elif id==1:
            self.incoming_flows[id] = self.incoming_flows[0] * (1 - action_dict[self.i2id(0)])
        elif id==2:
            self.incoming_flows[id] = self.Q2
        elif id==3:
            self.incoming_flows[id] = self.incoming_flows[2] * (1 - action_dict[self.i2id(2)]) + (self.incoming_flows[1]+self.S) * (action_dict[self.i2id(1)])
        else:
            print("Wrong ID")

        local_st.append(self.incoming_flows[id])
        return local_st

    def reset(self):
        global al
        self.dones = set()
        self.current_phase = 0

        self.action_hist = {}

        self.f_rew = [0, 0, 0, 0]
        self.pen = []

        self.n_viol = [0,0,0,0,0,0]
        self.set_new_season()
        #
        # if self.return_agent_actions:
        #     self.prev_actions = defaultdict(lambda: [0] * self.num_agents)

        if not self.local_obs:
            st = self.get_state()
        obs = {}
        self.prev_actions = np.array([0 for _ in range(self.num_agents - 1)]).astype(np.int64)
        for i in self.agent_in_phases[self.current_phase]:
            if self.local_obs:
                st = self.get_state(i)
            special_st = self.get_personal_state(i, self.action_hist, st)
            if self.return_agent_actions:
                # No previous actions so just pass in zeros

                obs[self.i2id(i)] = {"curr_obs": np.array(special_st), "other_agent_actions": self.prev_actions,
                                                "visible_agents": self.find_visible_agents(self.i2id(i))}
            else:
                obs[self.i2id(i)] = np.array(special_st)

        self.current_phase+=1
        return obs

    def get_seed(self):
        return self.seed

    def cal_violations(self, x):
        al = self.al
        x1,x2,x3,x4,x5,x6 = x
        v = [0,0,0,0,0,0,0,0,0]

        # v[0] = al[1]-x1
        # v[1] = al[2]-self.Q1+x1
        # v[2] = x2 - self.S -self.Q1 +x1
        # v[3] = al[4] - x3
        # v[4] = al[3] - x4
        # v[5] = al[4] - self.Q2 + x4
        # v[6] = al[6] - x5
        # v[7] = al[5] - x6
        # v[8] = al[6] - x2 -x3 + x6

        v[0] = al[1] - x1
        v[1] = al[2] - self.incoming_flows[1]
        v[2] = al[3] - x3
        v[3] = al[4] - x4
        v[4] = al[5] - x5
        v[5] = al[6] - x6
        pen = 0
        n_viol = [0, 0, 0, 0, 0, 0]
        t = np.random.randint(100)
        # if t<3:
        #     print("X:", x)
        #     print("Al:", al)
        #     print("Actions:", self.action_hist)
        #     print(self.incoming_flows)
        #     print("Q1, Q2, S", self.Q1, self.Q2, self.S)
        #     print(x1+x4+x6+x5+(self.incoming_flows[1]+self.S)*(1-self.action_hist[self.i2id(1)]*0.1), self.Q1+self.Q2+self.S)
        for i in range(6):
            if v[i]>0:
                n_viol[i]=1
                pen += (v[i]+1)*100
                # if t<3:
                #     print(i, "viol",)
            # if t<3:
            #     print("end")
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
        x1, x2, x4, x6 = list(np.array(actions))
        # x1 = al[1]+(self.Q1-al[2]-al[1])*x1
        # x4 = al[3]+(self.Q2-al[4]-al[3])*x4
        # x2 = (self.S+self.Q1-al[1])*x2
        # x6 = al[5]+(self.S+self.Q1+self.Q2-al[1]-al[3]-al[6]-al[5])*x6

        x1 = self.incoming_flows[0]*x1
        # x2 = self.incoming_flows[1]+(self.S)*x2
        x2 = (self.incoming_flows[1]+self.S)*x2
        x4 = self.incoming_flows[2]*x4
        x6 = self.incoming_flows[3]*x6

        x3 = self.Q2 - x4
        x5 = x2 + x3 - x6

        x = [x1,x2,x3,x4,x5,x6]

        f_rew = []
        for j in range(6):
            f_rew.append(a[j+1]*x[j]**2+b[j+1]*x[j]+c[j+1])
        pen, n_viol = self.cal_violations(x)

        return x, f_rew, pen, n_viol

    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}

        # make action shape normal
        for agent_id in action_dict:
            action_dict[agent_id] = action_dict[agent_id][0]
        # Actions are stored in action history
        for agent_id in action_dict:
            self.action_hist[agent_id] = action_dict[agent_id]

        if self.current_phase<self.total_phases-1:

            if not self.local_obs:
                st = self.get_state()

            for i in self.agent_in_phases[self.current_phase]:
                if self.local_obs:
                    st = self.get_state(i)
                special_st = self.get_personal_state(i, self.action_hist, st)

                if self.local_rew:
                    rewnow = self.f_rew[i] - self.pen
                else:
                    rewnow = sum(self.f_rew) - self.pen

                if self.return_agent_actions:
                    obs[self.i2id(i)] = {"curr_obs": np.array(special_st), "other_agent_actions": self.prev_actions,
                                                    "visible_agents": self.find_visible_agents(self.i2id(i))}
                    rew[self.i2id(i)], done[self.i2id(i)], info[self.i2id(i)] =  rewnow, False, {"viol":0}
                else:
                    obs[self.i2id(i)], rew[self.i2id(i)], done[self.i2id(i)], info[self.i2id(i)] = np.array(special_st), rewnow, False, {"viol":None}

            done["__all__"] = False

        else:
            # Final step on mini loop, all obs, rewards must be returned
            _, self.f_rew, self.pen, self.n_viol = self.cal_rewards(self.action_hist)

            if not self.local_obs:
                st = self.get_state()
            for i in self.agent_in_phases[self.current_phase]:
                if self.local_obs:
                    st = self.get_state(i)

                special_st = self.get_personal_state(i, self.action_hist, st)

                if self.local_rew:
                    rewnow = self.f_rew[i] - self.pen
                else:
                    rewnow = sum(self.f_rew) - self.pen

                if self.return_agent_actions:
                    prev_actions = np.array([self.action_hist[key] for key in sorted(self.action_hist.keys())
                                             if key != self.i2id(i)]).astype(np.int64)
                    obs[self.i2id(i)] = {"curr_obs": np.array(special_st), "other_agent_actions": prev_actions,
                                                    "visible_agents": self.find_visible_agents(self.i2id(i))}
                    rew[self.i2id(i)], done[self.i2id(i)], info[self.i2id(i)] =  rewnow, True, {"viol":self.n_viol}
                else:
                    obs[self.i2id(i)], rew[self.i2id(i)], done[self.i2id(i)], info[self.i2id(i)] = np.array(special_st), rewnow, True, {"viol":self.n_viol}
            done["__all__"] = True
        self.current_phase+=1
        return obs, rew, done, info


class WatershedSeqEnv(WatershedEnv):
    def __init__(self, *args, **kwargs):
        super(WatershedSeqEnv, self).__init__(*args, **kwargs)
        self.end_episode = False
        self.internal_step = 0;
        self.max_steps = 10;
        self.mybigreq = [24*10, 40*10, 24*10, 10*10]
        self.current_sums = [0,0,0,0]

        self.total_phases = 4
        self.current_phase = 0
        self.agent_in_phases = [[0], [1], [2], [3]]

    def reset(self):
        global al
        self.dones = set()
        self.current_phase = 0

        self.internal_step = 0
        self.current_sums = [0,0,0,0]
        self.end_episode = False

        self.action_hist = {}

        self.f_rew = [0, 0, 0, 0]
        self.pen = 0
        self.n_viol = [0,0,0,0,0,0]
        self.rew_sum_keeper = [0,0,0,0]
        self.set_new_season()


        if not self.local_obs:
            st = self.get_state()
        obs = {}

        self.prev_actions = [np.array([0 for _ in range(self.num_agents - 1)]).astype(np.int64) for _ in range(self.num_agents)]

        for i in self.agent_in_phases[self.current_phase]:
            if self.local_obs:
                st = self.get_state(i)
            special_st = self.get_personal_state(i, self.action_hist, st)
            if self.return_agent_actions:
                # No previous actions so just pass in zeros

                obs[self.i2id(i)] = {"curr_obs": np.array(special_st), "other_agent_actions": self.prev_actions[i],
                                                "visible_agents": self.find_visible_agents(self.i2id(i))}
            else:
                obs[self.i2id(i)] = np.array(special_st)

        self.current_phase+=1
        return obs

    def step(self, action_dict):

        # make action shape normal
        for agent_id in action_dict:
            action_dict[agent_id] = action_dict[agent_id][0]

        # Actions are stored in action history
        for agent_id in action_dict:
            self.action_hist[agent_id] = action_dict[agent_id]



        obs, rew, done, info = {}, {}, {}, {}



        if self.current_phase==self.total_phases:
            x, self.f_rew, self.pen, self.n_viol = self.cal_rewards(self.action_hist)
            x1,x2,x3,x4,x5,x6 = x
            self.prev_actions = [np.array([self.action_hist[key] for key in sorted(self.action_hist.keys())
                                     if key != self.i2id(i)]).astype(np.int64) for i in range(self.num_agents)]
            self.current_sums = list(np.array(self.current_sums) + np.array([x1, x2, x4, x6]))

            self.internal_step+=1
            p = np.random.randint(10000)
            if p<3:
                print("All rewards: ", self.f_rew)
                print("Penelaty: ", self.pen, " Violations: ", self.n_viol, " Total Viol: ", sum(self.n_viol))
                print("Current sums: ",self.current_sums)
                print("Big Reqs: ", self.mybigreq)
                print("X: ", x)
                print("Actions: ", self.action_hist)

            self.current_phase%=self.total_phases

        if self.internal_step>=self.max_steps:
            self.end_episode = True

        if self.end_episode and self.current_phase==self.total_phases-1:
            done["__all__"] = True
        else:
            done["__all__"] = False

        if not self.local_obs:
            st = self.get_state()

        for i in self.agent_in_phases[self.current_phase]:

            if self.local_obs:
                st = self.get_state(i)
            special_st = self.get_personal_state(i, self.action_hist, st)

            temp = []
            if self.end_episode:
                for j in range(self.num_agents):
                    temp.append(self.current_sums[j]/self.mybigreq[j]*100)

            if self.local_rew:
                rewnow = self.f_rew[i] - self.pen
                if self.end_episode:
                    rewnow += temp[i]
            else:
                rewnow = sum(self.f_rew) - self.pen
                if self.end_episode:
                    rewnow+=sum(temp)

            self.rew_sum_keeper[i]+=rewnow

            if self.return_agent_actions:

                obs[self.i2id(i)] = {"curr_obs": np.array(special_st), "other_agent_actions": self.prev_actions[i],"visible_agents": self.find_visible_agents(self.i2id(i))}
            else:
                obs[self.i2id(i)] = np.array(special_st)
            rew[self.i2id(i)], done[self.i2id(i)], info[self.i2id(i)] = rewnow, self.end_episode, {"viol":self.n_viol, "temp":sum(temp), "acts":self.action_hist, 'end':self.end_episode,'true_end':done["__all__"], "running_rew":self.rew_sum_keeper}



        if self.end_episode and done["__all__"]:
          p=np.random.randint(10000)
          if p<3:
            print("total episode rewards", self.rew_sum_keeper)
            print("total sum", self.current_sums)
            print("sum based rew", temp)
            print()

        self.current_phase+=1
        return obs, rew, done, info


class WatershedSeqCommEnv(WatershedSeqEnv):
    def __init__(self, *args, **kwargs):
        super(WatershedSeqCommEnv, self).__init__(*args, **kwargs)

        self.comm_agents = 4
        self.action_agents = 4
        self.num_agents = self.comm_agents + self.action_agents

        if self.return_agent_actions:
            # We will append on some extra values to represent the actions of other agents
            self.observation_space = gym.spaces.Dict({"curr_obs": gym.spaces.Box(low=-250, high=250, shape=(self.n_flows+self.n_reqs+ self.comm_agents+self.my_flow,)),
                         "other_agent_actions": gym.spaces.Box(low=0, high=10, shape=(self.action_agents - 1, ), dtype=np.int32,),
                         "visible_agents": gym.spaces.Box(low=0, high=self.num_agents, shape=(self.action_agents - 1,), dtype=np.int32)})
        else:
            self.observation_space = gym.spaces.Box(low=-250, high=250, shape=(self.n_flows+self.n_reqs+ self.comm_agents + self.my_flow,))


        self.observation_space_comm = gym.spaces.Box(low=-250, high=250, shape=(self.n_flows+self.n_reqs+self.comm_agents,))
        self.action_space_comm = gym.spaces.Discrete(5)

        if self.local_obs:
            self.prev_obs = [None]*self.comm_agents
        self.comm_phases = 2
        self.total_phases = self.comm_agents*self.comm_phases+self.action_agents
        self.current_phase = 0
        self.agent_in_phases = [[0], [1], [2], [3],[0], [1], [2], [3], [4],[5],[6],[7]]

    def find_visible_agents(self, agent_id):
        visible_agents = [1 for i in range(self.action_agents-1)]
        return np.array(visible_agents)

    def get_personal_state(self, actual_id, action_dict, st):
        local_st = st[:]
        offset  = self.comm_agents
        if actual_id >=self.comm_agents:
            id = actual_id-offset
            if id==0:
                self.incoming_flows[id] = self.Q1
            elif id==1:
                self.incoming_flows[id] = self.incoming_flows[0] * (1 - action_dict[self.i2id(offset+0)])
            elif id==2:
                self.incoming_flows[id] = self.Q2
            elif id==3:
                self.incoming_flows[id] = self.incoming_flows[2] * (1 - action_dict[self.i2id(offset+2)]) + (self.incoming_flows[1]+self.S) * (action_dict[self.i2id(offset+1)])
            else:
                print("Wrong ID")

            local_st.append(self.incoming_flows[id])
        return local_st

    def reset(self):
        global al
        self.dones = set()
        self.current_phase = 0

        self.internal_step = 0
        self.current_sums = [0,0,0,0]
        self.end_episode = False

        self.action_hist = {}
        self.watershed_action_hist = {}

        self.f_rew = [0, 0, 0, 0]
        self.pen = 0
        self.n_viol = [0,0,0,0,0,0]
        self.rew_sum_keeper = [0,0,0,0]
        self.set_new_season()

        self.firstCommStep = True
        self.lastCommStep = False
        if not self.local_obs:
            st = self.get_state()
        obs = {}

        self.prev_actions = [np.array([0 for _ in range(self.action_agents - 1)]).astype(np.int64) for _ in range(self.action_agents)]

        for i in self.agent_in_phases[self.current_phase]:
            if self.local_obs:
                st = self.get_state(i%self.comm_agents)
            special_st = self.get_personal_state(i, self.action_hist, st)
            # if self.return_agent_actions:
            #     # No previous actions so just pass in zeros
            #
            #     obs[self.i2id(i)] = {"curr_obs": np.array(special_st), "other_agent_actions": self.prev_actions[i%self.comm_agents],
            #                                     "visible_agents": self.find_visible_agents(self.i2id(i))}
            # else:
            if i < self.comm_agents:
                comm_actions = [-1 for i in range(self.comm_agents)]
                special_st.extend(comm_actions)
                obs[self.i2id(i)] = np.array(special_st)
            else:
                print("Wrong agent id, should start with comm agent")
                return

        self.current_phase+=1
        return obs

    def step(self, action_dict):
        # make action shape normal
        for agent_id in action_dict:
            if int(agent_id[-1])>=self.comm_agents:
              action_dict[agent_id] = action_dict[agent_id][0]
        # Actions are stored in action history
        for agent_id in action_dict:
            self.action_hist[agent_id] = action_dict[agent_id]
            if int(agent_id[-1])>=self.comm_agents:
                self.watershed_action_hist[agent_id] = action_dict[agent_id]



        obs, rew, done, info = {}, {}, {}, {}

        if self.current_phase==self.comm_agents:
            self.firstCommStep = False
        if self.current_phase//self.comm_agents == self.comm_phases-1:
            self.lastCommStep = True

        if self.current_phase==self.total_phases:
            x, self.f_rew, self.pen, self.n_viol = self.cal_rewards(self.watershed_action_hist)



            x1,x2,x3,x4,x5,x6 = x
            self.prev_actions = [np.array([self.watershed_action_hist[key] for key in sorted(self.watershed_action_hist.keys())
                                     if key != i]).astype(np.int64) for i in self.watershed_action_hist.keys()]
            self.current_sums = list(np.array(self.current_sums) + np.array([x1, x2, x4, x6]))

            p = np.random.randint(10000)
            if p<3:
                print("All rewards: ", self.f_rew)
                print("Penelaty: ", self.pen, " Violations: ", self.n_viol, " Total Viol: ", sum(self.n_viol))
                print("Current sums: ",self.current_sums)
                print("Big Reqs: ", self.mybigreq)
                print("X: ", x)
                print("Actions: ", self.action_hist)
            self.internal_step+=1

            self.current_phase%=self.total_phases
            self.firstCommStep = True
            self.lastCommStep = False


        if self.internal_step>=self.max_steps:
            self.end_episode = True

        if self.end_episode and self.current_phase==self.total_phases-1:
            done["__all__"] = True
        else:
            done["__all__"] = False

        if not self.local_obs:
            st = self.get_state()

        for i in self.agent_in_phases[self.current_phase]:

            if self.local_obs:
                st = self.get_state(i%self.comm_agents)
            special_st = self.get_personal_state(i, self.action_hist, st)

            temp = []
            if self.end_episode:
                for j in range(self.action_agents):
                    temp.append(self.current_sums[j]/self.mybigreq[j]*100)

            if self.local_rew:
                rewnow = self.f_rew[i%self.comm_agents] - self.pen
                if self.end_episode:
                    rewnow += temp[i%self.comm_agents]
            else:
                rewnow = sum(self.f_rew) - self.pen
                if self.end_episode:
                    rewnow+=sum(temp)
            if i>=self.comm_agents:
                self.rew_sum_keeper[i%self.comm_agents]+=rewnow

            if i < self.comm_agents:
                if self.firstCommStep:
                    comm_actions = [-1 for i in range(self.comm_agents)]
                else:
                    comm_actions = [self.action_hist[self.i2id(i)] for i in range(self.comm_agents)]
                special_st.extend(comm_actions)
                obs[self.i2id(i)] = np.array(special_st)

                if self.firstCommStep:
                    rew[self.i2id(i)] = rewnow
                else:
                    rew[self.i2id(i)] = 0
                if self.end_episode and self.lastCommStep:
                  done[self.i2id(i)] = True
                else:
                  done[self.i2id(i)] = False
            else:
                comm_actions = [self.action_hist[self.i2id(i)] for i in range(self.comm_agents)]
                special_st.extend(comm_actions)
                if self.return_agent_actions:
                    obs[self.i2id(i)] = {"curr_obs": np.array(special_st), "other_agent_actions": self.prev_actions[i%self.comm_agents],"visible_agents": self.find_visible_agents(self.i2id(i))}

                else:
                    obs[self.i2id(i)] = np.array(special_st)
                rew[self.i2id(i)], done[self.i2id(i)] = rewnow, self.end_episode
            info[self.i2id(i)] = {"viol":self.n_viol, "temp":sum(temp), "acts":self.action_hist, 'end':self.end_episode, 'true_end':done["__all__"], 'running_rew':self.rew_sum_keeper}


        if self.end_episode and done["__all__"]:
          p=np.random.randint(10000)
          if p<3:
            print("total episode rewards", self.rew_sum_keeper)
            print("total sum", self.current_sums)
            print("sum based rew", temp)
            print()

        self.current_phase+=1
        return obs, rew, done, info
