import numpy as np
# import torch
# from torch.distributions import one_hot_categorical
import time
from copy import deepcopy
from smac.env import StarCraft2Env
# from torch.multiprocessing import Pipe, Process
from multiprocessing import Process
import time

class Actor(Process):
    def __init__(self, args, child_conn, env_idx, buffer_out,wk_id):
        #守护进程
        super(Actor, self).__init__(daemon=True)
        # self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon

        self.child_conn = child_conn
        self.buffer_out = buffer_out
        self.env_idx = env_idx
        self.wk_id=wk_id
        print('Init Actor')

    def generate_episode(self, episode_num=None, evaluate=False):
        # if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
        #     self.env.close()
        # roll_start = time.time()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        #重置环境
        self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        # self.agents.policy.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)
            # obs是全部agents的观测，不是单独一个agent  (n_agents,obs_shape)
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []
            # 每个agent选择一个动作，所有agent都选择一个动作之后，也就是这个time_step结束
            #才会开始下一个time_step
            for agent_id in range(self.n_agents):
                # 一维，（n_actions，）
                avail_action = self.env.get_avail_agent_actions(agent_id)

                #这个管道不是用来传输episode的，是用来传输参数以便于获得动作的
                self.child_conn.send([obs[agent_id], last_action[agent_id], agent_id,
                                     avail_action, epsilon, evaluate, False])
                # print('agent :', agent_id)
                # action是一个int
                action = self.child_conn.recv()
                # action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                #                                        avail_action, epsilon, evaluate)
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            reward, terminated, info = self.env.step(actions)
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            o.append(obs)
            s.append(state)
            # 联合动作
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)

            r.append([reward])
            # print(r)
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # last obs
        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # print('padding', len(episode['padded'])) 60
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        self.child_conn.send([0, 0, 0, 0, 0, 0, terminated])
        #这里deepcopy了一次 将一整个episode的数据传入buffer管道
        self.buffer_out.send([deepcopy(episode), episode_reward, win_tag])
        # roll_end = time.time()
        # print('##rollout time is ：', roll_end - roll_start)
        #print(np.array(episode['o']).shape)
        return episode, episode_reward, win_tag

    def run(self):
        #加载父类的run方法
        super(Actor, self).run()
        # get agent obs type
        self.env = StarCraft2Env(map_name=self.args.map,
                            step_mul=self.args.step_mul,
                            difficulty=self.args.difficulty,
                            game_version=self.args.game_version,
                            replay_dir=self.args.replay_dir)
        #//是整除   平均给每个env需要产生的episode的个数
        rollouts_number= self.args.n_epoch // (self.args.env_nums * self.args.worker_nums)
        for episode_idx in range(rollouts_number):
            #episode是每个env独立生产的，所以各个env之间的序号是无序的 每产生一条就print一次
            print('ID {} Env {} run episode {}'.format(self.wk_id,self.env_idx, episode_idx))
            _, _, _ = self.generate_episode(episode_idx, False)
        #     if episode_idx % self.args.evaluate_cycle == 0 and episode_idx != 0:
        #         win_rate, episode_reward = self.evaluate()
        #         print('win_rate is ', win_rate)
        #         self.win_rates.append(win_rate)
        #         self.episode_rewards.append(episode_reward)
        #         self.plt(self.env_idx)
        # self.plt(self.env_idx)
        self.env.close()

