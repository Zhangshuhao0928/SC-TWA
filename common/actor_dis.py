# --- coding : utf-8 ---
# @Author   : Erhaoo
# @Time     : 2022/7/23  19:52
# @File     : actor_dis.py
# @Software : PyCharm

import numpy as np
import torch
# from torch.distributions import one_hot_categorical
import time
from copy import deepcopy
from smac.env import StarCraft2Env
import time
from threading import Thread,Lock
from policy import qmix_model
import matplotlib.pyplot as plt
import os
from torch.multiprocessing import Process

class Actor_dis(Thread):
    def __init__(self, args, env_idx,data_queue,share_model,ac_id,buffer,rank):
        super(Actor_dis,self).__init__(daemon=True)  #守护线程
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon

        self.env_idx = env_idx
        self.n_epoch=self.args.n_epoch
        self.n_env=self.args.env_nums
        self.data_queue=data_queue
        self.share_model=share_model
        self.ac_id=ac_id
        self.buffer=buffer
        self.rank=rank
        self.eval_hidden=None
        self.experience_batch=[]

        self.lock=Lock()

        self.actor_agent = qmix_model.QmixModel(args)
        self.rollouts_number = self.args.n_epoch//args.env_nums
        self.env = StarCraft2Env(map_name=self.args.map,
                                 step_mul=self.args.step_mul,
                                 difficulty=self.args.difficulty,
                                 game_version=self.args.game_version,
                                 replay_dir=self.args.replay_dir)

        self.actor_agent.create_hidden(1)


        print('Init Actor')



    def generate_episode(self, episode_num=None, evaluate=False):
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        #重置环境
        self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        while not terminated and step < self.episode_limit:
            # obs是全部agents的观测，不是单独一个agent  (n_agents,obs_shape)
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []
            # 每个agent选择一个动作，所有agent都选择一个动作之后，也就是这个time_step结束
            #才会开始下一个time_step
            for agent_id in range(self.n_agents):
                # 一维，（n_actions，）
                avail_action = self.env.get_avail_agent_actions(agent_id)

                # action是一个int
                action = self.actor_agent.choose_action(obs[agent_id], last_action[agent_id],agent_id,
                                                                 avail_action, epsilon, self.env_idx, evaluate=False)
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

        return episode, episode_reward, win_tag

    def run(self):
        n_rollout = 0
        while n_rollout < self.n_epoch // self.n_env:

            episode, episode_reward, win_tag=self.generate_episode(n_rollout,False)
            self.data_queue.put([episode_reward, win_tag])
            self.experience_batch.append(episode)
            print('Rank {} Env {} run episode {}'.format(self.rank,self.ac_id, n_rollout))
            self.actor_agent.restart_hidden(self.env_idx, 1)

            self.actor_agent.eval_rnn.load_state_dict(
                    {k.replace('module.', ''): v for k, v in self.share_model.state_dict().items()})

            if len(self.experience_batch)==self.args.batch_size//self.n_env+1:
                with self.lock:
                    self.buffer.store_episode(self.experience_batch)
                    print('Rank {} Buffer_size {}'.format(self.rank,self.buffer.current_size))
                    self.experience_batch=[]

            n_rollout+=1

        self.env.close()


class Plt_thread(Thread):
    def __init__(self,args,date_queue,rank):
        super(Plt_thread, self).__init__(daemon=True)
        self.args=args
        self.episode_rewards=[]
        self.win_rates=[]
        self.data_queue=date_queue
        self.rank=rank
        self.save_path=self.args.result_dir + '/' + 'mp_qmix' + '/' + args.map
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.start_time = time.time()

    def run(self,num=1):
        rcount = 0
        tmp_reward = 0
        win_number = 0
        last_time = 0

        while True:
            now_time = time.time()
            long_time = int(now_time - self.start_time)

            # 这里exper是一个字典，里面有很多个key，每个key的shape都是（1，x，x，x）
            # x就是和buff后面三个维度一致
            try:
                episode_reward, win_tag = self.data_queue.get_nowait()
            except:
                continue

            if win_tag:
                win_number = win_number + 1
            tmp_reward += episode_reward
            rcount += 1

            # 一分钟测一次
            # 只用0卡
            if long_time % 60 == 0 and long_time != last_time and self.rank == 0:
                tmp_reward /= rcount
                self.episode_rewards.append(tmp_reward)
                self.plt(num)
                if win_number / rcount > 0.8:
                    print('fast start time is {}'.format(long_time))
                self.win_rates.append(win_number / rcount)
                tmp_reward, win_number, rcount = 0, 0, 0
                last_time = long_time


    def plt(self, num):
        plt.figure(figsize=(12, 8))
        plt.axis([0, self.args.n_epoch, 0, 100])
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('time')
        # plt.xlabel('epoch')
        plt.ylabel('episode_rewards')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.win_rates)), self.win_rates)
        plt.xlabel('time')
        plt.ylabel('win_rate')

        plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
        np.save(self.save_path + '/win_rates_{}'.format(num), self.win_rates)
        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)