import numpy as np
import os
from common.actors import Actor
from policy import qmix_model
import matplotlib.pyplot as plt
from torch.multiprocessing import Pipe, Process
import torch.distributed as dist
import time
from copy import deepcopy
from threading import Thread
from smac.env import StarCraft2Env


# os.environ["CUDA_VISIBLE_DEVICES"] = "2" # 卡1更新 卡0，1,2，3收集数据

# cuda0 = torch.device('cuda:0')
# cuda1 = torch.device('cuda:1')

class Worker(Process):
    def __init__(self, args, share_model, data_queue, wk_id):
        super(Worker, self).__init__()
        # self.buffer = buffer # buffer不行 太大了
        self.args = args
        self.win_rates = []
        self.episode_rewards = []
        self.wk_id = wk_id

        self.envs = []
        self.parent_conns = []
        self.child_conns = []
        self.buffer_ins = []
        self.buffer_outs = []

        self.share_model = share_model
        self.data_queue = data_queue
        # 用来传回给actor动作的
        self.worker_agent = qmix_model.QmixModel(args)
        # self.train_steps = 0
        self.n_agents = args.n_agents
        # self.memory = [None for _ in range(self.args.env_nums)] # 同步？
        #
        # 定期更新每个worker 只负责交互 不需要更新 一个worker搭配多个actor
        # self.worker_agent.policy.eval_rnn.load_state_dict(self.share_model.state_dict())
        # 用来保存plt和pkl
        self.save_path = self.args.result_dir + '/' + 'mp_qmix' + '/' + args.map
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.threshold = 100
        self.upper_bound = 7.5
        self.lower_bound = 0
        self.increment = 0.1
        self.exp_count = 0

        self.env = StarCraft2Env(map_name=self.args.map,
                                 step_mul=self.args.step_mul,
                                 difficulty=self.args.difficulty,
                                 game_version=self.args.game_version,
                                 replay_dir=self.args.replay_dir)

    def run(self, num=1):
        for env_idx in range(self.args.env_nums):
            # 每个actor都有一个环境, 但是用的是Worker的网络
            parent_conn, child_conn = Pipe()
            buffer_in, buffer_out = Pipe()
            actor = Actor(self.args, child_conn, env_idx, buffer_out, self.wk_id)
            actor.start()
            print('Worker {} Env {} start'.format(self.wk_id, env_idx))
            self.envs.append(actor)
            self.parent_conns.append(parent_conn)
            self.child_conns.append(child_conn)
            self.buffer_ins.append(buffer_in)
            self.buffer_outs.append(buffer_out)
        print('Worker {}\'s {} envs start successfully'.format(self.wk_id, self.args.env_nums))

        # 传数据 进行交互
        env_ids = [i for i in range(self.args.env_nums)]
        self.worker_agent.create_hidden(1)
        rcount = 0
        tmp_reward = 0
        win_number = 0
        start_time = time.time()
        last_time = 0

        time_steps = 0
        evaluate_steps = -1

        while True:
            # 这里每个worker需要按顺序执行与每个actor的交互，也就是说worker与actor的交互并不是完全并行的
            # 每个agent执行动作也是按顺序执行的，并不是完全并行的
            # 每个worker需要与它有的全部actor都交互完一次(一个step)才可以更新一次网络参数，其频率与训练频率不一致
            # 每个环境
            for env_id, parent_conn, buffer_in in zip(env_ids, self.parent_conns, self.buffer_ins):
                # 每个step
                for i in range(self.n_agents):
                    # 每个智能体观测到的
                    obs, last_action, agent_id, avail_action, epsilon, evaluate, done = parent_conn.recv()
                    if done:
                        now_time = time.time()
                        # 这里存在一个问题，有可能例如122s的时候走到这里，但是余数会是2 所以第二分钟就没画图
                        long_time = int(now_time - start_time)

                        # 这里exper是一个字典，里面有很多个key，每个key的shape都是（1，x，x，x）
                        # x就是和buff后面三个维度一致
                        experience, episode_reward, win_tag, steps = buffer_in.recv()
                        time_steps += steps
                        # self.exp_count += 1
                        # 这里又deepcopy了一次   dataqueue只有一个 所以queue里的数据是多个环境一起放的
                        # 不仅仅是只有一个环境 ，甚至不仅仅是只有一个worker的环境
                        self.data_queue.put(deepcopy(experience))
                        # if self.exp_count % self.threshold == 0 and self.lower_bound <= self.upper_bound:
                        #     self.lower_bound += self.increment
                        # if episode_reward >= self.lower_bound:
                        #     self.data_queue.put(deepcopy(experience))
                        # else:
                        #     self.worker_agent.restart_hidden(env_id, 1)
                        #     break
                        if win_tag:
                            win_number = win_number + 1
                        tmp_reward += episode_reward
                        rcount += 1
                        # 这里print说明又有一条episode数据被该worker生产出来并放进了队列中
                        # print('======Worker {} Queue size:{}'.format(self.wk_id, self.data_queue.qsize()))
                        # 等待，不让队列中的数据太多，太多了首先会占很多的存储空间，第二都是一些比较旧的数据
                        if self.data_queue.qsize() > 200:
                            while self.data_queue.qsize() >= 100:
                                time.sleep(2)
                        # 一分钟测一次
                        # 只用0号worker测试 画图费时间 也可以每个都画
                        # if self.wk_id==0:
                        #     print("long time=",long_time)
                        # if long_time % 60 == 0 and long_time != last_time and self.wk_id == 0:
                        #     # print('plt,', long_time)
                        #     tmp_reward /= rcount
                        #     self.episode_rewards.append(tmp_reward)
                        #     self.plt_bytime(num)
                        #     # if win_number / rcount > 0.8:
                        #     #     print('fast start time is {}'.format(long_time))
                        #     self.win_rates.append(win_number / rcount)
                        #     tmp_reward, win_number, rcount = 0, 0, 0
                        #     last_time = long_time

                        if time_steps // self.args.evaluate_cycle > evaluate_steps and self.wk_id == 0:
                            win_rate, episode_reward = self.evaluate()
                            # print('win_rate is ', win_rate)
                            self.win_rates.append(win_rate)
                            self.episode_rewards.append(episode_reward)
                            self.plt(num)
                            evaluate_steps += 1

                        # self.worker_agent.restart_hidden(env_id, 1)
                        break
                    else:
                        # 由main agent作决策
                        action = self.worker_agent.choose_action(obs, last_action, agent_id,
                                                                 avail_action, epsilon, env_id, evaluate=False)
                        # print(action)
                        parent_conn.send(action)
            # 定期更新
            if self.args.ddp:
                self.worker_agent.eval_rnn.load_state_dict(
                    {k.replace('module.', ''): v for k, v in self.share_model.state_dict().items()})
            else:
                self.worker_agent.eval_rnn.load_state_dict(self.share_model.state_dict())


    def plt_bytime(self, num):
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

        plt.savefig(self.save_path + '/plt_{}_{}_{}.png'.format(num, self.args.per, self.args.optimizer), format='png')
        np.save(self.save_path + '/win_rates_{}_{}_{}'.format(num, self.args.per, self.args.optimizer), self.win_rates)
        np.save(self.save_path + '/episode_rewards_{}_{}_{}'.format(num, self.args.per, self.args.optimizer),
                self.episode_rewards)

    def plt(self, num):
        plt.figure(figsize=(12, 8))
        plt.axis([0, self.args.n_epoch, 0, 100])
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        # plt.xlabel('epoch')
        plt.ylabel('episode_rewards')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.win_rates)), self.win_rates)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('win_rate')

        plt.savefig(self.save_path + '/plt_{}_{}_{}.png'.format(num, self.args.per, self.args.optimizer), format='png')
        np.save(self.save_path + '/win_rates_{}_{}_{}'.format(num, self.args.per, self.args.optimizer), self.win_rates)
        np.save(self.save_path + '/episode_rewards_{}_{}_{}'.format(num, self.args.per, self.args.optimizer),
                self.episode_rewards)

    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag, _ = self.generate_episode_for_evaluate(epoch)
            episode_rewards += episode_reward
            print('test episode reward is:',episode_reward)
            if win_tag:
                win_number += 1
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

    def generate_episode_for_evaluate(self, episode_num=None, evaluate=True):
        # if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
        #     self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        # 重置环境
        self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.worker_agent.create_hidden_for_evaluate(1)

        # epsilon
        epsilon = 0 if evaluate else self.args.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.args.anneal_epsilon if epsilon > self.args.min_epsilon else epsilon

        while not terminated and step < self.args.episode_limit:
            # time.sleep(0.2)
            # obs是全部agents的观测，不是单独一个agent  (n_agents,obs_shape)
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []
            # 每个agent选择一个动作，所有agent都选择一个动作之后，也就是这个time_step结束
            # 才会开始下一个time_step
            for agent_id in range(self.args.n_agents):
                # 一维，（n_actions，）
                avail_action = self.env.get_avail_agent_actions(agent_id)

                action = self.worker_agent.choose_action_for_evaluate(obs[agent_id], last_action[agent_id], agent_id,
                                                                      avail_action, epsilon, evaluate)
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                # 是个one-hot，size为（n_agent,n_action)
                last_action[agent_id] = action_onehot

            reward, terminated, info = self.env.step(actions)
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            o.append(obs)
            s.append(state)
            # 联合动作 里面每一个shape都是(n_agent,1)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            # 里面每一个都是(n_agent,n_action)
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)

            r.append([reward])
            # print(r)
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.args.anneal_epsilon if epsilon > self.args.min_epsilon else epsilon
        episode_len = len(terminate)
        # print(
        #     "worker id={},actor id={},episode len={},reward={},win_tag={}".format(self.wk_id, self.env_idx, episode_len,
        #                                                                           episode_reward, win_tag))
        # last obs  这里是为target网络的输入做准备的
        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        # 这里是把上面加进来的last的都删掉，因为只有_next才需要
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
        for i in range(step, self.args.episode_limit):
            o.append(np.zeros((self.args.n_agents, self.args.obs_shape)))
            u.append(np.zeros([self.args.n_agents, 1]))
            s.append(np.zeros(self.args.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.args.n_agents, self.args.obs_shape)))
            s_next.append(np.zeros(self.args.state_shape))
            u_onehot.append(np.zeros((self.args.n_agents, self.args.n_actions)))
            avail_u.append(np.zeros((self.args.n_agents, self.args.n_actions)))
            avail_u_next.append(np.zeros((self.args.n_agents, self.args.n_actions)))
            # 1说明是填充的，0是正常数据
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
                       terminated=terminate.copy(),
                       episode_reward=episode_reward,
                       episode_len=episode_len
                       )
        # print('padding', len(episode['padded'])) 60
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.args.epsilon = epsilon
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()

        return episode, episode_reward, win_tag, step
