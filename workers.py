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
        #用来传回给actor动作的
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

    def run(self, num=1):
        for env_idx in range(self.args.env_nums):
            # 每个actor都有一个环境, 但是用的是Worker的网络
            parent_conn, child_conn = Pipe()
            buffer_in, buffer_out = Pipe()
            actor = Actor(self.args, child_conn, env_idx, buffer_out,self.wk_id)
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
        while True:
            # 每个环境
            for env_id, parent_conn, buffer_in in zip(env_ids, self.parent_conns, self.buffer_ins):
                # 每个step
                for i in range(self.n_agents):
                    # 每个智能体观测到的
                    obs, last_action, agent_id, avail_action, epsilon, evaluate, done = parent_conn.recv()
                    if done:
                        now_time = time.time()
                        long_time = int(now_time - start_time)

                        #这里exper是一个字典，里面有很多个key，每个key的shape都是（1，x，x，x）
                        #x就是和buff后面三个维度一致
                        experience, episode_reward, win_tag = buffer_in.recv()
                        if win_tag:
                            win_number = win_number + 1
                        tmp_reward += episode_reward
                        rcount += 1
                        #这里又deepcopy了一次   dataqueue只有一个 所以queue里的数据是多个环境一起放的
                        #不仅仅是只有一个环境 ，甚至不仅仅是只有一个worker的环境
                        self.data_queue.put(deepcopy(experience))
                        #这里print说明又有一条episode数据被该worker生产出来并放进了队列中
                        # print('======Worker {} Queue size:{}'.format(self.wk_id, self.data_queue.qsize()))
                        # 等待，不让队列中的数据太多，太多了首先会占很多的存储空间，第二都是一些比较旧的数据
                        if self.data_queue.qsize() > 200:
                            while self.data_queue.qsize() >= 100:
                            # while True:
                                time.sleep(4)
                        # 一分钟测一次
                        # 只用0号worker测试 画图费时间 也可以每个都画
                        # if self.wk_id==0:
                        #     print("long time=",long_time)
                        if long_time % 60 == 0 and long_time != last_time and self.wk_id == 0:
                            # print('plt,', long_time)
                            # print(111111111111111111111111111111)
                            tmp_reward /= rcount
                            self.episode_rewards.append(tmp_reward)
                            self.plt(num)
                            if win_number/rcount > 0.8:
                                print('fast start time is {}'.format(long_time))
                            self.win_rates.append(win_number / rcount)
                            tmp_reward, win_number, rcount = 0, 0, 0
                            last_time = long_time

                        self.worker_agent.restart_hidden(env_id, 1)
                        break
                    else:
                        # 由main agent作决策
                        action = self.worker_agent.choose_action(obs, last_action, agent_id,
                                                                 avail_action, epsilon, env_id, evaluate=False)
                        # print(action)
                         parent_conn.send(action)
            # 定期更新
            if self.args.ddp:
                self.worker_agent.eval_rnn.load_state_dict({k.replace('module.',''):v for k,v in self.share_model.state_dict().items()})
            else:
                self.worker_agent.eval_rnn.load_state_dict(self.share_model.state_dict())
    #
    # def evaluate(self):
    #     win_number = 0
    #     episode_rewards = 0
    #     for epoch in range(self.args.evaluate_epoch):
    #         episode_reward, win_tag = self.rolloutWorker.generate_evaluate(epoch)
    #         episode_rewards += episode_reward
    #         if win_tag:
    #             win_number += 1
    #     return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

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









