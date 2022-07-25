# --- coding : utf-8 ---
# @Author   : Erhaoo
# @Time     : 2022/7/23  19:53
# @File     : worker_dis.py
# @Software : PyCharm

import numpy as np
import os
from common.actor_dis import Actor_dis
from policy import qmix_model
import matplotlib.pyplot as plt
import time
from copy import deepcopy
from agent.agent_dis import Agents_dis
from common.replay_buffer import ReplayBuffer
# os.environ["CUDA_VISIBLE_DEVICES"] = "2" # 卡1更新 卡0，1,2，3收集数据

# cuda0 = torch.device('cuda:0')
# cuda1 = torch.device('cuda:1')

class Worker_dis(object):
    def __init__(self,args,rank):
        self.args = args
        self.win_rates = []
        self.episode_rewards = []

        self.agent=Agents_dis(args,rank)
        self.share_model = self.agent.policy.eval_rnn
        self.share_model.share_memory()
        self.actor=Actor_dis(args,0)

        self.n_agents = args.n_agents

        self.experience_batch=[]
        self.start_time=time.time()

        # 用来保存plt和pkl
        self.save_path = self.args.result_dir + '/' + 'mp_qmix' + '/' + args.map
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def data_maker(self, num=1):
        self.experience_batch=[]
        self.actor.actor_agent.create_hidden(1)
        rcount = 0
        tmp_reward = 0
        win_number = 0
        last_time = 0

        n_rollout=0
        while n_rollout < self.args.batch_size:
            now_time = time.time()
            long_time = int(now_time - self.start_time)

            #这里exper是一个字典，里面有很多个key，每个key的shape都是（1，x，x，x）
            #x就是和buff后面三个维度一致
            experience, episode_reward, win_tag = self.actor.generate_episode(n_rollout,False)
            if win_tag:
                win_number = win_number + 1
            tmp_reward += episode_reward
            rcount += 1

            self.experience_batch.append(experience)

            # 一分钟测一次
            # 只用0号worker测试 画图费时间 也可以每个都画
            if long_time % 60 == 0 and long_time != last_time and self.wk_id == 0:
                # print('plt,', long_time)
                tmp_reward /= rcount
                self.episode_rewards.append(tmp_reward)
                self.plt(num)
                if win_number/rcount > 0.8:
                    print('fast start time is {}'.format(long_time))
                self.win_rates.append(win_number / rcount)
                tmp_reward, win_number, rcount = 0, 0, 0
                last_time = long_time

            self.actor.actor_agent.restart_hidden(0, 1)

            # 定期更新
            if self.args.ddp:
                self.actor.actor_agent.eval_rnn.load_state_dict({k.replace('module.',''):v for k,v in self.share_model.state_dict().items()})
            else:
                self.actor.actor_agent.eval_rnn.load_state_dict(self.share_model.state_dict())
            n_rollout+=1

        return self.experience_batch


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
