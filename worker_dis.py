# --- coding : utf-8 ---
# @Author   : Erhaoo
# @Time     : 2022/7/23  19:53
# @File     : worker_dis.py
# @Software : PyCharm

import numpy as np
import os
from common.actor_dis import Actor_dis,Plt_thread
from policy import qmix_model
import matplotlib.pyplot as plt
import time
from copy import deepcopy
from agent.agent_dis import Agents_dis
from torch.multiprocessing import Queue


# os.environ["CUDA_VISIBLE_DEVICES"] = "2" # 卡1更新 卡0，1,2，3收集数据

# cuda0 = torch.device('cuda:0')
# cuda1 = torch.device('cuda:1')


class Worker_dis(object):
    def __init__(self, args, rank, buffer):
        self.args = args
        self.win_rates = []
        self.episode_rewards = []

        self.rank = rank
        self.buffer = buffer
        self.agent = Agents_dis(args, rank)
        self.share_model = self.agent.policy.eval_rnn
        self.share_model.share_memory()
        self.env_idx = 0
        self.data_queue = Queue()

        for i in range(self.args.env_nums):
            actor = Actor_dis(self.args, self.env_idx, self.data_queue, self.share_model, i, self.buffer,rank)
            actor.start()
            print('Rank {} Env {} start'.format(self.rank, i))



        self.save_path = self.args.result_dir + '/' + 'mp_qmix' + '/' + args.map
        self.plt_thread = Plt_thread(self.args, self.data_queue, self.rank)
        self.plt_thread.start()
        print("Plt Thread start!")

    # def data_maker(self, num=1):
    #     self.experience_batch = []
    #     rcount = 0
    #     tmp_reward = 0
    #     win_number = 0
    #     last_time = 0
    #
    #     n_rollout = 0
    #     while n_rollout < self.args.batch_size:
    #         now_time = time.time()
    #         long_time = int(now_time - self.start_time)
    #
    #         # 这里exper是一个字典，里面有很多个key，每个key的shape都是（1，x，x，x）
    #         # x就是和buff后面三个维度一致
    #         try:
    #             episode_reward, win_tag = self.data_queue.get_nowait()
    #         except:
    #             continue
    #
    #         if win_tag:
    #             win_number = win_number + 1
    #         tmp_reward += episode_reward
    #         rcount += 1
    #
    #         # 一分钟测一次
    #         # 只用0卡
    #         if long_time % 60 == 0 and long_time != last_time and self.rank==0:
    #             tmp_reward /= rcount
    #             self.episode_rewards.append(tmp_reward)
    #             self.plt(num)
    #             if win_number / rcount > 0.8:
    #                 print('fast start time is {}'.format(long_time))
    #             self.win_rates.append(win_number / rcount)
    #             tmp_reward, win_number, rcount = 0, 0, 0
    #             last_time = long_time
    #
    #         n_rollout += 1
