# --- coding : utf-8 ---
# @Author   : Erhaoo
# @Time     : 2022/6/23  20:39
# @File     : dataloader.py
# @Software : PyCharm
import time
import torch
import numpy as np
from common.replay_buffer import ReplayBuffer
from torch.multiprocessing import Pipe, Process, Queue
from copy import deepcopy


class DataLoader(Process):
    def __init__(self, data_queues, args, dataload_queue):
        super(DataLoader, self).__init__()
        self.queues = data_queues
        self.args = args
        self.buffer = ReplayBuffer(self.args)
        self.experience_list = []
        self.experience_batch = []
        self.dataload_queue = dataload_queue
        self.blog = True
        self.sort = []
        self.temp = []
    # def Sampler(self):


    def run(self):
        time.sleep(15)
        while self.blog:
            try:
                # 这里写成顺序是有问题的，因为如果第一个为空，则第二个第三个不会执行会继续积累
                # 并且如果第一个有，第二个第三个只要有一个没有，则这轮循环会作废，则第一个一定
                # 会白白浪费一个，所以第一个队列的样本数目这种情况下一定会小于第二个和第三个
                # 所以考虑可以交替执行
                # experience = self.queues[0].get_nowait()
                # # print(11111111111111)
                # # experience = self.data_queue1.get_nowait()
                # experience2 = self.data_queue2.get_nowait()
                # experience3 = self.data_queue3.get_nowait()
                # self.experience_list = [experience, experience2, experience3]
                # self.experience_batch.extend(self.experience_list)
                # print(222222222222222222)
                self.temp = []
                self.sort = []
                for i in range(self.args.worker_nums):
                    if self.queues[i].qsize() >= self.args.control_queue_size:
                        self.temp.append((i, self.queues[i].qsize()))
                self.temp = sorted(self.temp, key=lambda x: x[1], reverse=True)
                for i in range(len(self.temp)):
                    self.sort.append(self.temp[i][0])
                for i in self.sort:
                    experience = self.queues[i].get_nowait()
                    # print('----------')
                    self.experience_batch.append(experience)
                    # print('-------------',len(self.experience_batch))
                # print(111111111111111111111111111)
            except:
                continue

            # print(len(self.experience_batch))
            if len(self.experience_batch) >= self.args.batch_size:
                self.buffer.store_episode(self.experience_batch)
                self.experience_batch = []
            if self.buffer.current_size >= self.args.batch_size:
                # 从buffer中采样一个batch的数据用于训练 随机选的
                # 这里的time sleep是用来防止采样过快，导致队列中mini_batch太多
                time.sleep(0.05)
                mini_batch = self.buffer.sample(self.args.batch_size)
                self.dataload_queue.put(mini_batch)
                # if self.dataload_queue.qsize()>=20:
                #     self.dataload_queue.clear()
            # print('------------------', self.dataload_queue.qsize())
            # print('*****buffer size:', self.buffer.current_size)
