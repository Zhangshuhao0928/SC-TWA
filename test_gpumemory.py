# --- coding : utf-8 ---
# @Author   : Erhaoo
# @Time     : 2022/6/30  14:45
# @File     : test_gpumemory.py
# @Software : PyCharm

from workers import Worker
from agent.agent import Agents
from torch.multiprocessing import Queue
from common.arguments import get_common_args, get_mixer_args
# from tensorboardX import SummaryWriter
from common.replay_buffer import ReplayBuffer
import torch
import os
import time
from common.arguments import get_common_args, get_mixer_args
import numpy as np

# a = torch.tensor((1.)).cuda()
# b=torch.tensor((2)).cuda()

# args = get_common_args()
# args = get_mixer_args(args)
#
# if args.map == '3m':
#     args.n_actions = 9
#     args.n_agents = 3
#     args.state_shape = 48
#     args.obs_shape = 30
#     args.episode_limit = 60
#
# if args.map == '5m_vs_6m':
#     args.n_actions = 12
#     args.n_agents = 5
#     args.state_shape = 98
#     args.obs_shape = 55
#     args.episode_limit = 70
#
# if args.map == '8m':
#     args.n_actions = 14
#     args.n_agents = 8
#     args.state_shape = 168
#     args.obs_shape = 80
#     args.episode_limit = 120
#
# if args.map == '2s3z':
#     args.n_actions = 11
#     args.n_agents = 5
#     args.state_shape = 120
#     args.obs_shape = 80
#     args.episode_limit = 120
#
# batch = {'o': np.empty([args.batch_size, args.episode_limit, args.n_agents, args.obs_shape]),
#          'u': np.empty([args.batch_size, args.episode_limit, args.n_agents, 1]),
#          's': np.empty([args.batch_size, args.episode_limit, args.state_shape]),
#          'r': np.empty([args.batch_size, args.episode_limit, 1]),
#          'o_next': np.empty([args.batch_size, args.episode_limit, args.n_agents, args.obs_shape]),
#          's_next': np.empty([args.batch_size, args.episode_limit, args.state_shape]),
#          'avail_u': np.empty([args.batch_size, args.episode_limit, args.n_agents, args.n_actions]),
#          'avail_u_next': np.empty([args.batch_size, args.episode_limit, args.n_agents, args.n_actions]),
#          'u_onehot': np.empty([args.batch_size, args.episode_limit, args.n_agents, args.n_actions]),
#          'padded': np.empty([args.batch_size, args.episode_limit, 1]),
#          'terminated': np.empty([args.batch_size, args.episode_limit, 1])
#          }
#
# # a=Actor(args,None,None,None,None)
# # a.start()
# fighter_model = Agents(args)
# train_steps=0
# share_model = fighter_model.policy.eval_rnn
# share_model.share_memory()
# mini_batch=np.load('test.npy',allow_pickle=True)
# mini_batch=mini_batch.item()
#
# while True:
#     fighter_model.train(mini_batch, train_steps)
#     train_steps += 1
#     print('train %d times' % train_steps)
#     # share_model.load_state_dict(fighter_model.policy.eval_rnn.state_dict())

# 在不等待数据的情况下，单纯的训练我们的网络，gpu利用率是13%-18%
# while True:
#     pass

from concurrent.futures import ThreadPoolExecutor
import threading


# def aaa(i,b):
#     print("i=",i,"b=",b,"threading=",threading.current_thread().name)
#     time.sleep(0.1)
#
# t=ThreadPoolExecutor(3)
# # i_s=[1,2,3,4,5,6,7,8,9]
# # b_s=[5,6,7,8,9,10,11,12,13]
# # task=[t.submit(aaa,i,b) for i,b in zip(i_s,b_s)]
# from torch.multiprocessing import Queue
# queue=Queue()
# def cin(queue,i,j):
#     i=0
#     while i<1000000:
#         queue.put([j,i])
#         print("size:",queue.qsize())
#         i+=1
#
# task=[t.submit(cin,queue,i,j) for i,j in zip(range(3),range(3))]
# while True:
#     try:
#         data=queue.get_nowait()
#         print("cout:",data)
#         # time.sleep(0.1)
#     except:
#         continue

# class test111(threading.Thread):
#     def __init__(self):
#         super(test111, self).__init__(daemon=True)
#         self.eval_hidden = None
#         self.eval_hiddens = []
#         self.create_hidden(1)
#
#     def create_hidden(self, episode_num):
#         for _ in range(2):
#             self.eval_hidden = torch.zeros((1, 4))
#             print(id(self.eval_hidden))
#             self.eval_hiddens.append(self.eval_hidden)
#
#     def run(self):
#         self.eval_hiddens[0] = torch.randn((1, 4))
#         print(self.eval_hiddens)
#         print(id(self.eval_hiddens))
#
#
#
# for i in range(2):
#     actor = test111()
#     actor.start()
#     actor.join()


