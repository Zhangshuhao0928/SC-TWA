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

args = get_common_args()
args = get_mixer_args(args)

if args.map == '3m':
    args.n_actions = 9
    args.n_agents = 3
    args.state_shape = 48
    args.obs_shape = 30
    args.episode_limit = 60

if args.map == '5m_vs_6m':
    args.n_actions = 12
    args.n_agents = 5
    args.state_shape = 98
    args.obs_shape = 55
    args.episode_limit = 70

if args.map == '8m':
    args.n_actions = 14
    args.n_agents = 8
    args.state_shape = 168
    args.obs_shape = 80
    args.episode_limit = 120

if args.map == '2s3z':
    args.n_actions = 11
    args.n_agents = 5
    args.state_shape = 120
    args.obs_shape = 80
    args.episode_limit = 120

batch = {'o': np.empty([args.batch_size, args.episode_limit, args.n_agents, args.obs_shape]),
         'u': np.empty([args.batch_size, args.episode_limit, args.n_agents, 1]),
         's': np.empty([args.batch_size, args.episode_limit, args.state_shape]),
         'r': np.empty([args.batch_size, args.episode_limit, 1]),
         'o_next': np.empty([args.batch_size, args.episode_limit, args.n_agents, args.obs_shape]),
         's_next': np.empty([args.batch_size, args.episode_limit, args.state_shape]),
         'avail_u': np.empty([args.batch_size, args.episode_limit, args.n_agents, args.n_actions]),
         'avail_u_next': np.empty([args.batch_size, args.episode_limit, args.n_agents, args.n_actions]),
         'u_onehot': np.empty([args.batch_size, args.episode_limit, args.n_agents, args.n_actions]),
         'padded': np.empty([args.batch_size, args.episode_limit, 1]),
         'terminated': np.empty([args.batch_size, args.episode_limit, 1])
         }

# a=Actor(args,None,None,None,None)
# a.start()
fighter_model = Agents(args)
train_steps=0
share_model = fighter_model.policy.eval_rnn
share_model.share_memory()
mini_batch=np.load('test.npy',allow_pickle=True)
mini_batch=mini_batch.item()

while True:
    fighter_model.train(mini_batch, train_steps)
    train_steps += 1
    print('train %d times' % train_steps)
    # share_model.load_state_dict(fighter_model.policy.eval_rnn.state_dict())

#在不等待数据的情况下，单纯的训练我们的网络，gpu利用率是13%-18%
# while True:
#     pass
