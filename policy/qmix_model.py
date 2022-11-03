import torch
import os
from network.base_net import RNN
import numpy as np
import torch.nn as nn
import torch.nn.functional as f

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # 卡1更新 卡0，1,2，3收集数据

# cuda0 = torch.device('cuda:0')
# cuda1 = torch.device('cuda:1')

#
# class QmixModel(nn.Module):
#     def __init__(self, args):
#         super(QmixModel, self).__init__()
#         self.n_actions = args.n_actions
#         self.n_agents = args.n_agents
#         self.state_shape = args.state_shape
#         self.obs_shape = args.obs_shape
#         input_shape = self.obs_shape
#         # 根据参数决定RNN的输入维度
#         if args.last_action:
#             input_shape += self.n_actions
#         if args.reuse_network:
#             input_shape += self.n_agents
#
#         self.args = args
#         self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
#         self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
#         self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
#
#         # 执行过程中，要为每个agent都维护一个eval_hidden
#         self.eval_hidden = None
#         self.eval_hiddens = []
#         print('Init QmixModel for Worker')
#
#     def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, env_id, evaluate=False):
#         inputs = obs.copy()
#         avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose
#
#         # transform agent_num to onehot vector
#         agent_id = np.zeros(self.n_agents)
#         agent_id[agent_num] = 1.
#
#         if self.args.last_action:
#             inputs = np.hstack((inputs, last_action))
#         if self.args.reuse_network:
#             inputs = np.hstack((inputs, agent_id))
#
#         hidden_state = self.eval_hiddens[env_id][:, agent_num, :]
#         # transform the shape of inputs from (42,) to (1,42)
#         inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
#         avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
#         if self.args.cuda:
#             inputs = inputs.cuda()
#             hidden_state = hidden_state.cuda()
#
#         # get q value
#         # forward
#         x = f.relu(self.fc1(inputs))
#         h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
#         h = self.rnn(x, h_in)
#         q = self.fc2(h)
#         # 这样改对吗？
#         q_value, self.eval_hiddens[env_id][:, agent_num, :] = q, h
#
#         # choose action from q value
#         q_value[avail_actions == 0.0] = - float("inf")
#         if np.random.uniform() < epsilon:
#             action = np.random.choice(avail_actions_ind)  # action是一个整数
#         else:
#             action = torch.argmax(q_value)
#         return action
#
#     # 多环境配多个hidden，用于交互，用不到target
#     def create_hidden(self, episode_num):
#         for _ in range(self.args.env_nums):
#             self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
#             self.eval_hiddens.append(self.eval_hidden)
#
#     # 每个环境异步，因此需要独立初始化
#     def restart_hidden(self, env_id, episode_num):
#         self.eval_hiddens[env_id] = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
# #


class QmixModel:
    def __init__(self, args):
        super(QmixModel, self).__init__()
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape
        # 根据参数决定RNN的输入维度
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        # 神经网络
        self.eval_rnn = RNN(input_shape, args)  # 每个agent选动作的网络
        self.args = args
        if self.args.ger_in_gpu:
            self.eval_rnn.cuda()
        # if self.args.cuda:
        #     # torch.cuda.set_device(wk_id)
        #     self.eval_rnn.cuda()
        #     if self.args.dp:
        #         self.eval_rnn=nn.DataParallel(self.eval_rnn)

        # 执行过程中，要为每个agent都维护一个eval_hidden
        self.eval_hidden = None
        # 用于决策 交互
        self.eval_hiddens = []
        print('Init QmixModel for Worker')

        self.eval_hidden_for_evaluate=None

    # 多环境配多个hidden，用于交互，用不到target
    def create_hidden(self, episode_num):
        for _ in range(self.args.env_nums):
            self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
            self.eval_hiddens.append(self.eval_hidden)

    # 每个环境异步，因此需要独立初始化
    def restart_hidden(self, env_id, episode_num):
        self.eval_hiddens[env_id] = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, env_id, maven_z=None, evaluate=False):
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        #这里的hidden_state和learner中的不太一样，因为是每个agent做决策，并且episode_num一直是1，所以如下
        hidden_state = self.eval_hiddens[env_id][:, agent_num, :]
        # transform the shape of inputs from (42,) to (1,42)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        # if self.args.cuda:
        #     inputs = inputs.cuda()
        #     hidden_state = hidden_state.cuda()

        # get q value
        if self.args.ger_in_gpu:
            inputs=inputs.cuda()
            hidden_state=hidden_state.cuda()
            q_value, self.eval_hiddens[env_id][:, agent_num, :] = self.eval_rnn(inputs, hidden_state)
        else:
            q_value, self.eval_hiddens[env_id][:, agent_num, :] = self.eval_rnn(inputs, hidden_state)

        # choose action from q value
        q_value[avail_actions == 0.0] = - float("inf")
        if np.random.uniform() < epsilon:
            action = np.random.choice(avail_actions_ind)  # action是一个整数
        else:
            action = torch.argmax(q_value)
            if self.args.ger_in_gpu:
                action = action.cpu().numpy()
        return action

    def create_hidden_for_evaluate(self, episode_num):
        self.eval_hidden_for_evaluate = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))


    def choose_action_for_evaluate(self, obs, last_action, agent_num, avail_actions, epsilon, maven_z=None, evaluate=False):
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        hidden_state = self.eval_hidden_for_evaluate[:, agent_num, :]

        # transform the shape of inputs from (42,) to (1,42)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        # if self.args.cuda:
        #     inputs = inputs.cuda()
        #     hidden_state = hidden_state.cuda()

        # get q value
        if self.args.alg == 'maven':
            maven_z = torch.tensor(maven_z, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                maven_z = maven_z.cuda()
            q_value, self.eval_hidden_for_evaluate[:, agent_num, :] = self.eval_rnn(inputs, hidden_state, maven_z)
        else:
            q_value, self.eval_hidden_for_evaluate[:, agent_num, :] = self.eval_rnn(inputs, hidden_state)

        # choose action from q value
        if self.args.alg == 'coma' or self.args.alg == 'central_v' or self.args.alg == 'reinforce':
            action = self._choose_action_from_softmax(q_value.cpu(), avail_actions, epsilon, evaluate)
        else:
            q_value[avail_actions == 0.0] = - float("inf")
            if np.random.uniform() < epsilon:
                action = np.random.choice(avail_actions_ind)  # action是一个整数
            else:
                action = torch.argmax(q_value)
        return action