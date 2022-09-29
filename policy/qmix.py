import torch
import os
from network.base_net import RNN
from network.qmix_net import QMixNet
import torch.nn as nn
import torch.distributed as dist

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # 卡1更新 卡0，1,2，3收集数据
#
# cuda0 = torch.device('cuda:0')
# cuda1 = torch.device('cuda:1')

# 训练专用
class QMIX:
    def __init__(self, args,rank):
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

        # 神经网络 这就是所谓的modoel
        self.eval_rnn = RNN(input_shape, args) # 每个agent选动作的网络
        self.target_rnn = RNN(input_shape, args)
        self.eval_qmix_net = QMixNet(args)  # 把agentsQ值加起来的网络
        self.target_qmix_net = QMixNet(args)

        self.args = args
        self.rank=rank
        #将模型加载到gpu中

        if self.args.cuda:
            # self.eval_rnn.cuda()
            # self.target_rnn.cuda()
            # self.eval_qmix_net.cuda()
            # self.target_qmix_net.cuda()
            self.eval_rnn.to(rank)
            self.target_rnn.to(rank)
            self.eval_qmix_net.to(rank)
            self.target_qmix_net.to(rank)
            # if self.args.dp:
            #     self.eval_rnn=nn.DataParallel(self.eval_rnn)
            #     self.target_rnn = nn.DataParallel(self.target_rnn)
            #     self.eval_qmix_net = nn.DataParallel(self.eval_qmix_net)
            #     self.target_qmix_net = nn.DataParallel(self.target_qmix_net)
            if self.args.ddp:
                dist.init_process_group("nccl")
                self.eval_rnn=nn.parallel.DistributedDataParallel(self.eval_rnn,device_ids=[rank])
                self.target_rnn = nn.parallel.DistributedDataParallel(self.target_rnn,device_ids=[rank])
                self.eval_qmix_net = nn.parallel.DistributedDataParallel(self.eval_qmix_net,device_ids=[rank])
                self.target_qmix_net = nn.parallel.DistributedDataParallel(self.target_qmix_net,device_ids=[rank])

        self.model_dir = args.model_dir + '/' + 'mp_qmix' + '/' + args.map
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/3_rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/3_rnn_net_params.pkl'
                path_qmix = self.model_dir + '/3_qmix_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_locaap_location))
                self.eval_qmix_net.load_state_dict(torch.load(path_qmix, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_rnn, path_qmix))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_rnn.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        # 用于决策 交互
        # self.eval_hiddens = []
        print('Init alg QMIX')

    def  learn(self, batch, max_episode_len, train_step, epsilon=None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        '''
        在learn的时候，抽取到的数据是四维的，四个维度分别为 1——第几个episode 2——episode中第几个transition
        3——第几个agent的数据 4——具体obs维度。因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state，
        hidden_state和之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，然后一次给神经网络
        传入每个episode的同一个位置的transition
        '''
        episode_num = batch['o'].shape[0]
        #这里初始化hidden的episode_num就是一个batch的了，跟worker中不一样，不再是一直是1了
        self.init_hidden(episode_num)
        for key in batch.keys():  # 把batch里的数据转化成tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], \
                                                             batch['r'],  batch['avail_u'], batch['avail_u_next'],\
                                                             batch['terminated']
        # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
        #有用的time_step的mask处的值会是1，填充的time_step的mask值会是0
        mask = 1 - batch["padded"].float()

        # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len， n_agents， n_actions)
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        if self.args.cuda:
            s = s.cuda()
            u = u.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
        # q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
        #按照联合动作u中的每个智能体的动作分量去提取出相应的每个智能体的q值
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3).cuda()

        # 得到target_q
        q_targets[avail_u_next == 0.0] = - 9999999
        # [0]是返回值，[1]是返回相应的索引
        #将每个智能体都选择q值最大的动作，在qmix里也就满足了单调性原则，所以其组成的联合动作的qtot也最大
        q_targets = q_targets.max(dim=3)[0]
        # 返回的维度是（episode_num,max_episode_len,1)
        #获得qtot   q_evals输入在公式中既是联合动作u的输入
        q_total_eval = self.eval_qmix_net(q_evals, s)
        #获得max q‘tot  q_targets输入在公式中就是max联合动作u’的输入
        q_total_target = self.target_qmix_net(q_targets, s_next)
        #这里去掉了已经结束的轨迹的后续因为填充所导致的q_total_target
        targets = r + self.args.gamma * q_total_target * (1 - terminated)

        td_error = (q_total_eval - targets.detach())
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error

        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # 初始化梯度为0 梯度的形状和权重参数是一样的
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()
        # 更新target网络
        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

    def _get_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx（time_step）的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)

        # 给obs添加上一个动作、agent编号
        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一个时间步的经验，就让前一个动作为0向量
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            # 因为当前的obs三维的数据，每一维分别代表(episode编号，agent编号，obs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # 要把obs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        #每个时间步分别处理运算
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
            #这里直接将整个hidden传入了，因为是一个batch并且多个agent的数据作为input
            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

            # 把q_eval维度重新变回(8, 5,n_actions)
            #这里的q是每个episode的某一时间步下的每个智能体的每个动作的评分
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个列表，维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        #返回的都是tensor
        return q_evals, q_targets

    # 只给训练用
    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    # # 多环境配多个hidden，用于交互，用不到target
    # def create_hidden(self, episode_num):
    #     for _ in range(self.args.env_nums):
    #         self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
    #         self.eval_hiddens.append(self.eval_hidden)
    #
    # # 每个环境异步，因此需要独立初始化
    # def restart_hidden(self, env_id, episode_num):
    #     self.eval_hiddens[env_id] = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_qmix_net.state_dict(), self.model_dir + '/' + num + '_qmix_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')
