import numpy as np
import threading


# from common.SumTree import Tree

class Tree(object):
    # 指数据的指针，所以其本身的值加上n-1才是真实的idx
    data_pointer = 0
    eps = 0.9

    def __init__(self, capacity):
        self.capacity = capacity  # leaf node numbers
        self.tree = np.zeros(2 * capacity - 1)  # all nodes
        # dtype=object的含义是 数组中每个元素可以是不同类型的元素，比如[0]是array.[1]是list
        self.data = np.zeros(capacity, dtype=object)  # data[id] = data
        self.current_size = 0
        print('tree size', capacity)

    # def add(self, p, episode_batch):
    #     idx = self.data_pointer + self.capacity - 1  # 找到第一个空的叶子结点（没有写入数据）
    #     episode_num = len(episode_batch)
    #     for i in range(episode_num):
    #         self.data[self.data_pointer] = episode_batch[i]  # 存数据
    #         self.update_tree(idx, p)
    #         if self.data_pointer < self.capacity - 1:
    #             self.data_pointer += 1
    #         else:
    #             self.data_pointer = 0
    #         if self.current_size < self.capacity:
    #             self.current_size += 1
    #     # 这里可以保证重复填充buffer，不过目前是最简单的从最小的序号覆盖
    #     # if self.data_pointer > self.capacity - 1:
    #     #     self.data_pointer = 0

    def add(self, p, episode, train_step):
        idx = self.data_pointer + self.capacity - 1  # 找到第一个空的叶子结点（没有写入数据）

        episode['n_1'] = train_step
        episode['n_2'] = 0
        episode['ee_if'] = 1

        self.data[self.data_pointer] = episode  # 存数据
        # self.update_tree(idx, p, episode['ee_if'])
        self.update_tree(idx, p, 0)
        # self.data_pointer += 1
        # if self.current_size < self.capacity:
        #     self.current_size += 1
        # # 这里可以保证重复填充buffer，不过目前是最简单的从最小的序号覆盖
        # if self.data_pointer > self.capacity - 1:
        #     # self.data_pointer = 0
        #     self.data_pointer=np.argmin(self.tree[-self.capacity:])
        if self.current_size < self.capacity:
            self.data_pointer += 1
            self.current_size += 1
        if self.current_size == self.capacity:
            self.data_pointer = np.argmin(self.tree[-self.capacity:])

    # 只改节点的优先级
    def update_tree(self, idx, p, ee_if):
        change = (1-self.eps)*ee_if + self.eps*p - self.tree[idx]
        # # print('change', change)
        self.tree[idx] = (1-self.eps)*ee_if + self.eps*p
        # change = p - self.tree[idx]
        # self.tree[idx] = p
        # 便利祖先节点，增加祖先节点中的优先级之和的值
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change
        # print(self.tree)
        # self.propagate(idx, change)

    # def propagate(self, idx, change):
    #     parent = （idx - 1) // 2
    #     self.tree[parent] += change #向上传播
    #     if parent != 0:
    #         self.propagate(parent, change)

    # 返回优先级之和
    def total(self):
        return self.tree[0]

    def get(self, s):
        parent_idx = 0
        while True:
            left = 2 * parent_idx + 1
            right = left + 1
            if left >= len(self.tree):
                idx = parent_idx
                break
            else:
                if s <= self.tree[left]:
                    parent_idx = left
                else:
                    s -= self.tree[left]
                    parent_idx = right

        data_idx = idx - self.capacity + 1
        return idx, data_idx, self.tree[idx], self.data[data_idx]


class PEReplay(object):
    t_epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.3
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    c = -0.0001

    def __init__(self, args):
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape
        self.size = self.args.buffer_size
        self.episode_limit = self.args.episode_limit

        # memory management
        # self.current_idx = 0
        # self.current_size = 0
        # create the buffer to store info
        # a = np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape])

        # sumtree
        # if args.per:
        self.t = Tree(self.size)
        # else:
        #     self.buffers = {'o': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
        #                 'u': np.empty([self.size, self.episode_limit, self.n_agents, 1]),
        #                 's': np.empty([self.size, self.episode_limit, self.state_shape]),
        #                 'r': np.empty([self.size, self.episode_limit, 1]),
        #                 'o_next': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
        #                 's_next': np.empty([self.size, self.episode_limit, self.state_shape]),
        #                 'avail_u': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
        #                 'avail_u_next': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
        #                 'u_onehot': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
        #                 'padded': np.empty([self.size, self.episode_limit, 1]),
        #                 'terminated': np.empty([self.size, self.episode_limit, 1])
        #                 }
        # thread lock
        self.lock = threading.Lock()

    def store_sumtree(self, episode_batch, train_step):
        # 获取所有数据中优先级最高的数据的优先级
        max_p = np.max(self.t.tree[-self.t.capacity:])
        # print(max_p)
        if max_p == 0:
            max_p = self.abs_err_upper
        # with self.lock:
        # 给新加入的样本最大的优先级（确保能被训练到）
        self.t.add(max_p, episode_batch, train_step)
        # print(self.t.tree)

    def sample_sumtree(self, batch_size, train_step):
        # print('bs', batch_size)
        # exit()
        # 一维数组
        b_idx = np.empty((batch_size,), dtype=np.int32)
        b_data_idx = np.empty((batch_size,), dtype=np.int64)
        # print(self.tree.data[0]['o'].shape)
        # print(self.tree.data[0]['o'].size)
        b_memory = {'o': np.empty([batch_size, self.episode_limit, self.n_agents, self.obs_shape]),
                    'u': np.empty([batch_size, self.episode_limit, self.n_agents, 1]),
                    's': np.empty([batch_size, self.episode_limit, self.state_shape]),
                    'r': np.empty([batch_size, self.episode_limit, 1]),
                    'o_next': np.empty([batch_size, self.episode_limit, self.n_agents, self.obs_shape]),
                    's_next': np.empty([batch_size, self.episode_limit, self.state_shape]),
                    'avail_u': np.empty([batch_size, self.episode_limit, self.n_agents, self.n_actions]),
                    'avail_u_next': np.empty([batch_size, self.episode_limit, self.n_agents, self.n_actions]),
                    'u_onehot': np.empty([batch_size, self.episode_limit, self.n_agents, self.n_actions]),
                    'padded': np.empty([batch_size, self.episode_limit, 1]),
                    'terminated': np.empty([batch_size, self.episode_limit, 1]),
                    'episode_reward': np.empty(batch_size),
                    'episode_len': np.empty(batch_size),
                    # generate train step
                    'n_1': np.empty(batch_size),
                    # using times
                    'n_2': np.empty(batch_size),
                    'ee_if': np.empty(batch_size)
                    }

        # print(type(b_memory), b_memory.shape)
        ISWeights = np.empty([batch_size, self.episode_limit, 1])
        pri_seg = self.t.total() / batch_size  # priority segment
        print("pri_seg:", pri_seg)
        # print("total={},pri_seg={}".format(self.t.total(),pri_seg))
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.t.tree[-self.t.capacity:]) / self.t.total()
        # print('min_prob', min_prob)
        if min_prob == 0:
            min_prob = 0.00001

        # 报nan的错误原因是prob=0，所以最后prob/min_prob=0，而指数是负数，所以会出现除以0的错误
        # 所以最终该序号的isweights是inf,最后用这个数据算loss的时候，loss值为nan，之后abs error全是nan
        # 所以之后self.t.total是nan，pri_seg是nan，a，b是nan，所以np.random.uniform会报错
        # 并且p不能是负值
        for i in range(batch_size):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, data_idx, p, data = self.t.get(v)
            # print(type(data))
            if p == 0.:
                continue
            prob = p / self.t.total()
            # print("prob={},min_prob={}".format(prob,min_prob))
            # 这里每个输出是一个大列表，列表的每一项是一个小列表，每个小列表中只有一个值，值完全相同
            # print("prob/min_prob={}".format(np.power(prob / min_prob, -self.beta)))
            # prob/min_prob >= 1 prob/min_prob越小 isweights特大 beta越大，isweight越小
            ISWeights[i] = [[np.power(prob / min_prob, -self.beta)]] * self.episode_limit

            b_idx[i] = idx
            b_data_idx[i] = data_idx
            for key in data.keys():
                b_memory[key][i] = data[key]
            self.update_ee_if(int(data_idx), train_step)

        # print('b_memory', b_memory['o'].shape)
        return b_idx, b_data_idx, b_memory, ISWeights

    def update_ee_if(self, idx, train_step):
        self.t.data[idx]['n_2'] += 1
        delta = train_step - self.t.data[idx]['n_1']
        self.t.data[idx]['ee_if'] = np.max((self.t.data[idx]['episode_reward'] / self.t.data[idx]['episode_len'] + \
                                    self.c * delta * np.log(self.t.data[idx]['n_2']),0))
        # print("ee_if:", self.t.data[idx]['ee_if'], "delta:", delta,'using times:',self.t.data[idx]['n_2'])

    def batch_update(self, tree_idx, tree_data_idx, abs_errors):
        # print('update', self.t.tree)
        # abs_errors = abs_errors.data.cpu().numpy()
        # print(abs_errors)
        # print(tree_idx.shape) # (32,)
        # print(abs_errors)
        abs_errors = abs(abs_errors)
        abs_errors += self.t_epsilon  # convert to abs and avoid 0
        # 比较两个数组并返回一个包含元素最小值的新数组(形状不一样的话则需要广播）
        # (episode_num,1) 并且其中最大数就是1
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        # print('c=',clipped_errors)
        # alpha越大，结果越小；clipped_errors大，结果越大； 无论如何，最大值无限接近于1但是比1小
        ps = np.power(clipped_errors, self.alpha)
        ee_ifs = []
        for i in tree_data_idx:
            ee_if = self.t.data[i]['ee_if']
            ee_ifs.append(ee_if)
        # print("ps=",ps)
        for idx, p, ee_if in zip(tree_idx, ps, ee_ifs):
            # print(ti, p)
            self.t.update_tree(idx, p, ee_if)
        # for idx, p in zip(tree_idx, ps):
        #     self.t.update_tree(idx, p, ee_if=0)
