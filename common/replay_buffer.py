import numpy as np
import threading


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape
        self.size = self.args.buffer_size
        self.episode_limit = self.args.episode_limit
        # memory management
        self.current_idx = 0
        self.current_size = 0
        # create the buffer to store info
        print([self.size, self.episode_limit, self.n_agents, self.obs_shape])

        self.c = 0.001

        self.buffers = {'o': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        'u': np.empty([self.size, self.episode_limit, self.n_agents, 1]),
                        's': np.empty([self.size, self.episode_limit, self.state_shape]),
                        'r': np.empty([self.size, self.episode_limit, 1]),
                        'o_next': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        's_next': np.empty([self.size, self.episode_limit, self.state_shape]),
                        'avail_u': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'avail_u_next': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'u_onehot': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'padded': np.empty([self.size, self.episode_limit, 1]),
                        'terminated': np.empty([self.size, self.episode_limit, 1]),

                        'episode_reward': np.empty(self.size),
                        'episode_len': np.empty(self.size),
                        # generate train step
                        'n_1': np.empty(self.size),
                        # using times
                        'n_2': np.empty(self.size),
                        'pri': np.empty(self.size)
                        }
        if self.args.alg == 'maven':
            self.buffers['z'] = np.empty([self.size, self.args.noise_dim])
        # thread lock
        # self.lock = threading.Lock()

        # store the episode

    def store_episode(self, episode_batch, gen_train_step):
        # 这个东西一直都是1
        batch_size = episode_batch['o'].shape[0]
        # batch_size = len(episode_batch)
        # with self.lock:
        idxs = self._get_storage_idx(inc=batch_size)  # store the informations
        # idxs=self.get_pri_storage_idx(batch_size)
        # for i, idx in enumerate(idxs):
        self.buffers['o'][idxs] = episode_batch['o']
        self.buffers['u'][idxs] = episode_batch['u']
        self.buffers['s'][idxs] = episode_batch['s']
        # print(episode_batch['r'])
        self.buffers['r'][idxs] = episode_batch['r']
        self.buffers['o_next'][idxs] = episode_batch['o_next']
        self.buffers['s_next'][idxs] = episode_batch['s_next']
        self.buffers['avail_u'][idxs] = episode_batch['avail_u']
        self.buffers['avail_u_next'][idxs] = episode_batch['avail_u_next']
        self.buffers['u_onehot'][idxs] = episode_batch['u_onehot']
        self.buffers['padded'][idxs] = episode_batch['padded']
        self.buffers['terminated'][idxs] = episode_batch['terminated']
        self.buffers['episode_reward'][idxs] = episode_batch['episode_reward']
        self.buffers['episode_len'][idxs] = episode_batch['episode_len']
        self.buffers['n_1'][idxs] = gen_train_step
        self.buffers['n_2'][idxs] = 0
        self.buffers['pri'][idxs] = 1
        if self.args.alg == 'maven':
            self.buffers['z'][idxs] = episode_batch['z']

            # self.buffers['o'][idx] = episode_batch[i]['o']
            # self.buffers['u'][idx] = episode_batch[i]['u']
            # self.buffers['s'][idx] = episode_batch[i]['s']
            # # print(episode_batch['r'])
            # self.buffers['r'][idx] = episode_batch[i]['r']
            # self.buffers['o_next'][idx] = episode_batch[i]['o_next']
            # self.buffers['s_next'][idx] = episode_batch[i]['s_next']
            # self.buffers['avail_u'][idx] = episode_batch[i]['avail_u']
            # self.buffers['avail_u_next'][idx] = episode_batch[i]['avail_u_next']
            # self.buffers['u_onehot'][idx] = episode_batch[i]['u_onehot']
            # self.buffers['padded'][idx] = episode_batch[i]['padded']
            # self.buffers['terminated'][idx] = episode_batch[i]['terminated']
            # if self.args.alg == 'maven':
            #     self.buffers['z'][idx] = episode_batch[i]['z']

    def sample(self, batch_size, train_steps):
        temp_buffer = {}
        # idx是一个数组，用来当做取batch的下标
        idx = np.random.randint(0, self.current_size, batch_size)
        # idx = np.argpartition(self.buffers['pri'][:self.current_size], -batch_size)[-batch_size:]
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        # print("pri all:", self.buffers['pri'][:self.current_size], "pri batch:", self.buffers['pri'][idx])
        # self.update_pri(idx, train_steps)
        return temp_buffer

    def update_pri(self, idx, train_steps):
        self.buffers['n_2'][idx] += 1
        delta = train_steps - self.buffers['n_1'][idx]
        self.buffers['pri'][idx] = self.buffers['episode_reward'][idx] / self.buffers['episode_len'][idx] - \
                                   self.c * delta * np.log(self.buffers['n_2'][idx])
        # print("batch pri:{},batch using times:{},now train time:{},gen train time:{},batch delta:{}".format(
        #     self.buffers['pri'][idx], self.buffers['n_2'][idx],
        #     train_steps, self.buffers['n_1'][idx], delta))


    # 这里应该是对于前面改成一个batch再存储一次是没啥问题的
    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx

    # 该方法目前的逻辑不支持batch存储
    def get_pri_storage_idx(self,inc):
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)[0]
            self.current_idx += inc
        elif self.current_idx == self.size:
            idx=np.argmin(self.buffers['pri'])
        self.current_size = min(self.size, self.current_size + inc)
        return idx