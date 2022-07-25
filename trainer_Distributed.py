from workers import Worker
from worker_dis import Worker_dis
from agent.agent import Agents
from torch.multiprocessing import Queue
from common.arguments import get_common_args, get_mixer_args
# from tensorboardX import SummaryWriter
from dataloader import DataLoader
import torch.distributed as dist
import torch
import os
import time
from common.arguments import get_common_args, get_mixer_args
import torch.multiprocessing as mp
from common.replay_buffer import ReplayBuffer

# -----初始化CUDA相关------
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
# cuda0 = torch.device('cuda:0')
# cuda1 = torch.device('cuda:1')
# -------------------------


def trainer_dis(args):
    begin = time.time()
    rank=args.local_rank
    # dist.init_process_group("nccl")
    torch.cuda.set_device(args.local_rank)
    fighter_model=Worker_dis(args,rank)
    buffer=ReplayBuffer(args)

    train_steps = 0
    while train_steps<=args.n_epoch//args.batch_size:
        data_batch = fighter_model.data_maker()
        buffer.store_episode(data_batch)
        mini_batch=buffer.sample(args.batch_size)
        fighter_model.agent.train(mini_batch, train_steps)
        train_steps += 1
        print('train %d times' % train_steps)

        if train_steps == 500 or train_steps == 1000 or train_steps == 5000:
            end = time.time()
            print('------------------------', end - begin)

        # share_model始终保持最新的模型参数
        fighter_model.share_model.load_state_dict(fighter_model.agent.policy.eval_rnn.state_dict())

def trainer_dp(args):
    begin = time.time()
    # torch.multiprocessing.set_start_method('spawn')  # linux默认采用fork启动子进程，在这种情况下多进程中重新初始化CUDA会报错
    data_queue = Queue()  # 和worker交互用
    data_queue2 = Queue()
    data_queue3 = Queue()
    data_queues = [data_queue, data_queue2, data_queue3]
    dataload_queue = Queue()

    fighter_model = Agents(args)
    # fighter_model.policy.load_state_dict(share_model.state_dict())

    share_model = fighter_model.policy.eval_rnn
    share_model.share_memory()

    for wk_id, queue in zip(list(range(args.worker_nums)), data_queues):
        worker = Worker(args, share_model, queue, wk_id)
        # work继承了process类，所以也有相应的start方法，效果类似process.start
        worker.start()
    print('{} workers start successfully'.format(args.worker_nums))

    # 开启dataloader process
    dataloader = DataLoader(data_queues, args, dataload_queue)
    dataloader.start()
    print('dataloader start successfully')

    train_steps = 0  # 统计更新的次数：一次更新=1024帧数据

    while True:
        # print("-----Trainer Queue size:", data_queue.qsize())
        # print('----dataloader queue size:',dataload_queue.qsize())
        # time.sleep(4)
        try:
            # print("-----Trainer Queue size:", data_queue.qsize()+data_queue2.qsize()+data_queue3.qsize())
            mini_batch = dataload_queue.get_nowait()
            fighter_model.train(mini_batch, train_steps)
            train_steps += 1
            print('train %d times' % train_steps)
        except:
            continue

        if train_steps == 500 or train_steps == 1000 or train_steps == 5000:
            end = time.time()
            print('------------------------', end - begin)

        # share_model始终保持最新的模型参数
        share_model.load_state_dict(fighter_model.policy.eval_rnn.state_dict())

if __name__ == "__main__":
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

    #distributed default
    args.worker_nums = 1
    args.env_nums = 1
    # args.dp = True
    if args.learn:
        print('learn')
        # trainer_dp(args)
        if args.ddp:
            trainer_dis(args)