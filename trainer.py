import numpy as np

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


# -----初始化CUDA相关------
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# cuda0 = torch.device('cuda:0')
# cuda1 = torch.device('cuda:1')
# -------------------------


def trainer(args):
    begin=time.time()
    buffer = ReplayBuffer(args)
    rank = args.local_rank
    # dist.init_process_group("nccl")
    # torch.cuda.set_device(args.local_rank)
    # 这一句代码是显存爆掉的根源
    torch.multiprocessing.set_start_method('spawn')  # linux默认采用fork启动子进程，在这种情况下多进程中重新初始化CUDA会报错
    data_queue = Queue()  # 和worker交互用

    fighter_model = Agents(args,args.local_rank)
    # fighter_model.policy.load_state_dict(share_model.state_dict())

    share_model = fighter_model.policy.eval_rnn
    share_model.share_memory()

    workers = []
    for wk_id in range(args.worker_nums):
        worker = Worker(args, share_model, data_queue, wk_id)
        # work继承了process类，所以也有相应的start方法，效果类似process.start
        worker.start()
        workers.append(worker)
    print('{} workers start successfully'.format(args.worker_nums))

    train_steps = 0  # 统计更新的次数：一次更新=1024帧数据
    experience_batch = []
    while True:
        try:
            # 当队列为空时，再调用get函数，程序会阻塞，导致无法正常执行后面的代码，程序也不会退出，可以用get_nowait函数，
            # 当队列为空，不会等待，直接抛出异常，若想输出后面的内容，可以用try…finally…捕获异常执行。
            experience = data_queue.get_nowait()
            experience_batch.append(experience)

        except:
            continue
        # 返回队列的大致大小。注意，qsize() > 0 不保证后续的 get() 不被阻塞，
        # qsize() < maxsize 也不保证 put() 不被阻塞。
        print("-----Trainer Queue size:", data_queue.qsize())
        # print("更新第"+str(update_count)+"次")
        # 可以改成每次存一批 在输入端改  现在是一次存一条episode
        if len(experience_batch) == args.batch_size:
            buffer.store_episode(experience_batch)
            experience_batch = []

        print('*****buffer size:', buffer.current_size)
        if data_queue.qsize() <= 20:
            while data_queue.qsize() <= 50:
                time.sleep(4)
        # 这里可能需要改一下，目前是大于0就会采样数据
        if buffer.current_size >= args.batch_size:
            # 从buffer中采样一个batch的数据用于训练 随机选的
            mini_batch = buffer.sample(args.batch_size)
            # print(mini_batch)
            # np.save('test.npy',mini_batch)
            fighter_model.train(mini_batch, train_steps)
            train_steps += 1
            print('train %d times' % train_steps)

        if train_steps==500 or train_steps==1000 or train_steps==5000:
            end=time.time()
            print('------------------------',end-begin)

        # share_model始终保持最新的模型参数
        share_model.load_state_dict(fighter_model.policy.eval_rnn.state_dict())
    # while True:
    #     pass


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

    args.worker_nums = 3
    args.env_nums = 4
    args.ddp = False
    if args.learn:
        print('learn')
        trainer(args)

# 一个worker 不开actor是2.9G显存占用，开一个actor是4.6G，平均一个actor占1.6G显存
# 两个worker 不开actor是4.4G显存占用，各开一个actor是8.1G，平均一个actor占1.6G显存
# 三个worker 不开actor是5.9G显存占用，各开一个actor是11.3G，平均一个actor是1.8G显存
# 四个worker 不开actor是7.4G显存占用，各开一个actor是14.5G，平均一个actor是1.8G显存

# 总结 主进程是1.5G左右的显存占用，每个worker1.5G显存占用，每个actor1.7G显存占用
# 并且开多个worker-actor时，GPU利用率会显著提升

#这个代码不会创建僵尸进程（把gpu清空之后）  会在tmp中创建6个sc的文件夹

#worker和actor都是进程， 53s 500次训练
#worker是线程，actor是进程，79s 500次训练
#worker是进程，actor是线程，66s 500次训练 但是数据队列中的数据最多就60多 没有到200
#worker和actor都是线程， 274s 500次训练 明显感觉到数据产生慢