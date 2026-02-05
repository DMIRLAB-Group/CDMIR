import os
import time
import socket
import logging
import argparse
from random import random

import rospy as rp
import numpy as np

from mpi4py import MPI
from collections import deque
from world import Environment
from torch.utils.tensorboard import SummaryWriter

from Logger import Logger
from sac_cnn import SAC_CNN
from mask_sac_ae_caps import SAC_Ae_Caps
from sac_vae import SAC_Vae
from utils import generate_points, Cycle_position, seed_everything


parser = argparse.ArgumentParser()
parser.add_argument("--policy", default="SAC_Ae")                   # Policy name 
parser.add_argument("--num_agent", default=8)                       # Num of agents in environment
parser.add_argument("--num_barrier", default=0)                     # Num of barrier in environment
parser.add_argument("--seed", default=0) 
parser.add_argument("--batch_size", default=128, type=int)          # Batch size for both actor and critic
parser.add_argument("--replayer_buffer", default=20000, type=int)
parser.add_argument("--discount", default=0.99)                     # Discount factor
parser.add_argument("--tau", default=0.01)                          # Target network update rate
parser.add_argument("--learning_rate", default=1e-4)                # Learning rate
parser.add_argument("--max_episodes", default=401, type=int)        # Max episodes to train
parser.add_argument("--max_timesteps", default=150, type=int)       # Max time steps to run environment
parser.add_argument("--episode_step", default=20, type=int)         # Time steps to save model
parser.add_argument("--init_steps", default=1000, type=int)   
parser.add_argument("--obs_shape", default=[4, 84, 84], type=list)
parser.add_argument("--action_shape", default=3, type=int)
parser.add_argument("--hidden_dim", default=1024, type=int)
parser.add_argument("--lam_a", default=0.5)
parser.add_argument("--lam_s", default=0.5)
parser.add_argument("--eps_s", default=0.2)
parser.add_argument("--mode", default='train')
parser.add_argument("--encoder_type", default='pixel')
parser.add_argument("--decoder_type", default='pixel')
parser.add_argument("--encoder_feature_dim", default=50, type=int)
args = parser.parse_args()

kwargs = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "replayer_buffer":args.replayer_buffer,
		"obs_shape": args.obs_shape,
        "num_env":args.num_agent,
		"action_shape": args.action_shape,
		"discount": args.discount,
		"tau": args.tau,
        "lr": args.learning_rate,
        "hidden_dim": args.hidden_dim,
        "init_steps": args.init_steps,
        "mode": args.mode,
	}


def run(comm, env, policy, starting_epoch):
    for epoch in range(starting_epoch, args.max_episodes):

        # random.seed(epoch)
        # np.random.seed(epoch)
        terminal = False
        terminals = None
        next_episode = False
        liveflag = True
        ep_reward = 0
        step = 1

        # generate random pose
        if env.index == 0:
            pose_list, goal_list, barrier_list = generate_points(ptBlu=[0, 9], num_env=args.num_agent, num_barrier=args.num_barrier, maxdist=12, dis=2) #根据上述参数在指定空间内生成一定数量的初始位置、目标点和障碍物
            # pose_list, goal_list, barrier_list = generate_points(ptBlu=[0, 9], num_env=args.num_agent, num_barrier=args.num_barrier, maxdist=16, dis=3)
            # pose_list, goal_list, barrier_list = Cycle_position(ptBlu=[0, 9], num_env=args.num_agent, radius=8)
            # print("\npose_list:", pose_list,"\ngoal_list:", goal_list,"\nbarrier_list:", barrier_list)
        else:
            pose_list, goal_list, barrier_list = None, None, None
        
        env.reset_world()
        rp.sleep(2)
        pose_list = comm.bcast(pose_list,root=0)    #将生成的位置和目标点信息传递给所有智能体
        goal_list = comm.bcast(goal_list,root=0)
        pose_ctrl = pose_list[env.index]
        goal_ctrl = goal_list[env.index]

        env.drones_init()
        env.reset_barrier_pose(barrier_list, args.num_barrier)
        comm.barrier()  #同步点操作，确保在继续执行后续代码之前，所有进程都已经完成了前面的操作。
        init_pose = list(env.get_position())    #获取环境中无人机的当前位置，并将其存储在名为 init_pose 的列表中。这个位置信息将用作智能体的初始位置
        env.reset_pose(init_pose, pose_ctrl)#重置智能体的位置。它接受两个参数，init_pose 表示初始位置，pose_ctrl 表示智能体的控制目标位置。智能体的初始位置将被设置为 init_pose，这通常是环境中的一个随机位置。然后，智能体将被移动到 pose_ctrl 所表示的位置，以便开始训练。
        comm.barrier()
        env.generate_goal_point(goal_ctrl)  #用于生成智能体的目标点
        img = env.get_image()   #获取当前环境中无人机的深度图像信息
        Observation = deque([img, img, img, img], maxlen=args.obs_shape[0])#创建了一个双向队列 Observation，用于存储历史观测数据，以便在处理当前观测数据时可以考虑先前的观测状态，
        O_z = np.asarray(Observation)   #将观测历史队列 Observation 转换为NumPy数组，存储在 O_z 中
        goal, speed= env.get_local_goal_and_speed() #获取了智能体的局部目标点和速度信息，并将它们分别存储在 goal 和 speed 中
        O_g = np.asarray(goal)  #将局部目标点信息转换为NumPy数组，存储在 O_g 中。
        O_v = np.asarray(speed)
        state = [O_z, O_g, O_v] #将深度图像观测历史 O_z、局部目标点 O_g 和速度 O_v 组合成一个状态列表 state，该状态将作为智能体的初始状态
        state_list = comm.gather(state, root=0) #将每个智能体的初始状态 state 收集到根进程（rank为0的进程）。这样，根进程将包含所有智能体的初始状态信息。
        env.plot_last_pos = []  #初始化或清空存储智能体轨迹信息的变量
        while not next_episode and not rp.is_shutdown():    #它在满足两个条件之一时终止：next_episode 为真或 rp（可能是一个ROS节点或其他控制程序）要求关闭。
            # generate actions at rank==0
            actions, percentage_zeros_mask_one, percentage_zeros_mask_two = policy.generate_action(env=env, state_list=state_list)    #生成智能体的动作
            # execute actions
            action = comm.scatter(actions, root=0)  #将数据从根进程分发给其他进程
            if liveflag == True:    #如果 liveflag 为真，表示当前智能体仍然活跃，可以执行动作和观测。
                env.control_vel(action) #将动作应用于环境，以控制智能体的速度
                img = env.get_image()
                r, terminal, result = env.get_reward_and_terminate(step, img)   #计算奖励、终止标志和结果
                not_done = 1. - float(terminal) #计算 not_done 变量，它表示智能体是否仍然在执行任务。如果 terminal 为真，表示任务已经结束，not_done 将为 0。否则，not_done 为 1。
                ep_reward += r  #将当前时间步的奖励值 r 添加到当前的总回报 ep_reward 中，以跟踪整个回合的回报
                step += 1
                Observation.append(img) #将当前图像观测 img 添加到观测历史队列 Observation 中。这是为了在处理当前观测数据时考虑先前的观测状态
                next_O_z = np.asarray(Observation)  #将观测历史队列 Observation 转换为NumPy数组 next_O_z，以便将其用作下一个时间步的状态
                next_goal, next_speed = env.get_local_goal_and_speed()  #获取下一个时间步的局部目标和速度
                next_O_g = np.asarray(next_goal)    #将下一个时间步的局部目标和速度转换为NumPy数组
                next_O_v = np.asarray(next_speed)
                next_state = [next_O_z, next_O_g, next_O_v] #将新的状态信息组合成 next_state 列表，以准备用于下一个时间步的计算和决策。
                exp = [O_z, O_g, O_v, action, next_O_z, next_O_g, next_O_v, r, not_done]    #创建一个经验（experience）列表，用于在强化学习中记录智能体与环境互动的数据
            else:
                env.drones_terminal()   #用于在环境中处理无人机（drones）终止的情况
                exp = None  #在当前时间步内不存储任何经验数据
                rp.sleep(0.2)

            if terminal:
                liveflag = False

            # next state
            state = next_state  #将当前时间步的状态更新为下一时间步的状态，以便在下一个时间步中使用
            O_z = next_O_z
            O_g = next_O_g
            O_v = next_O_v

            #liveflag：表示智能体是否仍然“活跃”，即智能体是否仍然在与环境互动。这个信息通常用于确定是否应继续进行强化学习的迭代
            #exp：包含了关于智能体与环境互动的经验数据。这个数据可能包括当前状态、执行的动作、下一个状态、获得的奖励等等
            #state：表示智能体的当前状态
            #ep_reward：表示智能体在当前阶段（episode）中已经获得的总奖励
            info = [liveflag, exp, state, ep_reward]
            gather_info = comm.gather(info, root=0) #将所有智能体的信息发送到根节点，以便在根节点进行进一步的处理或记录。

            if env.index == 0:
                terminals = [i[0] for i in gather_info] #创建了一个名为 terminals 的列表，其中包含了从 gather_info 中提取的所有智能体的 liveflag 值。gather_info 包含了每个智能体在当前时间步的信息，其中 liveflag 表示每个智能体是否仍然活跃。
                exp_list = [i[1] for i in gather_info]  #创建了一个名为 exp_list 的列表，其中包含了从 gather_info 中提取的所有智能体的经验数据，即 exp。经验数据通常包括当前状态、执行的动作、下一个状态、获得的奖励等等
                state_list = [i[2] for i in gather_info] #创建了一个名为 state_list 的列表，其中包含了从 gather_info 中提取的所有智能体的当前状态信息。在多智能体环境中，每个智能体都有自己的状态
                policy.step(exp_list)   #调用 policy.step(exp_list)，智能体使用这些经验数据来更新自己的策略，以更好地适应环境


            terminal_list = comm.bcast(terminals, root=0)   #将 terminals 列表的值广播（broadcast）给所有智能体

            if True not in terminal_list:   #terminal_list 包含了每个智能体的 liveflag 值，如果列表中没有 True，意味着所有智能体都已经终止了当前的训练轮次
                next_episode = True #表示下一轮的训练即将开始
                if env.index == 0:
                    ep_rewards = [i[3] for i in gather_info]    #从 gather_info 列表中提取每个智能体的累积奖励值，并将这些奖励值存储在名为 ep_rewards 的列表中
                    mean_epr = np.array(ep_rewards).mean()  #计算 ep_rewards 列表中所有智能体的平均累积奖励值，并将结果存储在 mean_epr 变量中
                    writer.add_scalar("Train/reward", mean_epr, epoch)
                    writer.add_scalar("Percentage/mask_one", percentage_zeros_mask_one, epoch)
                    writer.add_scalar("Percentage/mask_two", percentage_zeros_mask_two, epoch)
                    if epoch != 0 and epoch % 1 == 0:   #检查是否已经经过了至少一个训练轮次（epoch != 0）并且 epoch 是1的倍数（epoch % 1 == 0）
                        policy.learn(writer, epoch)  #将利用智能体在当前训练轮次中收集到的经验数据来更新策略，以便智能体在下一轮训练中表现得更好

        logger.info('Env %02d, Goal (%05.1f, %05.1f, %05.1f), Episode %05d, step %03d, Reward %-5.1f, %s'% \
                        (env.index, goal_ctrl[0], goal_ctrl[1], goal_ctrl[2], epoch + 1, step, ep_reward, result))
        logger_cal.info(ep_reward)

        if env.index == 0:
            writer.flush()  #这是一个日志写入器（writer）的方法，用于将任何还没有写入到磁盘的日志消息强制写入磁盘。这可以确保日志消息被及时记录
            if epoch != 0 and epoch % args.episode_step == 0:   #检查当前训练轮次 epoch 是否不为0且是否是args.episode_step的倍数。
                policy.save(epoch, policy_path) #将模型的当前状态保存到磁盘上的指定路径policy_path中
                logger.info('########################## model saved when update {} times#########'
                            '################'.format(epoch))
            
    if env.index == 0:
        writer.close()  #通过执行writer.close()，它会关闭日志写入器，这将确保在日志文件中记录的所有信息都被保存并写入磁盘文件中。这个操作通常发生在训练结束时。

            
if __name__ == '__main__':
    # config log
    hostname = socket.gethostname()
    # 指定子文件夹名称
    # subfolder_name = "experiment_6"

    # 生成带有时间戳的子文件夹名称
    current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())  # 生成当前时间的时间戳
    subfolder_name = "experiment_" + current_time  # 使用时间戳创建子文件夹名称

    # seed = 11

    if not os.path.exists('../log/' + hostname):
        os.makedirs('../log/' + hostname)
    output_file = '../log/' + hostname + '/output.log'
    cal_file = '../log/' + hostname + '/cal.log'

    logger = Logger(output_file, clevel=logging.INFO, Flevel=logging.INFO, CMD_render=True) #创建了一个名为 logger 的日志记录器对象，它将程序的输出写入到 output_file 文件中。这个日志记录器被配置为记录 INFO 级别的日志，并且可以在终端（CMD_render=True）上显示日志消息
    logger_cal = Logger(cal_file, clevel=logging.INFO, Flevel=logging.INFO, CMD_render=False)  #创建了另一个名为 logger_cal 的日志记录器对象，它用于记录计算相关的日志。它也被配置为记录 INFO 级别的日志，但不在终端上显示日志消息（CMD_render=False）。

    # seed_everything(seed)   #设置随机种子

    comm = MPI.COMM_WORLD   #创建了一个 MPI 通信对象，用于多进程通信
    rank = comm.Get_rank()  #获取当前进程的 MPI 排名（rank），即该进程在 MPI 通信中的唯一标识符。
    size = comm.Get_size()  #获取 MPI 通信中的进程总数，即通信组的大小
    env = Environment(rank, args.max_timesteps) #创建了一个名为 env 的环境对象，并传递了当前进程的 MPI 排名和 args.max_timesteps 参数作为参数。这个环境对象用于模拟和控制程序的环境，每个进程都有一个独立的环境对象。

    # Initialize policy
    if args.policy == "SAC_CNN":
        policy = SAC_CNN(env, **kwargs)#创建一个 SAC_CNN 类的对象，并将这个对象赋值给变量 policy。使用 env 和 kwargs 作为参数来初始化 SAC_CNN 类的对象。kwargs 是一个字典，包含了一些配置参数，这些参数会传递给 SAC_CNN 类的初始化函数
    elif args.policy == "SAC_Ae":
        kwargs["encoder_type"] = args.encoder_type
        kwargs["decoder_type"] = args.decoder_type
        kwargs["lam_a"] = -1
        kwargs["lam_s"] = -1
        kwargs["eps_s"] = args.eps_s
        policy =SAC_Ae_Caps(env, **kwargs)
    elif args.policy == "SAC_Vae":
        policy =SAC_Vae(env, **kwargs)
        

    starting_epoch = 0

    if rank == 0:
        writer = SummaryWriter("my_experiment/" + subfolder_name)   #创建一个名为writer的TensorBoard SummaryWriter对象，用于记录训练过程中的数据，例如训练曲线、奖励等
        policy_path = '../policy'
        if not os.path.exists(policy_path):
            os.makedirs(policy_path)

        model_file = policy_path + '/model'
        if os.path.exists(model_file):
            logger.info('####################################')
            logger.info('############Loading Model###########')
            logger.info('####################################')
            
            starting_epoch = policy.load(model_file, args.mode)
        else:
            logger.info('#####################################')
            logger.info('############Start Training###########')
            logger.info('#####################################')
    else:
        actor = None
        critic = None
        policy_path = None

    try:
        # starting_epoch = comm.bcast(starting_epoch, root=0)
        starting_epoch = 0
        run(comm=comm, env=env, policy=policy, starting_epoch=starting_epoch)
    except KeyboardInterrupt:
        pass
