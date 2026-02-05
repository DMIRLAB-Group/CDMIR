import os
import socket
import logging
import argparse

import rospy as rp
import numpy as np
import time
import random

from mpi4py import MPI
from collections import deque
from world import Environment

from Logger import Logger
from sac_cnn import SAC_CNN
from mask_sac_ae_caps import SAC_Ae_Caps
from sac_vae import SAC_Vae
from utils import generate_points, Cycle_position, Statistics, seed_everything



parser = argparse.ArgumentParser()
parser.add_argument("--policy", default="SAC_Ae")                   # Policy name 
parser.add_argument("--num_agent", default=8)                       # Num of agents in environment
parser.add_argument("--num_barrier", default=4)                     # Num of agents in environment
parser.add_argument("--batch_size", default=128, type=int)          # Batch size for both actor and critic
parser.add_argument("--replayer_buffer", default=20000, type=int)
parser.add_argument("--discount", default=0.99)                     # Discount factor
parser.add_argument("--tau", default=0.005)                         # Target network update rate
parser.add_argument("--learning_rate", default=1e-3)                # Learning rate
parser.add_argument("--max_episodes", default=375, type=int)        # Max episodes to train 375
parser.add_argument("--max_timesteps", default=250, type=int)       # Max time steps to run environment
parser.add_argument("--episode_step", default=20, type=int)         # Time steps to save model
parser.add_argument("--init_steps", default=1000, type=int)  
parser.add_argument("--obs_shape", default=[4,84,84], type=list)
parser.add_argument("--action_shape", default=3, type=int)
parser.add_argument("--hidden_dim", default=1024, type=int)
parser.add_argument("--noise_std", default=0)
parser.add_argument("--lam_a", default=1)
parser.add_argument("--lam_s", default=0.5)
parser.add_argument("--eps_s", default=0.2)
parser.add_argument("--mode", default='test')
parser.add_argument("--encoder_type", default='pixel')
parser.add_argument("--decoder_type", default='pixel')
parser.add_argument("--encoder_feature_dim", default=50, type=int)
args = parser.parse_args()

kwargs = {
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
        "mode": args.mode
	}

color_rgba = [[1,0,0,0.75],[1,0.6471,0,0.75],[1,1,0,0.75],[0,1,0,0.75],[0,0.5,1,0.75],[0,0,1,0.75],[0.55,0,1,0.75],[0.5,0.5,0.5,0.75],
                [1,0,0,0.75],[1,0.6471,0,0.75],[1,1,0,0.75],[0,1,0,0.75],[0,0.5,1,0.75],[0,0,1,0.75],[0.55,0,1,0.75],[0.5,0.5,0.5,0.75]]
render_plot = False

def run(comm, env, policy, starting_epoch):
    c_suceess = 0
    epo_success_count = 0  # 统计一轮八架无人机都成功的轮次
    c_crash = 0
    cnt = 0
    spl = 0
    statistics = Statistics(capacity=50000)


    for epoch in range(int(3000 / args.num_agent)+1):
        terminal = False
        next_episode = False
        liveflag = True
        epo_success = True  # 统计一轮八架无人机都成功的轮次
        step = 1
        path = 0
        velocity = 0
        env.client.simFlushPersistentMarkers()  #清除之前的任务的标记，以便在新的任务开始时，环境处于干净的状态

        if epoch != 0  and epoch % (int(1000 / args.num_agent)) == 0 and env.index == 0:    #验证当前epoch是否是1000除以args.num_agent的整数倍,验证env.index是否等于0
            print("Success rate: %.3f, Crash rate: %.3f, SPL: %.3f, Extra Distance: %.3f/%.3f, Average Speed: %.3f/%.3f, Count:%04d"
            %(c_suceess / cnt, c_crash / cnt, spl / cnt, statistics.memory['Extra Distance'].mean(), statistics.memory['Extra Distance'].std(), statistics.memory['Average Speed'].mean(), statistics.memory['Average Speed'].std(), cnt))
            statistics = Statistics(capacity=50000) #创建了一个新的Statistics对象，并将其赋值给statistics变量。这个对象可能用于在后续的任务中收集和计算任务统计信息
            c_suceess = 0
            c_crash = 0
            cnt = 0
            spl = 0
        # generate random pose
        if env.index == 0:
            # print(np.random.randint(1, 101))
            # print(np.random.randint(1, 101))
            # if epoch < (int(1000 / args.num_agent)):
            #     pose_list, goal_list, barrier_list = generate_points(ptBlu=[0, 9], num_env=args.num_agent, num_barrier=args.num_barrier, maxdist=9, dis=2)
            # elif epoch < (int(2000 / args.num_agent)):
            # pose_list, goal_list, barrier_list = generate_points(ptBlu=[0, 9], num_env=args.num_agent, num_barrier=args.num_barrier, maxdist=12, dis=2)
            # print("\npose_list:", pose_list,"\ngoal_list:", goal_list,"\nbarrier_list:", barrier_list)

            # else:
            #     pose_list, goal_list, barrier_list = generate_points(ptBlu=[0, 9], num_env=args.num_agent, num_barrier=args.num_barrier, maxdist=15, dis=2)
            pose_list, goal_list, barrier_list = Cycle_position(ptBlu=[0, 9], num_env=args.num_agent, num_barrier=args.num_barrier, radius=16)
        else:
            pose_list, goal_list, barrier_list = None, None, None
        
        env.reset_world()   #重置模拟环境的世界状态
        rp.sleep(2)
        pose_list = comm.bcast(pose_list,root=0)    #使用 MPI 通信库中的 comm.bcast 函数，从根进程（root=0）广播 pose_list 和 goal_list 的值给其他进程
        goal_list = comm.bcast(goal_list,root=0)
        pose_ctrl = pose_list[env.index]    #根据当前进程的索引 env.index，从 pose_list 和 goal_list 中选择出当前进程需要使用的位置和目标信息
        goal_ctrl = goal_list[env.index]

        env.drones_init()   #初始化模拟环境中的无人机
        env.reset_barrier_pose(barrier_list, args.num_barrier)  #将障碍物的位置重置为 barrier_list 中指定的位置信息
        comm.barrier()  #从根进程（root=0）广播 pose_list 和 goal_list 的值给其他进程
        init_pose = list(env.get_position())
        env.reset_pose(init_pose, pose_ctrl)  #将虚拟飞行器的位置和姿态重置为 init_pose 中指定的初始值，并尝试将其移动到 pose_ctrl 中指定的目标位置和姿态
        init_pose = list(env.get_position())  #获取模拟环境中的位置信息
        comm.barrier()  #同步各个进程，确保它们都达到了这一点，然后继续执行
        distance = env.generate_goal_point(goal_ctrl)   #生成一个目标点，并计算当前位置到目标点的距离。
        img = env.get_image(noise_std=args.noise_std)   #获取模拟环境中的图像数据
        Observation = deque([img, img, img, img], maxlen=args.obs_shape[0]) #创建了一个长度为 args.obs_shape[0] 的队列（deque），并将刚刚获取的图像数据添加到队列中。这个队列似乎用于保存最近的观测数据，以便后续使用
        O_z = np.asarray(Observation)   #将队列中的观测数据转换为 NumPy 数组，以便后续的处理。
        goal, speed= env.get_local_goal_and_speed() #获取局部目标点和速度信息
        O_g = np.asarray(goal)
        O_v = np.asarray(speed)
        state = [O_z, O_g, O_v]
        while not next_episode and not rp.is_shutdown():    #直到满足 next_episode 为 True 或 rp.is_shutdown() 为 True 的条件才会退出循环
            state_list = comm.gather(state, root=0) #comm.gather函数将每个进程的状态信息汇总到根进程（root=0）中
            # generate actions at rank==0
            actions = policy.generate_action(env=env, state_list=state_list)    #在根进程（rank==0）中，根据当前的状态信息(state_list)生成动作(actions)
            # execute actions
            action = comm.scatter(actions, root=0)  #将根进程生成的动作广播到所有其他进程
            if liveflag == True:
                env.control_vel(action) #根据接收到的动作(action)来控制虚拟飞行器的速度或位置等
                init_pose, path = env.plot_trajecy(init_pose, path, color_rgba, render_plot)    #调用env的plot_trajecy函数来更新虚拟飞行器的轨迹，并返回更新后的初始位置(init_pose)和路径长度(path)
                img = env.get_image(noise_std=args.noise_std)   #获取虚拟相机的图像信息，并考虑了噪声
                r, terminal, result = env.get_reward_and_terminate(step, img)   #根据当前步数(step)和图像信息(img)计算奖励(r)，并检查是否达到终止条件(terminal)以及达到了什么结果(result)，例如，是否成功到达目标或发生了碰撞
                step += 1
                Observation.append(img)
                next_O_z = np.asarray(Observation)
                next_goal, next_speed = env.get_local_goal_and_speed()  #获取下一状态的本地目标位置和速度信息
                next_O_g = np.asarray(next_goal)
                next_O_v = np.asarray(next_speed)
                velocity += np.sqrt(next_speed[0]**2 + next_speed[1]**2 + next_speed[2]**2) #计算速度，这里使用了欧几里得范数来计算速度的大小
                next_state = [next_O_z, next_O_g, next_O_v] #构建下一状态
            else:   #表示虚拟飞行器已经终止，执行相关终止操作
                env.drones_terminal()   #执行虚拟飞行器终止操作
                rp.sleep(0.2)

            if terminal:    #terminal通常用于表示虚拟飞行器是否已经完成了当前任务，可能是成功到达目标或发生了碰撞等
                liveflag = False

            state = next_state

            terminal_list = comm.gather(liveflag, root=0)   #将虚拟飞行器是否终止的信息汇总到根进程
            terminal_list = comm.bcast(terminal_list, root=0)   #根进程将终止信息广播给所有其他进程，以同步它们的终止状态


            if True not in terminal_list:   #检查 terminal_list 中是否没有任何一个元素的值为 True，也就是所有虚拟飞行器都已经终止
                next_episode = True     #表示下一个场景或周期将开始，当前场景已经结束
                if result == "Reach Goal":  #如果 result 等于 "Reach Goal"（表示虚拟飞行器成功到达目标），则计算 w_spl（成功到达目标的分数）、extra_distance（超出目标距离）和 avr_speed（平均速度）
                    w_spl = distance / max(distance, path)
                    extra_distance = path - distance
                    avr_speed = velocity / (step - 1 )
                else:
                    w_spl = 0
                    extra_distance = 0
                    avr_speed = 0
                result_list = comm.gather([result, w_spl, extra_distance, avr_speed], root=0)   #将计算得到的指标 [result, w_spl, extra_distance, avr_speed] 收集到 result_list 中，这通常用于在多个并行运行的虚拟飞行器之间共享结果

            if env.index == 0 and next_episode: #这段代码负责在每个场景或周期结束后，根据虚拟飞行器的执行结果计算和输出一系列汇总统计信息
                for r in result_list:   #遍历 result_list 中的每个元素 r，其中 result_list 包含了上一个场景或周期中计算的指标
                    if r[0] == "Reach Goal":
                        c_suceess += 1  #统计成功的虚拟飞行器数量
                        statistics.store(r[2],r[3]) #将额外距离和平均速度存储到 statistics 中，以便后续分析和汇总
                    elif r[0] == "Crashed": #表示虚拟飞行器发生了碰撞
                        c_crash += 1    #统计碰撞的虚拟飞行器数量
                        epo_success = False
                    else:
                        epo_success = False
                    spl += r[1]
                cnt += args.num_agent
                if epo_success:
                    epo_success_count += 1  # 如果一轮中有八架成功，则轮成功次数加一
                print("Success rate: %.3f, Epo_Success rate: %.3f, Crash rate: %.3f, SPL: %.3f, Extra Distance: %.3f/%.3f, Average Speed: %.3f/%.3f, Count:%04d"
                        % (c_suceess / cnt, epo_success_count / (epoch + 1), c_crash / cnt, spl / cnt, statistics.memory['Extra Distance'].mean(), statistics.memory['Extra Distance'].std(), statistics.memory['Average Speed'].mean(), statistics.memory['Average Speed'].std(), cnt))




        
            
if __name__ == '__main__':
    # config log
    hostname = socket.gethostname()

    seed = 11

    if not os.path.exists('../log/' + hostname):
        os.makedirs('../log/' + hostname)
    output_file = '../log/' + hostname + '/output.log'
    cal_file = '../log/' + hostname + '/cal.log'
    policy_path = '../policy'

    logger = Logger(output_file, clevel=logging.INFO, Flevel=logging.INFO, CMD_render=True)
    logger_cal = Logger(cal_file, clevel=logging.INFO, Flevel=logging.INFO, CMD_render=False)

    # seed_everything(seed)  # 设置随机种子

    comm = MPI.COMM_WORLD

    # seed = comm.bcast(seed, root=0)

    rank = comm.Get_rank()
    size = comm.Get_size()
    env = Environment(rank, args.max_timesteps)

    # Initialize policy
    if args.policy == "SAC_CNN":
        policy = SAC_CNN(env, **kwargs)
        model_file = policy_path + '/model-cnn'
    elif args.policy == "SAC_Ae":
        kwargs["encoder_type"] = args.encoder_type
        kwargs["decoder_type"] = args.decoder_type
        kwargs["lam_a"] = -1
        kwargs["lam_s"] = -1
        kwargs["eps_s"] = args.eps_s
        model_file = policy_path + '/AE'
        policy =SAC_Ae_Caps(env, **kwargs)
    elif args.policy == "SAC_Vae":
        policy =SAC_Vae(env, **kwargs)
        model_file = policy_path + '/VAE'

    starting_epoch = 0

    if rank == 0:
        if not os.path.exists(policy_path):
            os.makedirs(policy_path)
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

    try:
        run(comm=comm, env=env, policy=policy, starting_epoch=starting_epoch)
    except KeyboardInterrupt:
        pass