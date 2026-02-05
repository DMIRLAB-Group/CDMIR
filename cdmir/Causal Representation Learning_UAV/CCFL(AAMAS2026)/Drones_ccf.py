import os
import time
import socket
import logging
import argparse

import rospy as rp
import numpy as np

from mpi4py import MPI
from collections import deque
from world import Environment
from torch.utils.tensorboard import SummaryWriter

from Logger import Logger
from sac_ccf import SAC_CCF_Caps as ccf
from sac_ae_caps import SAC_Ae_Caps as ae
from utils import generate_points, Cycle_position, Statistics, send_data, start_connection, close_connection


parser = argparse.ArgumentParser()
parser.add_argument("--policy", default="SAC_ccf")                   # Policy name
parser.add_argument("--num_agent", default=8)                       # Num of agents in environment
parser.add_argument("--num_barrier", default=1)                     # Num of barrier in environment
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
        terminal = False
        terminals = None
        next_episode = False
        liveflag = True
        ep_reward = 0
        step = 1
        if epoch < 0:

            args.num_barrier = 0


        # generate random pose
        if env.index == 0:
            pose_list, goal_list, barrier_list = generate_points(ptBlu=[0, 9], num_env=args.num_agent, num_barrier=args.num_barrier, maxdist=12, dis=2)
            # pose_list, goal_list, barrier_list = generate_points(ptBlu=[0, 9], num_env=args.num_agent, num_barrier=args.num_barrier, maxdist=16, dis=3)
            # pose_list, goal_list, barrier_list = Cycle_position(ptBlu=[0, 9], num_env=args.num_agent, radius=8)
        else:
            pose_list, goal_list, barrier_list = None, None, None
        
        env.reset_world()
        rp.sleep(2)
        pose_list = comm.bcast(pose_list,root=0)
        goal_list = comm.bcast(goal_list,root=0)
        pose_ctrl = pose_list[env.index]
        goal_ctrl = goal_list[env.index]

        env.drones_init()
        env.reset_barrier_pose(barrier_list, args.num_barrier)
        comm.barrier()
        init_pose = list(env.get_position())
        env.reset_pose(init_pose, pose_ctrl)
        comm.barrier()
        env.generate_goal_point(goal_ctrl)

        comm.barrier()
        if env.index == 0:
            env.start_simPause()
            send_data('0')
        comm.barrier()
        img0 = env.get_image()  # 获取虚拟相机的图像信息，并考虑了噪声
        comm.barrier()
        if env.index == 0:
            send_data('1')
        comm.barrier()
        img1 = env.get_image()  # 获取虚拟相机的图像信息，并考虑了噪声
        comm.barrier()
        env.reset_barrier_pose_below(barrier_list, args.num_barrier)
        comm.barrier()
        img2 = env.get_image()  # 获取虚拟相机的图像信息，并考虑了噪声
        comm.barrier()
        if env.index == 0:
            env.finish_simPause()
        comm.barrier()

        img = env.get_image()
        Observation = deque([img, img, img, img], maxlen=args.obs_shape[0])
        Observation0 = deque([img0, img0, img0, img0], maxlen=args.obs_shape[0])
        Observation1 = deque([img1, img1, img1, img1], maxlen=args.obs_shape[0])
        Observation2 = deque([img2, img2, img2, img2], maxlen=args.obs_shape[0])

        O_z = np.asarray(Observation)
        O_z0 = np.asarray(Observation0)
        O_z1 = np.asarray(Observation1)
        O_z2 = np.asarray(Observation2)

        goal, speed= env.get_local_goal_and_speed()
        O_g = np.asarray(goal)
        O_v = np.asarray(speed)
        state = [O_z, O_g, O_v]
        state_list = comm.gather(state, root=0)
        env.plot_last_pos = []
        while not next_episode and not rp.is_shutdown():
            # generate actions at rank==0
            actions, percentage_zeros_mask_one, percentage_zeros_mask_two = policy.generate_action(env=env, state_list=state_list)
            # execute actions
            action = comm.scatter(actions, root=0)
            if liveflag == True:
                env.control_vel(action)
            env.reset_barrier_pose(barrier_list, args.num_barrier)

            if (step - 1) % 3 == 0:
                comm.barrier()
                if env.index == 0:
                    env.start_simPause()
                    send_data('0')
                comm.barrier()
                img0 = env.get_image()  # 获取虚拟相机的图像信息，并考虑了噪声
                # cv2.imwrite(f'/home/robot/cube_{rank}/cube_image_{timestamp}.png', img0)
                comm.barrier()
                if env.index == 0:
                    send_data('1')
                comm.barrier()
                img1 = env.get_image()  # 获取虚拟相机的图像信息，并考虑了噪声
                # cv2.imwrite(f'/home/stu/ball_{rank}/ball_image_{timestamp}.png', img1)
                comm.barrier()
                env.reset_barrier_pose_below(barrier_list, args.num_barrier)
                comm.barrier()
                img2 = env.get_image()  # 获取虚拟相机的图像信息，并考虑了噪声
                # cv2.imwrite(f'/home/robot/uav_{rank}/uav_image_{timestamp}.png', img2)
                comm.barrier()
                if env.index == 0:
                    env.finish_simPause()
                comm.barrier()

            if liveflag == True:
                img = env.get_image()
                r, terminal, result = env.get_reward_and_terminate(step, img)
                not_done = 1. - float(terminal)
                ep_reward += r - 0.01
                Observation.append(img)
                if (step - 1) % 3 == 0:
                    Observation0.append(img0)
                    Observation1.append(img1)
                    Observation2.append(img2)

                next_O_z = np.asarray(Observation)
                next_O_z0 = np.asarray(Observation0)
                next_O_z1 = np.asarray(Observation1)
                next_O_z2 = np.asarray(Observation2)

                next_goal, next_speed = env.get_local_goal_and_speed()
                next_O_g = np.asarray(next_goal)
                next_O_v = np.asarray(next_speed)
                next_state = [next_O_z, next_O_g, next_O_v]
                shape_0_z = [O_z0, O_z1, O_z2]
                exp = [O_z, O_g, O_v, action, next_O_z, next_O_g, next_O_v, r, not_done]
            if liveflag == False:
                env.drones_terminal()
                exp = None
                rp.sleep(0.2)

            if terminal:
                liveflag = False

            # next state
            state = next_state
            O_z = next_O_z
            O_z0 = next_O_z0
            O_z1 = next_O_z1
            O_z2 = next_O_z2
            O_g = next_O_g
            O_v = next_O_v

            info = [liveflag, exp, state, ep_reward]
            gather_info = comm.gather(info, root=0)
            shape_info = comm.gather(shape_0_z, root=0)

            if env.index == 0:
                terminals = [i[0] for i in gather_info]
                exp_list = [i[1] for i in gather_info]
                state_list = [i[2] for i in gather_info]
                policy.step(exp_list)
                if (step - 1) % 3 == 0:
                    policy.step_2(shape_info)

            step += 1
            terminal_list = comm.bcast(terminals, root=0)

            if True not in terminal_list:
                next_episode = True
                if env.index == 0:
                    ep_rewards = [i[3] for i in gather_info]
                    mean_epr = np.array(ep_rewards).mean()
                    writer.add_scalar("Train/reward", mean_epr, epoch)
                    writer.add_scalar("Percentage/mask_one", percentage_zeros_mask_one, epoch)
                    writer.add_scalar("Percentage/mask_two", percentage_zeros_mask_two, epoch)
                    if epoch != 0 and epoch % 1 == 0:
                        policy.learn(writer, epoch)

        logger.info('Env %02d, Goal (%05.1f, %05.1f, %05.1f), Episode %05d, step %03d, Reward %-5.1f, %s'% \
                        (env.index, goal_ctrl[0], goal_ctrl[1], goal_ctrl[2], epoch + 1, step, ep_reward, result))
        logger_cal.info(ep_reward)

        if env.index == 0:
            writer.flush()
            if epoch != 0 and epoch % args.episode_step == 0:
                policy.save(epoch, policy_path)
                logger.info('########################## model saved when update {} times#########'
                            '################'.format(epoch))
            
    if env.index == 0:
        writer.close()

            
if __name__ == '__main__':
    # config log
    hostname = socket.gethostname()
    # 生成带有时间戳的子文件夹名称
    current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())  # 生成当前时间的时间戳
    subfolder_name = "CVAE+CFS_L1mask权重正则化1e-2" + current_time  # 使用时间戳创建子文件夹名称
    # print("name", hostname)
    if not os.path.exists('../log/' + hostname):
        os.makedirs('../log/' + hostname)
    output_file = '../log/' + hostname + '/output.log'
    cal_file = '../log/' + hostname + '/cal.log'

    logger = Logger(output_file, clevel=logging.INFO, Flevel=logging.INFO, CMD_render=True)
    logger_cal = Logger(cal_file, clevel=logging.INFO, Flevel=logging.INFO, CMD_render=False)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    env = Environment(rank, args.max_timesteps)

    # Initialize policy
    if args.policy == "SAC_ccf":
        kwargs["encoder_type"] = args.encoder_type
        kwargs["decoder_type"] = args.decoder_type
        kwargs["lam_a"] = -1
        kwargs["lam_s"] = -1
        kwargs["eps_s"] = args.eps_s
        policy = ccf(env, **kwargs)
    elif args.policy == "SAC_Ae":
        kwargs["encoder_type"] = args.encoder_type
        kwargs["decoder_type"] = args.decoder_type
        kwargs["lam_a"] = -1
        kwargs["lam_s"] = -1
        kwargs["eps_s"] = args.eps_s
        policy = ae(env, **kwargs)

        

    starting_epoch = 0

    if rank == 0:
        writer = SummaryWriter("my_experiment/" + subfolder_name)
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
        start_connection()
        run(comm=comm, env=env, policy=policy, starting_epoch=starting_epoch)
    except KeyboardInterrupt:
        pass
    finally:
        # 关闭 socket 连接
        close_connection()
