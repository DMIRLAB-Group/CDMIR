import os
import socket
import logging
import argparse

import rospy as rp
import numpy as np
import time


from mpi4py import MPI
from collections import deque
from world import Environment
from vae import VAE

from Logger import Logger
from sac_ae_caps import SAC_Ae_Caps
from utils import generate_points, Cycle_position, Statistics


parser = argparse.ArgumentParser()
parser.add_argument("--policy", default="SAC_CNN")                   # Policy name 
parser.add_argument("--num_agent", default=8)                       # Num of agents in environment
parser.add_argument("--num_barrier", default=0)                     # Num of agents in environment
parser.add_argument("--batch_size", default=128, type=int)          # Batch size for both actor and critic
parser.add_argument("--replayer_buffer", default=20000, type=int)
parser.add_argument("--discount", default=0.99)                     # Discount factor
parser.add_argument("--tau", default=0.005)                         # Target network update rate
parser.add_argument("--learning_rate", default=1e-3)                # Learning rate
parser.add_argument("--max_episodes", default=375, type=int)        # Max episodes to train 
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


def run(comm, env, policy, starting_epoch, autoencoder):
    for epoch in range(starting_epoch, args.max_episodes):
        terminal = False
        next_episode = False
        liveflag = True
        step = 1

        # generate random pose
        if env.index == 0:
            pose_list, goal_list, barrier_list = generate_points(ptBlu=[0, 9], num_env=args.num_agent, num_barrier=args.num_barrier, maxdist=12, dis=2)
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
        init_pose = list(env.get_position())
        comm.barrier()
        distance = env.generate_goal_point(goal_ctrl)
        img = env.get_image(noise_std=args.noise_std)
        Observation = deque([img, img, img, img], maxlen=args.obs_shape[0])
        O_z = np.asarray(Observation)
        goal, speed= env.get_local_goal_and_speed()
        O_g = np.asarray(goal)
        O_v = np.asarray(speed)
        state = [O_z, O_g, O_v]
        while not next_episode and not rp.is_shutdown():
            state_list = comm.gather(state, root=0)
            # generate actions at rank==0
            actions = policy.generate_action(env=env, state_list=state_list)
            # execute actions
            action = comm.scatter(actions, root=0)
            if liveflag == True:
                env.control_vel(action)
                img = env.get_image(noise_std=args.noise_std)
                r, terminal, result = env.get_reward_and_terminate(step, img)
                step += 1
                Observation.append(img)
                next_O_z = np.asarray(Observation)
                next_goal, next_speed = env.get_local_goal_and_speed()
                next_O_g = np.asarray(next_goal)
                next_O_v = np.asarray(next_speed)
                next_state = [next_O_z, next_O_g, next_O_v]
            else:
                env.drones_terminal()
                rp.sleep(0.2)

            if terminal:
                liveflag = False
            
            state = next_state

            gather_info = comm.gather(O_z, root=0)
            if env.index == 0:
                autoencoder.store(gather_info)


            terminal_list = comm.gather(liveflag, root=0)
            terminal_list = comm.bcast(terminal_list, root=0)


            if True not in terminal_list:
                next_episode = True
                if env.index == 0:
                    if epoch != 0 and epoch % 1 == 0:
                        autoencoder.learn()
                    if epoch != 0 and epoch % args.episode_step == 0:
                        autoencoder.save(epoch, AE_path)
                        logger.info('########################## model saved when update {} times#########'
                                    '################'.format(epoch))
        
if __name__ == '__main__':
    # config log
    hostname = socket.gethostname()
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


    kwargs["encoder_type"] = args.encoder_type
    kwargs["decoder_type"] = args.decoder_type
    kwargs["lam_a"] = args.lam_a
    kwargs["lam_s"] = args.lam_s
    kwargs["eps_s"] = args.eps_s
    policy =SAC_Ae_Caps(env, **kwargs)

    starting_epoch = 0

    if rank == 0:
        vae = VAE(z_dim=args.encoder_feature_dim).to('cuda')
        policy_path = '../policy'
        AE_path = '../policy/vae'
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
            logger.info('############Loading Failed###########')
            logger.info('#####################################')
    else:
        actor = None
        critic = None
        policy_path = None
        AE_path = None
        vae = None

    try:
        run(comm=comm, env=env, policy=policy, starting_epoch=starting_epoch, autoencoder=vae)
    except KeyboardInterrupt:
        pass