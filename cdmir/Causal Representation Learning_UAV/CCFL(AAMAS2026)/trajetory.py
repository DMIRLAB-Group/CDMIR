import os
import logging
import argparse
import time

import rospy as rp
import numpy as np


from mpi4py import MPI
from collections import deque
from world import Environment

from Logger import Logger
from sac_cnn import SAC_CNN
from sac_vae import SAC_Vae
from sac_ccf import SAC_CCF_Caps as ccf
from sac_ae_caps import SAC_Ae_Caps
from utils import Cycle_position


parser = argparse.ArgumentParser()
parser.add_argument("--policy", default="SAC_ccf")                   # Policy name
parser.add_argument("--num_agent", default=8)                      # Num of agents in environment
parser.add_argument("--num_barrier", default=4)                     # Num of agents in environment
parser.add_argument("--batch_size", default=128, type=int)          # Batch size for both actor and critic
parser.add_argument("--replayer_buffer", default=20000, type=int)
parser.add_argument("--discount", default=0.99)                     # Discount factor
parser.add_argument("--tau", default=0.005)                         # Target network update rate
parser.add_argument("--learning_rate", default=1e-3)                # Learning rate
parser.add_argument("--max_episodes", default=375, type=int)        # Max episodes to train 
parser.add_argument("--max_timesteps", default=400, type=int)       # Max time steps to run environment
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
    result_list = [False] * args.num_agent
    if not os.path.exists('../log/position/'+ args.policy):
        os.makedirs('../log/position/'+ args.policy)
    file = '../log/position/'+ args.policy + '/uav_' + str(env.index) +'.log'
    logger = Logger(file, clevel=logging.INFO, Flevel=logging.INFO, CMD_render=False)   
    while False in result_list:
        os.remove(file)
        logger = Logger(file, clevel=logging.INFO, Flevel=logging.INFO, CMD_render=False)   
        terminal = False
        next_episode = False
        liveflag = True
        step = 1
        path = 0
        env.client.simFlushPersistentMarkers()

        # generate random pose
        if env.index == 0:
            pose_list, goal_list, barrier_list = Cycle_position(ptBlu=[0, 9], num_env=args.num_agent, radius=12)
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
            actions, _, _  = policy.generate_action(env=env, state_list=state_list)
            # execute actions
            action = comm.scatter(actions, root=0)
            if liveflag == True:
                position = list(env.get_position())
                logger.info('%.3f,%.3f,%.3f' % (position[0], position[1], position[2]))
                env.control_vel(action)
                init_pose, path = env.plot_trajecy(init_pose, path, color_rgba, render_plot)
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

            terminal_list = comm.gather(liveflag, root=0)
            terminal_list = comm.bcast(terminal_list, root=0)

            if True not in terminal_list:
                next_episode = True
                if result == 'Reach Goal':
                    r = True
                else:
                    r = False
                result_list = comm.gather(r, root=0)
                result_list = comm.bcast(result_list, root=0)
                if env.index == 0:
                    print(result_list)
                
                
            


        
            
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    env = Environment(rank, args.max_timesteps)
    policy_path = '../policy'

    # Initialize policy
    if args.policy == "SAC_CNN":
        policy = SAC_CNN(env, **kwargs)
        model_file = policy_path + '/model-cnn'
    elif args.policy == "SAC_ccf":
        kwargs["encoder_type"] = args.encoder_type
        kwargs["decoder_type"] = args.decoder_type
        kwargs["lam_a"] = -1
        kwargs["lam_s"] = -1
        kwargs["eps_s"] = args.eps_s
        model_file = policy_path + '/CCF'
        policy =ccf(env, **kwargs)
    elif args.policy == "SAC_Vae":
        policy =SAC_Vae(env, **kwargs)
        model_file = policy_path + '/VAE'

    starting_epoch = 0

    if rank == 0:
        if not os.path.exists(policy_path):
            os.makedirs(policy_path)

        if os.path.exists(model_file):
            print('####################################')
            print('############Loading Model###########')
            print('####################################')
            
            starting_epoch = policy.load(model_file, args.mode)
        else:
            print('#####################################')
            print('############Start Training###########')
            print('#####################################')
    else:
        actor = None
        critic = None
        policy_path = None

    try:
        run(comm=comm, env=env, policy=policy, starting_epoch=starting_epoch)
    except KeyboardInterrupt:
        pass