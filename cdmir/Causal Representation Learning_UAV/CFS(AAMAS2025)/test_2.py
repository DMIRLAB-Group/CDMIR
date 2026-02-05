import os
import socket
import logging
import argparse

import rospy as rp
import numpy as np


from mpi4py import MPI
from collections import deque
from world import Environment

from Logger import Logger
from TD3 import TD3
from td3_ae import TD3_Ae
from sac_ae import SAC_Ae
from sac_ae_caps import SAC_Ae_Caps
from utils import generate_points


parser = argparse.ArgumentParser()
parser.add_argument("--policy", default="SAC_Ae_Caps")                   # Policy name 
parser.add_argument("--num_agent", default=8)                       # Num of agents in environment
parser.add_argument("--expl_noise", default=0.2)                    # Std of Gaussian exploration noise
parser.add_argument("--batch_size", default=128, type=int)          # Batch size for both actor and critic
parser.add_argument("--replayer_buffer", default=20000, type=int)
parser.add_argument("--discount", default=0.99)                     # Discount factor
parser.add_argument("--tau", default=0.005)                         # Target network update rate
parser.add_argument("--learning_rate", default=1e-3)                # Learning rate
parser.add_argument("--policy_noise", default=0.2)                  # Noise added to target policy during critic update
parser.add_argument("--noise_clip", default=0.5)                    # Range to clip target policy noise
parser.add_argument("--policy_freq", default=2, type=int)           # Frequency of delayed policy updates
parser.add_argument("--max_episodes", default=401, type=int)        # Max episodes to train 
parser.add_argument("--max_timesteps", default=250, type=int)       # Max time steps to run environment
parser.add_argument("--episode_step", default=20, type=int)         # Time steps to save model
parser.add_argument("--init_steps", default=1000, type=int)
parser.add_argument("--start_update_step", default=200, type=int)    
parser.add_argument("--obs_shape", default=[3,84,84], type=list)
parser.add_argument("--action_shape", default=3, type=int)
parser.add_argument("--feature_dim", default=32, type=int)
parser.add_argument("--hidden_dim", default=256, type=int)
parser.add_argument("--lam_a", default=1)
parser.add_argument("--lam_s", default=0.5)
parser.add_argument("--eps_s", default=0.5)
parser.add_argument("--mode", default='test')
parser.add_argument("--encoder_type", default='pixel')
parser.add_argument("--decoder_type", default='pixel')
parser.add_argument("--encoder_feature_dim", default=50, type=int)
parser.add_argument("--add_attention", default=True)
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
        "mode": args.mode,
	}

color_rgba = [[1,0,0,0.75],[1,0.6471,0,0.75],[1,1,0,0.75],[0,1,0,0.75],[0,0.5,1,0.75],[0,0,1,0.75],[0.55,0,1,0.75],[0.5,0.5,0.5,0.75]]
render_plot = False

def run(comm, env, policy, starting_epoch):
    c_suceess = 0
    c_crash = 0
    cnt = 0
    for _ in range(args.max_episodes):
        terminal = False
        next_episode = False
        liveflag = True
        step = 1
        env.plot_last_pos = []
        env.client.simFlushPersistentMarkers()

        # generate random pose
        if env.index == 0:
            pose_list, goal_list= generate_points(ptBlu=[0, 9], num_env=args.num_agent, maxdist=12, dis=2)
            # pose_list, goal_list= generate_points(ptBlu=[0, 9], num_env=args.num_agent, maxdist=16, dis=3)
        else:
            pose_list, goal_list = None, None
        
        env.reset_world()
        rp.sleep(2)
        pose_list = comm.bcast(pose_list,root=0)
        goal_list = comm.bcast(goal_list,root=0)
        pose_ctrl = pose_list[env.index]
        goal_ctrl = goal_list[env.index]

        env.drones_init()
        comm.barrier()
        init_pose = list(env.get_position())
        env.reset_pose(init_pose, pose_ctrl)
        comm.barrier()
        env.generate_goal_point(goal_ctrl)
        img = env.get_image()
        img = np.clip(img, a_min=0.2 ,a_max=20)
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
                env.plot_trajecy(color_rgba, render_plot)
                img = env.get_image()
                img = np.clip(img, a_min=0.2 ,a_max=20)
                r, terminal, result = env.get_reward_and_terminate(step, img)
                step += 1
                Observation.append(img)
                next_O_z = np.asarray(Observation)
                next_goal, next_speed = env.get_local_goal_and_speed()
                next_O_g = np.asarray(next_goal)
                next_O_v = np.asarray(next_speed)
                next_state = [next_O_z, next_O_g, next_O_v]
            else:
                action = [0, 0, 0]
                env.control_vel(action)
                rp.sleep(0.1)

            if terminal and liveflag == True:
                liveflag = False
            
            state = next_state

            terminal_list = comm.gather(liveflag, root=0)
            terminal_list = comm.bcast(terminal_list, root=0)

            if True not in terminal_list:
                next_episode = True
                result_list = comm.gather([result, step], root=0)
                
            if env.index == 0 and next_episode:
                for r in result_list:
                    if r[0] == "Reach Goal":
                        c_suceess += 1
                    elif r[0] == "Crashed":
                        c_crash += 1
                cnt += args.num_agent
                print("Success rate: %.3f, Crash rate: %.3f, Count:%04d"
                %(c_suceess / cnt, c_crash / cnt, cnt))
            


        
            
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

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["feature_dim"] = args.feature_dim
        kwargs["add_attention"] = args.add_attention
        kwargs["policy_freq"] = args.policy_freq
        kwargs["policy_noise"] = args.policy_noise
        kwargs["noise_clip"] = args.noise_clip
        kwargs["expl_noise"] = args.expl_noise
        kwargs["start_update_step"] = args.start_update_step
        policy = TD3(env, **kwargs)
    elif args.policy == "TD3_Ae":
        kwargs["hidden_dim"] = args.hidden_dim
        kwargs["encoder_type"] = args.encoder_type
        kwargs["decoder_type"] = args.decoder_type
        kwargs["policy_freq"] = args.policy_freq
        kwargs["policy_noise"] = args.policy_noise
        kwargs["noise_clip"] = args.noise_clip
        kwargs["expl_noise"] = args.expl_noise
        kwargs["start_update_step"] = args.start_update_step
        policy = TD3_Ae(env, **kwargs)
    elif args.policy == "SAC_Ae":
        kwargs["hidden_dim"] = args.hidden_dim
        kwargs["encoder_type"] = args.encoder_type
        kwargs["decoder_type"] = args.decoder_type
        kwargs["init_steps"] = args.init_steps
        policy =SAC_Ae(env, **kwargs)
    elif args.policy == "SAC_Ae_Caps":
        kwargs["hidden_dim"] = args.hidden_dim
        kwargs["encoder_type"] = args.encoder_type
        kwargs["decoder_type"] = args.decoder_type
        kwargs["init_steps"] = args.init_steps
        kwargs["lam_a"] = args.lam_a
        kwargs["lam_s"] = args.lam_s
        kwargs["eps_s"] = args.eps_s
        policy =SAC_Ae_Caps(env, **kwargs)

    starting_epoch = 0

    if rank == 0:
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
        run(comm=comm, env=env, policy=policy, starting_epoch=starting_epoch)
    except KeyboardInterrupt:
        pass