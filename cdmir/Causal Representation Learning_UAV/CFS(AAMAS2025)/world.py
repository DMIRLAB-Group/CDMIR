import tf
import copy
import time
import airsim
import numpy as np
import rospy as rp
from utils import global2body, to_xy_3

from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, Range
from airsim_ros_pkgs.msg import VelCmd, GPSYaw

import random
import torch

class Environment():
    def __init__(self, index, max_timesteps):
        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient("127.0.0.1", port=41453)
        self.client.confirmConnection()
        self.index = index
        self.max_timesteps = max_timesteps
        node_name = 'CWorldEnv_' + str(index)
        rp.init_node(node_name, anonymous=None)
        self.rate = rp.Rate(20)
        self.reset_flag = False
        self.hover_flag = False
        self.plot_last_pos = []
        # self.lobs = 0
        self.pre_img = None
        self.empty_array = 0

        # -----------Publisher and Subscriber-------------
        cmd_vel_topic = 'airsim_node3/Drone_' + str(index) + '/vel_cmd_body_frame'  #用于发布无人机速度控制指令的ROS话题（Topic）名称。该话题的名称包括无人机的索引（index）以及控制命令的类型（vel_cmd_body_frame）
        self.cmd_vel = rp.Publisher(cmd_vel_topic, VelCmd, queue_size=10)   #ROS发布者对象，它使用指定的话题名称（cmd_vel_topic）来发布类型为 VelCmd 的消息。消息会被发布到该话题，并可以被其他ROS节点订阅

        dist_topic = '/airsim_node3/Drone_'+ str(index) + '/distance/Distance_' + str(index)    #用于订阅距离数据的ROS话题名称。该话题的名称包括无人机的索引（index）和距离数据的类型（Distance_）
        self.dist_sub = rp.Subscriber(dist_topic, Range, self.dist_callback)    #创建了一个 ROS 订阅者（Subscriber）对象，并将其与指定的 ROS 话题（Topic）以及消息类型（Range）以及回调函数（self.dist_callback）关联起来

        o_gps_topic = '/airsim_node3/origin_geo_point'  #用于订阅原点地理位置信息的ROS话题名称
        self.o_gps_sub = rp.Subscriber(o_gps_topic, GPSYaw, self.o_gps_callback)

        m_gps_topic = '/airsim_node3/Drone_' + str(index) +'/gps/gps'   #用于订阅无人机GPS位置信息的ROS话题名称
        self.m_gps_sub = rp.Subscriber(m_gps_topic, NavSatFix, self.m_gps_callback)

        self.speed = None
        self.o_lat = None
        self.m_lat = None

        while  self.o_lat is None or self.m_lat is None:
            pass


    def reset_world(self):  #在模拟器中重置世界
        if self.index == 0:
            self.client.reset()

    def drones_init(self):  #初始化无人机，启用API控制和解锁（arm）无人机
        self.client.enableApiControl(True, "Drone_" + str(self.index))
        self.client.armDisarm(True, "Drone_" + str(self.index))

    def start_simPause(self):
        self.client.simPause(True)

    def finish_simPause(self):
        self.client.simPause(False)

    def drones_terminal(self):  #终止无人机，悬停或复位无人机
        if self.hover_flag:
            self.hover_flag = False #表示停止悬停无人机
            self.client.hoverAsync(vehicle_name="Drone_" + str(self.index)) #无人机悬停
        if self.reset_flag: #于在实验或任务结束时安全地将无人机控制状态归零。
            self.reset_flag = False #表示停止需要重置无人机
            Pose = airsim.Pose()    #创建了一个 airsim.Pose 对象，用于表示无人机的姿态和位置信息。
            pose = [0, self.index, 0]   #创建了一个包含三个元素的列表 pose，表示无人机的位置。这个位置的具体值似乎是从 self.index 属性中获取的
            [Pose.position.x_val, Pose.position.y_val, Pose.position.z_val] = pose  #将 pose 列表中的值分别赋给 Pose 对象的 position 属性，从而设置了无人机的位置。
            qtn = tf.transformations.quaternion_from_euler(0, 0, 0, 'rxyz') #使用 tf.transformations 库将欧拉角转换为四元数 qtn，表示无人机的姿态。
            [Pose.orientation.x_val, Pose.orientation.y_val, Pose.orientation.z_val, Pose.orientation.w_val] = qtn  #将计算得到的四元数 qtn 的值分别赋给 Pose 对象的 orientation 属性，从而设置了无人机的姿态
            self.client.simSetVehiclePose(vehicle_name="Drone_" + str(self.index),ignore_collision=True, pose=Pose) #过调用 self.client 客户端对象的 simSetVehiclePose 方法来设置无人机的姿态和位置。这相当于将无人机复位到指定的位置和姿态。
            rp.sleep(0.1)
            self.client.enableApiControl(False, "Drone_" + str(self.index))
            self.client.armDisarm(False, "Drone_" + str(self.index))

    
    def reset_pose(self, init_pos, target_pos): #将无人机的姿态重置为指定的姿态，通常在任务开始时使用
        assert len(target_pos) == 3 or len(target_pos) == 4 #检查 target_pos 的长度是否为3或4。如果不满足这个条件，将引发 AssertionError 异常。这是一种用于确保输入数据有效性的方式
        Pose = airsim.Pose()
        pose = list(np.array(target_pos[:3]) - np.array(init_pos))  #计算出无人机从 init_pos 到 target_pos 的位移，将其存储在 pose 变量中。这个位移是一个三维向量，表示无人机需要移动的距离。
        [Pose.position.x_val, Pose.position.y_val, Pose.position.z_val] = pose  #将计算得到的位移分别赋给 Pose 对象的 position 属性，从而设置了无人机的位置。这相当于将无人机的位置设置为 init_pos 和 target_pos 之间的相对位移
        Pose.position.z_val = -target_pos[2] #将无人机的高度位置设置为目标高度的负值
        if len(target_pos) == 3:    #确定目标位置 target_pos 是否包含了旋转信息。如果长度为 3，表示目标位置没有旋转信息，只包含 x、y 和 z 坐标。如果长度不等于 3，那么就假定目标位置包含了旋转信息
            qtn = tf.transformations.quaternion_from_euler(0, 0, np.random.uniform(0, 2 * np.pi), 'rxyz')   # 0, 0：这是旋转的 roll 和 pitch 角度,yaw 角度通过 np.random.uniform 从 0 到 2π（360 度）之间的随机值生成
        else:   #如果 target_pos 的长度不等于 3,表示目标位置包含了旋转信息
            qtn = tf.transformations.quaternion_from_euler(0, 0, target_pos[3], 'rxyz') #四元数qtn可以用于将无人机的方向设置为特定的航向角度，以确保它朝向所需的方向
        [Pose.orientation.x_val, Pose.orientation.y_val, Pose.orientation.z_val, Pose.orientation.w_val] = qtn  #将计算得到的四元数 qtn 的值分别赋给 Pose 对象的 orientation 属性，从而设置了无人机的姿态
        self.client.simSetVehiclePose(vehicle_name="Drone_" + str(self.index), ignore_collision=True, pose=Pose)    #将无人机的位置和姿态设置为 Pose 对象中指定的位置和姿态
        rp.sleep(1)
        self.client.moveToZAsync(-target_pos[2], velocity=2, vehicle_name="Drone_" + str(self.index)).join()    #将无人机移动到指定的高度。它传递了目标高度 -target_pos[2] 以及速度参数。join() 方法用于等待无人机到达目标高度后再继续执行后续操作
        rp.sleep(0.5)
    
    def reset_barrier_pose(self, barrier_list, num_barrier):    #在模拟器中重置障碍物的位置
        if self.index == 0:
            if num_barrier > 0:
                for i in range(num_barrier):
                    pose = list(barrier_list[i])    #从 barrier_list 中获取第 i 个障碍物的位置信息，并将其转换为一个包含 x、y 和 z 坐标的列表 pose
                    Pose = airsim.Pose()    #创建了一个空的 AirSim 姿态（Pose）对象，该对象用于表示物体的位置和方向
                    [Pose.position.x_val, Pose.position.y_val, Pose.position.z_val] = pose  #将之前获取的 pose 列表中的 x、y 和 z 坐标分别分配给 Pose 对象的 position 属性，从而设置了障碍物的新位置
                    self.client.simSetObjectPose('Obstacle_' + str(i), Pose)    #使用 AirSim 客户端的 simSetObjectPose 方法来将名为 'Obstacle_' + str(i) 的障碍物设置到新的位置
                    rp.sleep(0.01)
            else:
                pass
        else:
            pass

    #处理ROS订阅主题的回调函数，用于获取GPS信息和距离数据
    def o_gps_callback(self, o_lla_msg):    #用于处理来自原点地理位置的信息
        self.o_lat, self.o_lon, self.o_alt = o_lla_msg.latitude, o_lla_msg.longitude, o_lla_msg.altitude    #从传入的 o_lla_msg 中提取了三个关键信息：latitude（纬度）、longitude（经度）和 altitude（海拔高度），然后将它们分别赋值给了类成员变量 self.o_lat、self.o_lon 和 self.o_alt

    def m_gps_callback(self, m_lla_msg):    #处理来自无人机 GPS 传感器的信息
        self.m_lat, self.m_lon, self.m_alt = m_lla_msg.latitude, m_lla_msg.longitude, m_lla_msg.altitude

    def dist_callback(self, range_msg): #处理来自距离传感器的信息
        self.range = range_msg.range    #将距离传感器测得的距离信息提取出来并保存在类的实例变量 self.range 中

    def get_position(self): #获取无人机的位置信息，并将其转换为XY坐标
        o_lat, o_lon, o_alt = self.o_lat, self.o_lon, self.o_alt    #将类实例中存储的原点（o_lat、o_lon 和 o_alt）的经纬度和海拔高度分别赋值给局部变量 o_lat、o_lon 和 o_alt。
        m_lat, m_lon, m_alt = self.m_lat, self.m_lon, self.m_alt
        x, y, z = to_xy_3(m_lat, m_lon, m_alt, o_lat, o_lon, o_alt) #传入了原点和无人机的经纬度信息，然后计算出了无人机相对于原点的XY坐标和海拔高度，并将其分别赋值给 x、y 和 z。
        return x, y, z
    
    def get_local_goal_and_speed(self): #获取局部目标和速度信息，涉及到无人机的姿态。

        state = self.client.getMultirotorState(vehicle_name="Drone_" + str(self.index)) #获取无人机的当前状态信息
        Quaternious = state.kinematics_estimated.orientation    #从无人机状态信息中提取了姿态信息。无人机的姿态通常使用四元数（quaternion）来表示，这些四元数存储在 kinematics_estimated.orientation 中。
        [roll, pitch, yaw] = tf.transformations.euler_from_quaternion([Quaternious.x_val, Quaternious.y_val, Quaternious.z_val, Quaternious.w_val])#将四元数转换为欧拉角，分别获取了滚转角（roll）、俯仰角（pitch）和偏航角（yaw）
        GT_goal = np.asarray(self.goal) #将类中存储的 self.goal 转换为 NumPy 数组，表示全局目标位置。self.goal 通常包含无人机要达到的全局目标的坐标信息。
        pos = np.asarray(self.get_position())   #获取无人机的当前位置，并将其转换为 NumPy 数组
        local_goal = global2body(roll, pitch, yaw, GT_goal, pos)    #这行代码调用了一个函数 global2body，该函数用来将全局坐标系中的目标位置 GT_goal 转换为相对于无人机当前位置 pos 的局部目标位置 local_goal。转换中考虑了无人机的姿态（滚转、俯仰和偏航角）。
        v_xyz = np.array([state.kinematics_estimated.linear_velocity.x_val, state.kinematics_estimated.linear_velocity.y_val, state.kinematics_estimated.linear_velocity.z_val])#获取了无人机的线速度（velocity）信息，包括在局部坐标系中的 x、y 和 z 方向的速度分量，并将其转换为 NumPy 数组 v_xyz。
        v_xyz = global2body(roll, pitch, yaw, v_xyz, np.array([0, 0, 0]))   #将全局坐标系中的线速度矢量 v_xyz 转换为无人机当前局部坐标系中的线速度
        vx, vz = v_xyz[0], v_xyz[2] #获取无人机在局部 x 轴、Z轴方向上的速度，
        vw = state.kinematics_estimated.angular_velocity.z_val  #获取了无人机绕垂直轴（z轴）的角速度
        local_speed = np.asarray([vx, vz, vw])  #局部速度信息 vx、vz 和 vw 存储在 NumPy 数组 local_speed 中
        return local_goal, local_speed
    
    
    def get_crash_state(self):  #检查无人机是否发生碰撞
        crash_info = self.client.simGetCollisionInfo(vehicle_name="Drone_" + str(self.index))   #获取与指定无人机相关的碰撞信息。碰撞信息包括了是否发生碰撞以及碰撞的详细信息
        return crash_info.has_collided  #在获取了碰撞信息后，此行代码返回了一个布尔值，表示是否发生了碰撞。具体地，它检查 crash_info 对象的 has_collided 属性，如果为 True，则表示发生了碰撞；如果为 False，则表示未发生碰撞
    
    def get_range(self):    #获取无人机的测距信息
        return self.range 

    def get_image(self, noise_std=0):   #获取无人机的深度图像信息，可能包括噪声


        responses = self.client.simGetImages([airsim.ImageRequest("camera_" + str(self.index), airsim.ImageType.DepthPerspective, True, False)], vehicle_name="Drone_" + str(self.index))   #获取无人机的深度图像信息

        response = responses[0] #获取了响应列表中的第一个响应，即深度图像的响应

        # Reshape to a 2d array with correct width and height
        depth_img_in_meters = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)#图像数据（response.image_data_float）重新排列成二维数组，以正确的宽度和高度表示深度图像。深度图像以浮点数表示，每个像素的值代表距离（通常以米为单位）
        size = depth_img_in_meters.shape    #获取深度图像数组的形状（尺寸）
        noise = np.random.normal(scale=noise_std,size=size) #生成一个具有与深度图像相同形状的随机噪声数组，该噪声数组将被添加到深度图像中，以模拟传感器数据的噪声
        img = depth_img_in_meters + noise
        img = np.clip(img, a_min=0.2 ,a_max=20) #裁剪深度图像的像素值，确保图像中的值位于指定范围 [0.2, 20] 内。这是因为深度图像通常包含从相机到物体的距离，所以将不合理的距离值限制在合理范围内

        if img.size != 0:   #如果img是空数组，取上一时刻img值；否则保留当前时刻img
            self.pre_img = np.copy(img)
        else:
            img = np.copy(self.pre_img)
            self.empty_array += 1
            print('The number of zero size array is :', self.empty_array)

        return img 

    def generate_goal_point(self, goal):    #生成目标点，并计算到目标点的距离
        self.goal = goal    #将传递的目标点坐标存储在类的成员变量 self.goal 中，以便后续使用
        local_goal, _ = self.get_local_goal_and_speed()
        self.distance = np.sqrt((local_goal[0]) ** 2 + (local_goal[1]) ** 2 + (local_goal[2]) ** 2) #计算了局部坐标差值的欧氏距离，即无人机当前位置到目标点的距离。这个距离被存储在类的成员变量 self.distance 中
        return self.distance    #距离值通常用于衡量无人机是否已经接近或到达了目标点
    
    def get_reward_and_terminate(self, t, img): #计算奖励和终止条件，例如是否到达目标或发生碰撞。
        d_safe = 5  #表示无人机与障碍物之间的安全距离
        # weight
        w_g = 3 #到达目标的奖励权重
        w_c = -0.05 #表示碰撞惩罚权重

        terminate = False
        local_goal, local_speed= self.get_local_goal_and_speed()

        d_obs = np.min(img)  # 计算了深度图像信息中的最小值，即离无人机最近的障碍物的距离
        self.pre_distance = copy.deepcopy(self.distance)    #存储上一个时间步的距离值，以便在计算奖励或执行其他操作时能够与当前时间步的距离值进行比较，以确定无人机是否靠近或远离目标点，从而影响奖励的计算
        self.distance = np.sqrt(local_goal[0] ** 2 + local_goal[1] ** 2 + local_goal[2] ** 2)   #计算无人机当前位置与目标点之间的距离
        r_g = (self.pre_distance - self.distance) * w_g #计算了基于距离变化的奖励，即无人机是否靠近目标。奖励值等于距离变化的负数乘以权重 w_g
        r_c = w_c * max(d_safe - d_obs, 0)  #计算了基于碰撞的惩罚，如果无人机离障碍物太近（小于 d_safe），则会受到惩罚。奖励值等于 d_safe - d_obs 的最大值乘以权重 w_c。
        result = 0
        is_crash = self.get_crash_state()   #检查无人机是否发生了碰撞
        
        if t > self.max_timesteps:  #如果时间超过了最大时间步数 self.max_timesteps，则任务终止
            terminate = True
            result = 'Time out'

        if self.distance < 0.5: #如果无人机距离目标点小于0.5（可以根据具体任务调整），则任务终止
            terminate = True
            r = 50
            result = 'Reach Goal'
            self.hover_flag = True
            return r, terminate, result

        if is_crash is True or not(np.any(local_speed)):    #如果发生了碰撞或者速度为0，则任务终止
            terminate = True
            r = -10
            result = 'Crashed'
            self.reset_flag = True
            return r, terminate, result

        # if self.index == 0:
        #     print(r_g, r_c)
        r = r_g + r_c 

        return r, terminate, result

    def control_vel(self, action):  #控制无人机的速度
        [v, v_z, v_w] = action  #将 action 向量解包为三个变量 v、v_z 和 v_w
        v = float(v) * 1.5  #将线性速度 v 从字符串或其他数据类型转换为浮点数，并乘以1.5，以调整速度的幅度
        v_w = float(v_w) * 0.5
        v_z = float(v_z) * 0.5
        vel_cmd = VelCmd()  #创建了一个名为 vel_cmd 的 VelCmd 类的实例，用于存储速度控制命令
        vel_cmd.twist.linear.x = v  #分别设置 vel_cmd 实例的线性速度的 X 和 Z 分量，以及角速度的 Z 分量，从而构建了速度控制命令
        vel_cmd.twist.linear.y = 0
        vel_cmd.twist.linear.z = v_z
        vel_cmd.twist.angular.x = 0
        vel_cmd.twist.angular.y = 0
        vel_cmd.twist.angular.z = v_w
        tik = time.time()   #记录了当前时间
        while time.time() - tik < 0.2:  #这是一个循环，它会一直运行，直到时间间隔超过0.2秒。
            self.cmd_vel.publish(vel_cmd)   #将速度控制命令发送给无人机，以实际控制其运动
            self.rate.sleep()   #通过 rate 对象调用 sleep 方法，以确保循环以合适的速度执行，这有助于控制速度命令的发送频率。

        
    def plot_trajecy(self, init_pose, path, color_rgba, render_plot):   #绘制无人机的轨迹
            [x_last, y_last, z_last] = init_pose
            x_now, y_now, z_now = self.get_position()
            path += np.sqrt((x_now-x_last) ** 2 + (y_now-y_last) ** 2 + (z_now-z_last) ** 2)
            plot_v_start = [airsim.Vector3r(x_now , y_now, -z_now)]
            if render_plot:
                self.client.simPlotLineList(self.plot_last_pos + plot_v_start, color_rgba=color_rgba[self.index], thickness=4, is_persistent=True)
            self.plot_last_pos = plot_v_start
            return [x_now, y_now, z_now], path
