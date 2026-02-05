import math
import copy
import os

import torch
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from math import sqrt
from random import randrange as rd
import torch.distributed as dist

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity), columns=['O_z','O_g','O_v', 'action', 'reward', 'next_O_z','next_O_g','next_O_v', 'not_done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in
                self.memory.columns)

class Statistics:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(columns=['Extra Distance', 'Average Speed'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)
    



def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )

def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs

def to_xy_3(M_lat, M_lon, M_alt, O_lat, O_lon, O_alt):
    """[Calculating NED plane coordinates by longitude, latitude and altitute]"""
    Ea = 6378137   # Equatorial radius
    Eb = 6356725   # Polar radius
    M_lat = math.radians(M_lat)
    M_lon = math.radians(M_lon)
    O_lat = math.radians(O_lat)
    O_lon = math.radians(O_lon)
    Ec = Ea*(1-(Ea-Eb)/Ea*((math.sin(M_lat))**2)) + M_alt
    Ed = Ec * math.cos(M_lat)
    d_lat = M_lat - O_lat
    d_lon = M_lon - O_lon
    x = d_lat * Ec
    y = d_lon * Ed
    z = M_alt - O_alt
    return x, y, z

def log_normal_density(x, mean, log_std, std):
    """returns guassian density given x on log scale"""
    variance = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * variance) - 0.5 * np.log(2 * np.pi) - log_std    
    log_density = log_density.sum(dim=-1, keepdim=True) 
    return log_density

def global2body(roll, pitch, yaw, g_goal, g_drone):
    R_x = np.array([[1, 0, 0],[0, np.cos(roll), np.sin(roll)], [0, -np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, -np.sin(pitch)], [0, 1, 0], [np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    R_g2b = np.matmul(R_z, R_y)
    R_g2b = np.matmul(R_g2b, R_x)
    return np.matmul(R_g2b, (g_goal - g_drone))

def generate_points(ptBlu=[1,0], maxdist=12, num_env=20, num_barrier=5, dis=2):
    barrier_position = None

    noise = 20
    if maxdist > 12:
        min_goal = 10
        max_goal = 12
    elif maxdist == 12: #maxdist这个参数指定了红色点和蓝色点之间的最大欧几里得距离
        min_goal = 8
        max_goal = 10
    elif maxdist < 12:
        noise = 10
        min_goal = 6
        max_goal = 8

    redptlist=[] #inizialize a void lists for red point coordinates,存储红色点（机器人的起始位置）的坐标信息。通常，每个红色点的坐标会以 [x, y] 的形式添加到这个列表中
    xredptlist=[]   #存储红色点的 x 坐标信息。通常，红色点的 x 坐标会以单个值的形式添加到这个列表中
    yredptlist=[]

    blueptlist=[] #inizialize a void lists for blue point coordinates
    xblueptlist=[] 
    yblueptlist=[]

    pointcounter = 0 #initizlize counter for the while loop,初始化了一个计数器变量pointcounter，用于追踪已生成的红色点数量。
    #set the maximum euclidean distance redpoint can have from blu point
    maxc=int(sqrt((maxdist**2)/2)) #from the euclidean distance formula you can get the max coordinate,根据maxdist的值，计算了在给定距离内的最大坐标范围。这个范围会影响红色点的生成位置
    while True: #create a potentailly infinite loop! pay attention!不断生成红色点，直到满足特定的条件才会结束循环
        if pointcounter < num_env + num_barrier: #set the number of point you want to add (in this case 20)pointcounter是否小于指定的点的总数，其中num_env是要生成的红色点的数量，num_barrier是障碍物的数量。这个条件确保不会生成太多的红色点
            x_RedPtshift = rd(-maxc,maxc,dis) #x shift of a red point 这两行代码分别生成了红色点在x和y坐标上的随机偏移量。rd函数用于生成随机数，-maxc和maxc是随机数生成的范围，dis用于控制随机数生成的离散程度
            y_RedPtshift = rd(-maxc,maxc,dis) #y shift of a red point
            ptRedx = ptBlu[0]+ x_RedPtshift #x coordinate of a red point计算了生成的红色点的x和y坐标，通过将随机偏移量应用到蓝色点的坐标（ptBlu)
            ptRedy = ptBlu[1]+ y_RedPtshift #y coordinate of a red point
            ptRed = [ptRedx,ptRedy] #list with the x,y,z coordinates生成的红色点的x和y坐标组成一个包含两个元素的列表，表示红色点的完整坐标
            if ptRed not in redptlist: #avoid to create red point with the same coordinates检查生成的红色点是否已经存在于redptlist中，以避免生成重复的红色点。
                redptlist.append(ptRed) #add to a list with this notation [x1,y1],[x2,y2]如果红色点是新的（不在列表中），则将其添加到redptlist中，以记录生成的红色点
                xredptlist.append(ptRedx) #add to a list with this notation [x1,x2,x3...] for plotting将生成的红色点的x和y坐标分别添加到xredptlist和yredptlist中，以便后续的可视化或其他处理
                yredptlist.append(ptRedy) #add to a list with this notation [y1,y2,y3...] for plotting
                pointcounter += 1 #add one to the counter of how many points you have in your list 每次成功生成一个红色点后，将pointcounter递增，以追踪生成的红色点数量。
        else: #when pointcounter reach the number of points you want the while cicle ends
            pointcounter = 0    #将pointcounter重置为0，以准备生成下一组红色点
            break

    # position_x = [xredptlist[i] + int(np.random.uniform(-noise, noise)) / 100 for i in range(num_env)]
    # position_y = [yredptlist[i] + int(np.random.uniform(-noise, noise)) / 100 for i in range(num_env)]
    position_x = [xredptlist[i] for i in range(num_env)]#使用列表解析从xredptlist中提取了前num_env个元素，这些元素包含了红色点（机器人起始位置）的x坐标。这将生成一个名为position_x的列表，其中包含了机器人的x坐标
    position_y = [yredptlist[i] for i in range(num_env)]
    position_z = [int(np.random.uniform(30, 70)) / 10 for _ in range(num_env)] #生成了一组随机的z坐标，用于表示机器人的高度
    # print("\nseed:", np.random.get_state()[1][0])
    position = [list(t) for t in zip(position_x, position_y, position_z)]   #使用zip函数将前面生成的position_x、position_y和position_z三个列表中的元素按顺序配对成元组，然后通过list函数将每个元组转换为列表
    #xredptlist[-num_barrier:]：这部分代码从xredptlist列表中获取最后num_barrier个元素，这些元素包含了障碍物的x坐标
    #yredptlist[-num_barrier:]：这部分代码从yredptlist列表中获取最后num_barrier个元素，这些元素包含了障碍物的y坐标
    #[-int(np.random.uniform(30, 70)) / 10 for _ in range(num_barrier)]：这部分代码生成了一组随机的z坐标，用于表示障碍物的高度
    #zip(xredptlist[-num_barrier:], yredptlist[-num_barrier:], [-int(np.random.uniform(30, 70)) / 10 for _ in range(num_barrier)])：这部分代码使用zip函数将前面生成的x、y和z坐标组合成三元组，并且通过list()函数将每个三元组转换为列表。
    if num_barrier > 0:
        barrier_position = [list(t) for t in zip(xredptlist[-num_barrier:], yredptlist[-num_barrier:], [-int(np.random.uniform(30, 70))  / 10 for _ in range(num_barrier)])]

    while True: #create a potentailly infinite loop! pay attention!
        if pointcounter < num_env: #set the number of point you want to add (in this case 20)
            x_BluePtshift = rd(-maxc,maxc,dis) #x shift of a blue point 生成了随机的x和y坐标偏移量，用于在蓝色点的位置附近随机生成新的蓝色点
            y_BluePtshift = rd(-maxc,maxc,dis) #y shift of a blue point
            ptBluex = ptBlu[0]+ x_BluePtshift #x coordinate of a blue point
            ptBluey = ptBlu[1]+ y_BluePtshift #y coordinate of a blue point
            distance = sqrt((ptBluex-position_x[pointcounter])**2 + (ptBluey-position_y[pointcounter])**2)#计算了新生成蓝色点的位置与之前生成的机器人位置的欧几里得距离。这个距离用于检查新生成的蓝色点是否在合理的距离范围内
            if (min_goal <= distance <= max_goal):  #检查计算得到的距离是否在指定的最小和最大距离范围内。只有当距离在指定范围内时，才会继续生成新的蓝色点
                ptBlue = [ptBluex,ptBluey] #list with the x,y,z coordinates如果距离在合理范围内，那么新蓝色点的坐标将被保存在ptBlue列表中，表示新生成蓝色点的x和y坐标
            else:
                continue
            #据是否存在障碍物（num_barrier > 0），以及新生成蓝色点的坐标是否与已有点（ptBlue not in blueptlist）和障碍物点（ptBlue not in barrier_position）重叠，来决定是否将新蓝色点添加到列表中
            if num_barrier > 0:
                if ptBlue not in blueptlist and ptBlue not in barrier_position: #avoid to create blue point with the same coordinates
                    blueptlist.append(ptBlue) #add to a list with this notation [x1,y1],[x2,y2]将新生成的蓝色点坐标ptBlue添加到蓝色点列表
                    xblueptlist.append(ptBluex) #add to a list with this notation [x1,x2,x3...] for plotting将新生成蓝色点的x坐标ptBluex添加到x坐标列表xblueptlist中
                    yblueptlist.append(ptBluey) #add to a list with this notation [y1,y2,y3...] for plotting
                    pointcounter += 1 #add one to the counter of how many points you have in your list 
            else:   #检查新生成的蓝色点ptBlue是否不在蓝色点列表blueptlist中，并且不为None。这是因为在没有障碍物点的情况下，不需要检查与障碍物的重叠，只需确保不生成重复的蓝色点
                if ptBlue not in blueptlist and ptBlue: #avoid to create blue point with the same coordinates
                    blueptlist.append(ptBlue) #add to a list with this notation [x1,y1],[x2,y2]
                    xblueptlist.append(ptBluex) #add to a list with this notation [x1,x2,x3...] for plotting
                    yblueptlist.append(ptBluey) #add to a list with this notation [y1,y2,y3...] for plotting
                    pointcounter += 1 #add one to the counter of how many points you have in your list 
        else: #when pointcounter reach the number of points you want the while cicle ends
            pointcounter = 0
            break

    goal_x = [xblueptlist[i] + int(np.random.uniform(-noise, noise)) / 100 for i in range(len(xblueptlist))]#创建了一个列表goal_x，用于存储目标点的x坐标。它通过遍历蓝色点的x坐标列表xblueptlist，并对每个x坐标应用一些随机扰动。扰动的大小由int(np.random.uniform(-noise, noise)) / 100计算，其中np.random.uniform(-noise, noise)生成一个在-noise到noise之间的随机浮点数，然后取整数部分，最后除以100将其缩小
    goal_y = [yblueptlist[i] + int(np.random.uniform(-noise, noise)) / 100  for i in range(len(yblueptlist))]
    goal_z = [int(np.random.uniform(30, 70)) / 10 for _ in range(num_env)]#通过生成一个在30到70之间的随机浮点数，然后除以10来获得z坐标的值。这样做是为了在z坐标上引入一些随机变化，使目标点在垂直方向上有所变化
    goal = [list(t) for t in zip(goal_x, goal_y, goal_z)]#将生成的x、y和z坐标组合成一个三维坐标点，并将它们存储在goal列表中。zip(goal_x, goal_y, goal_z)将goal_x、goal_y和goal_z中的对应元素逐一组合成元组，然后list(t)将每个元组转换为列表，最终得到包含所有目标点坐标的goal列表。

    # plt.plot(position_x, position_y, 'bo') #plot blue points
    # plt.plot(goal_x, goal_y, 'ro') #plot blue points
    # plt.show()

    return position, goal, barrier_position

def Cycle_position(num_env, radius, num_barrier=4, ptBlu=[0, 9]):
    assert num_env % 2 == 0
    # barrier_position = None
    barrier_position = []
    position = []
    for i in range(num_env):
        point = [radius * math.cos(ptBlu[0] + 2 * math.pi * i / num_env), ptBlu[1] + radius * math.sin(2 * math.pi * i / num_env)]
        point.append(5)
        point.append(i * 2 * np.pi / num_env + np.pi)
        position.append(point)
    goal = copy.copy(position)
    #按照圆形同一高度生成****************************************************************************************************************************************************************************
    # for i in range(num_barrier):
    #     point = [0.5 * np.random.uniform(40, 140) / 10 * math.cos(ptBlu[0] + 2 * math.pi * i / num_barrier), ptBlu[1] + 0.5 * np.random.uniform(40, 140) / 10 * math.sin(2 * math.pi * i / num_barrier)]
    #     point.append(-5)
    #     # point.append(i * 2 * np.pi / num_barrier + np.pi)
    #     barrier_position.append(point)

    # 圆柱区域随机生成************************************************************************************************************************************************************************************
    # 障碍物位置集合
    # barrier_positions_set = set()
    # # 生成障碍物位置
    # while len(barrier_position) < num_barrier:
    #     # 随机生成半径和角度
    #     rand_radius = np.random.uniform(2, radius-2)
    #     rand_angle = np.random.uniform(0, 2 * np.pi)
    #     x = ptBlu[0] + rand_radius * math.cos(rand_angle)
    #     y = ptBlu[1] + rand_radius * math.sin(rand_angle)
    #
    #     point = (round(x, 6), round(y, 6))  # 使用元组并四舍五入以确保精度
    #
    #     # 检查位置是否已存在
    #     if point not in barrier_positions_set:
    #         barrier_positions_set.add(point)
    #         barrier_position.append([x, y, -np.random.uniform(3, 8)])  # 添加高度

    # 方形区域随机生成****************************************************************************************************************************************************************************
    redptlist = []  # inizialize a void lists for red point coordinates,存储红色点（机器人的起始位置）的坐标信息。通常，每个红色点的坐标会以 [x, y] 的形式添加到这个列表中
    xredptlist = []  # 存储红色点的 x 坐标信息。通常，红色点的 x 坐标会以单个值的形式添加到这个列表中
    yredptlist = []
    pointcounter = 0  # initizlize counter for the while loop,初始化了一个计数器变量pointcounter，用于追踪已生成的红色点数量。
    # set the maximum euclidean distance redpoint can have from blu point
    maxc = int(sqrt((8 ** 2) / 2))  # from the euclidean distance formula you can get the max coordinate,根据maxdist的值，计算了在给定距离内的最大坐标范围。这个范围会影响红色点的生成位置
    while True:  # create a potentailly infinite loop! pay attention!不断生成红色点，直到满足特定的条件才会结束循环
        if pointcounter < num_env + num_barrier:  # set the number of point you want to add (in this case 20)pointcounter是否小于指定的点的总数，其中num_env是要生成的红色点的数量，num_barrier是障碍物的数量。这个条件确保不会生成太多的红色点
            x_RedPtshift = rd(-maxc, maxc, 2)  # x shift of a red point 这两行代码分别生成了红色点在x和y坐标上的随机偏移量。rd函数用于生成随机数，-maxc和maxc是随机数生成的范围，dis用于控制随机数生成的离散程度
            y_RedPtshift = rd(-maxc, maxc, 2)  # y shift of a red point
            ptRedx = ptBlu[0] + x_RedPtshift  # x coordinate of a red point计算了生成的红色点的x和y坐标，通过将随机偏移量应用到蓝色点的坐标（ptBlu)
            ptRedy = ptBlu[1] + y_RedPtshift  # y coordinate of a red point
            ptRed = [ptRedx, ptRedy]  # list with the x,y,z coordinates生成的红色点的x和y坐标组成一个包含两个元素的列表，表示红色点的完整坐标
            if ptRed not in redptlist:  # avoid to create red point with the same coordinates检查生成的红色点是否已经存在于redptlist中，以避免生成重复的红色点。
                redptlist.append(ptRed)  # add to a list with this notation [x1,y1],[x2,y2]如果红色点是新的（不在列表中），则将其添加到redptlist中，以记录生成的红色点
                xredptlist.append(ptRedx)  # add to a list with this notation [x1,x2,x3...] for plotting将生成的红色点的x和y坐标分别添加到xredptlist和yredptlist中，以便后续的可视化或其他处理
                yredptlist.append(ptRedy)  # add to a list with this notation [y1,y2,y3...] for plotting
                pointcounter += 1  # add one to the counter of how many points you have in your list 每次成功生成一个红色点后，将pointcounter递增，以追踪生成的红色点数量。
        else:  # when pointcounter reach the number of points you want the while cicle ends
            pointcounter = 0  # 将pointcounter重置为0，以准备生成下一组红色点
            break
    if num_barrier > 0:
        barrier_position = [list(t) for t in zip(xredptlist[-num_barrier:], yredptlist[-num_barrier:], [-int(np.random.uniform(30, 70)) / 10 for _ in range(num_barrier)])]


    for i in range(num_env):
        if i < num_env / 2:
            goal[i] = position[int(num_env / 2 + i)][:3]
        else:
            goal[i] = position[int(i - num_env / 2)][:3]
    return position, goal, barrier_position
#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

