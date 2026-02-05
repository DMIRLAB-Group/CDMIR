from utils import generate_points, Cycle_position, Statistics, seed_everything
import random
import numpy as np
seed = 0

seed_everything(seed)  # 设置随机种子

for i in range(0,50):
    print(np.random.randint(1, 101))
    print(np.random.randint(1, 101))