import airsim
import time

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.enableApiControl(True)   # get control
client.armDisarm(True)          # unlock
client.takeoffAsync().join()    # takeoff

# square flight
client.moveToZAsync(-3, 1).join()               # 上升到3m高度
client.moveToPositionAsync(5, 0, -3, 1).join()  # 飞到（5,0）点坐标
client.moveToPositionAsync(5, 5, -3, 1).join()  # 飞到（5,5）点坐标
client.moveToPositionAsync(0, 5, -3, 1).join()  # 飞到（0,5）点坐标
client.moveToPositionAsync(0, 0, -3, 1).join()  # 回到（0,0）点坐标

client.landAsync().join()       # land
client.armDisarm(False)         # lock
client.enableApiControl(False)  # release control