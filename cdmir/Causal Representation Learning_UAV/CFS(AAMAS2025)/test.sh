source ~/catkin_airsim/devel/setup.bash
source /home/hhx/anaconda3/bin/activate uavrl
#mpiexec --use-hwthread-cpus -np 8 python test_1.py 
mpiexec -np 8 python test_1.py 
