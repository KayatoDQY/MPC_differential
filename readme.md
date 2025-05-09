# 差速车MPC控制

## 测试环境

Ubuntu 20.04 

[ROS Noetic Ninjemys](https://wiki.ros.org/noetic/Installation)

**[osqp](https://github.com/osqp/osqp)**

```sh
git clone --recursive https://github.com/osqp/osqp
cd osqp
mkdir build
cd build
cmake ..
sudo make install
```

[osqp-eigen](https://github.com/robotology/osqp-eigen)

```sh
git clone https://github.com/robotology/osqp-eigen.git
cd osqp-eigen
mkdir build
cd build
cmake ..
sudo make
sudo make install
```

OpenCV 4.5.0 

## 编译代码

```sh
mkdir catkin_ws
cd catkin_ws
nkdir src
cd src
git clone https://github.com/KayatoDQY/MPC_differential.git
cd ..
catkin build mpc_ctrl
```



## 代码说明

### V1.0

使用MPC控制差速车在仿真环境进行轨迹跟踪

启动仿真环境

```sh
roslaunch mini_sim simulation_camera.launch
```

启动mpc节点，MPC实现参见https://robotology.github.io/osqp-eigen/md_pages_mpc.html

```sh
rosrun mpc_ctrl mpc_node
```

启动轨迹发布节点，给MPC发送期望轨迹

```
rosrun mpc_ctrl trajectory_publisher.py
```

### V2.0

使用MPC控制差速车在真实环境中跟踪轨迹

使用hdl_localization在真实环境中重定位

运行节点录制轨迹

```sh
rosrun mpc_ctrl write_odom.py
```

录制完成后保存为CSV文件，回放轨迹

```
rosrun mpc_ctrl trajectory_publisher.py
```

使用MPC跟踪轨迹

使用pid调整yaw角误差过大时MPC无法正常工作

```
rosrun mpc_ctrl mpc_node
```

详细间img中视频

### V3.0/main

添加了规划模块，原理见https://github.com/hku-mars/IPC/tree/master/IPC

各代码文件作用

[grid_map.cc](https://github.com/KayatoDQY/MPC_differential/blob/main/src/mpc_ctrl/src/grid_map.cc)订阅mid360雷达数据，膨胀后转换为2维栅格地图

[Astar.cc](https://github.com/KayatoDQY/MPC_differential/blob/main/src/mpc_ctrl/src/Astar.cc)以机器人本体坐标系进行路径搜索，并将搜索后的路径简化后发布（避免A*路径频繁转向）

[EllipseUtils.hpp](https://github.com/KayatoDQY/MPC_differential/blob/main/src/mpc_ctrl/include/EllipseUtils.hpp)求解安全可行空间原理见Planning Dynamically Feasible Trajectories for Quadrotors using Safe Flight Corridors in 3-D Complex Environments

[qp_new.hpp](https://github.com/KayatoDQY/MPC_differential/blob/main/src/mpc_ctrl/include/qp_new.hpp)改进后的MPC，添加了安全可行空间作为约束

[SDC.cc](https://github.com/KayatoDQY/MPC_differential/blob/main/src/mpc_ctrl/src/SDC.cc)主函数

启动A*以及栅格地图构建

```
roslaunch mpc_ctrl ipc.launch
```

启动控制主程序

```
rosrun mpc_ctrl sdc
```

#### TODO

BUG1：输入MPC算法的期望yaw角错误，导致MPC无法正常预测轨迹

BUG2：极端情况应发送停止指令，但实际并不会停止

