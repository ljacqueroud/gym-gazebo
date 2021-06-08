# My env

### 1. Setup

- install all packages listed in [gym_gazebo](https://github.com/ljacqueroud/gym-gazebo/blob/master/INSTALL.md) and [xplore](https://github.com/EPFLXplore/main_NAV_ws) (or check next section)
- clone [catkin workspace repo](https://github.com/ljacqueroud/rover_catkin_ws)
- `catkin init ; catkin build`
- add `source catkin_ws/devel/setup.bash` to `.bashrc`
- clone this repo
- `cd gym_gazebo`
- `sudo pip install -e .`
**WARNING** when using `catkin_make` make sure other workspaces are not sourced!

  
### 2. Build packages

Here is a list of all packages to install:
```
sudo apt install \
ros-melodic-desktop-full \
python-pip \
python3-vcstool python3-pyqt4 pyqt5-dev-tools \
libbluetooth-dev libspnav-dev pyqt4-dev-tools libcwiid-dev \
cmake gcc g++ qt4-qmake libqt4-dev libusb-dev libftdi-dev \
python-defusedxml \
ros-melodic-octomap-msgs ros-melodic-joy ros-melodic-geodesy ros-melodic-octomap-ros         \
ros-melodic-control-toolbox ros-melodic-pluginlib ros-melodic-trajectory-msgs ros-melodic-control-msgs	       \
ros-melodic-std-srvs ros-melodic-nodelet ros-melodic-urdf ros-melodic-rviz		       \
ros-melodic-kdl-conversions ros-melodic-eigen-conversions ros-melodic-tf2-sensor-msgs     \
ros-melodic-pcl-ros ros-melodic-navigation ros-melodic-sophus \
ros-melodic-ros-control ros-melodic-ros-controllers ros-melodic-roslint \
ros-melodic-robot-state-publisher ros-melodic-pcl-ros ros-melodic-tf-conversions \
python-catkin-tools \
ros-melodic-joint-state-publisher-gui \
libeigen3-dev \
ros-melodic-grid-map ros-melodic-tf2-sensor-msgs \
ros-melodic-navigation ros-melodic-ar-track-alvar
```

and with pip
```
sudo python -m pip install gym
```


###  3. Helpful modifications

##### 3.1 GUI

Open GUI automatically when launching:\
in launch file `gym-gazebo/gym-gazebo/envs/assets/launch`
set `<arg name="gui" default="true">`

##### 3.2 Create new gym environment

- in main script: call `env = gym.make('MyNewEnvironment-v0')`
- in environment definition (`gym-gazebo/gym-gazebo/envs`): create new environment
- in `gym-gazebo/gym-gazebo/__init__.py` register new environment with id (name) and entry point (env class defined in last point)
- in launch files definition (`gym-gazebo/gym-gazebo/envs/assets/launch`): create new launch file


##### 3.3 Change world

World file in `gym-gazebo/gym-gazebo/envs/assets/worlds`\
In corresponding launch file (`gym-gazebo/gym-gazebo/envs/assets/launch`) change
`<arg name="world_file"  default="$(env MY_WORLD_PATH)"/>`\
In world file, link to corresponding mesh (dae file): `<mesh><uri>file:///path_to_my_world</uri></mesh>`


##### 3.4 Change robot model (URDF)

Urdf file in `catkin_ws/src/rover/rover_description/urdf`\
or `gym-gazebo/gym_gazebo/envs/assets/urdf`\
In corresponding launch file (`gym-gazebo/gym-gazebo/envs/assets/launch`) change `<arg name="urdf_file"  default="$(env MY_URDF_PATH)"/>`\


##### 3.5 Access internal sensor data

- IMU topic: `os1_cloud_node/imu/...`
- Joint angles: `rover/joint_states` (for each join: position, velocity, effort) or `gazebo/link_states` (for each link: position (x,y,z), orientation (x,y,z,w) linear (x,y,z), angular (x,y,z)


##### 3.6 Change verbose level

In `$ROS_ROOT/config/rosconsole.config` set verbose to `DEBUG`,`INFO`,`WARN`,`ERROR`,`FATAL`


### 4. Helpful Paths

- examples:\
  `gym-gazebo/examples`
- environments:\
  `gym-gazebo/gym-gazebo/envs`
- worlds:\
  `gym-gazebo/gym-gazebo/envs/assets/worlds`
- robots (turtlebot):\
  `catkin_ws/src/turtlebot/turtlebot_description/robots`
- bash setups:\
  `gym-gazebo/gym-gazebo/envs/installation`
- launch files:\
  `gym-gazebo/gym-gazebo/envs/assets/launch`


- urdf launch:\
  `catkin_ws/src/turtlebot_simulator/turtlebot_gazebo/launch/includes`
- world launch:\
  `catkin_ws/src/gazebo_ros_pkgs/gazebo_ros/launch`
___

___

___

<img src="data/logo.jpg" width=25% align="right" /> [![Build status](https://travis-ci.org/erlerobot/gym-gazebo.svg?branch=master)](https://travis-ci.org/erlerobot/gym-gazebo)


**THIS REPOSITORY IS DEPRECATED, REFER TO https://github.com/AcutronicRobotics/gym-gazebo2 FOR THE NEW VERSION.**

# An OpenAI gym extension for using Gazebo known as `gym-gazebo`

<!--[![alt tag](https://travis-ci.org/erlerobot/gym.svg?branch=master)](https://travis-ci.org/erlerobot/gym)-->

This work presents an extension of the initial OpenAI gym for robotics using ROS and Gazebo. A whitepaper about this work is available at https://arxiv.org/abs/1608.05742. Please use the following BibTex entry to cite our work:

```
@article{zamora2016extending,
  title={Extending the OpenAI Gym for robotics: a toolkit for reinforcement learning using ROS and Gazebo},
  author={Zamora, Iker and Lopez, Nestor Gonzalez and Vilches, Victor Mayoral and Cordero, Alejandro Hernandez},
  journal={arXiv preprint arXiv:1608.05742},
  year={2016}
}
```

-----

**`gym-gazebo` is a complex piece of software for roboticists that puts together simulation tools, robot middlewares (ROS, ROS 2), machine learning and reinforcement learning techniques. All together to create an environment whereto benchmark and develop behaviors with robots. Setting up `gym-gazebo` appropriately requires relevant familiarity with these tools.**

**Code is available "as it is" and currently it's not supported by any specific organization. Community support is available [here](https://github.com/erlerobot/gym-gazebo/issues). Pull requests and contributions are welcomed.**

-----

## Table of Contents
- [Environments](#community-maintained-environments)
- [Installation](#installation)
- [Usage](#usage)


## Community-maintained environments
The following are some of the gazebo environments maintained by the community using `gym-gazebo`. If you'd like to contribute and maintain an additional environment, submit a Pull Request with the corresponding addition.

| Name | Middleware | Description | Observation Space | Action Space | Reward range |
| ---- | ------ | ----------- | ----- | --------- | -------- |
| ![GazeboCircuit2TurtlebotLidar-v0](imgs/GazeboCircuit2TurtlebotLidar-v0.png)`GazeboCircuit2TurtlebotLidar-v0` | ROS | A simple circuit with straight tracks and 90 degree turns. Highly discretized LIDAR readings are used to train the Turtlebot. Scripts implementing **Q-learning** and **Sarsa** can be found in the _examples_ folder. | | | |
| ![GazeboCircuitTurtlebotLidar-v0](imgs/GazeboCircuitTurtlebotLidar-v0.png)`GazeboCircuitTurtlebotLidar-v0.png` | ROS | A more complex maze  with high contrast colors between the floor and the walls. Lidar is used as an input to train the robot for its navigation in the environment. | | | TBD |
| `GazeboMazeErleRoverLidar-v0` | ROS, [APM](https://github.com/erlerobot/ardupilot) | **Deprecated** | | | |
| `GazeboErleCopterHover-v0` | ROS, [APM](https://github.com/erlerobot/ardupilot) | **Deprecated** | | | |

## Other environments (no support provided for these environments)

The following table compiles a number of other environments that **do not have
community support**.

| Name | Middleware | Description | Observation Space | Action Space | Reward range |
| ---- | ------ | ----------- | ----- | --------- | -------- |
| ![cartpole-v0.png](imgs/cartpole.jpg)`GazeboCartPole-v0` | ROS | | Discrete(4,) | Discrete(2,) | 1) Pole Angle is more than ±12° 2)Cart Position is more than ±2.4 (center of the cart reaches the edge of the display) 3) Episode length is greater than 200 |
| ![GazeboModularArticulatedArm4DOF-v1.png](imgs/GazeboModularArticulatedArm4DOF-v1.jpg)`GazeboModularArticulatedArm4DOF-v1` | ROS | This environment present a modular articulated arm robot with a two finger gripper at its end pointing towards the workspace of the robot.| Box(10,) | Box(3,) | (-1, 1) [`if rmse<5 mm 1 - rmse else reward=-rmse`]|
| ![GazeboModularScara4DOF-v3.png](imgs/GazeboModularScara4DOF-v3.png)`GazeboModularScara4DOF-v3` | ROS | This environment present a modular SCARA robot with a range finder at its end pointing towards the workspace of the robot. The goal of this environment is defined to reach the center of the "O" from the "H-ROS" logo within the workspace. This environment compared to `GazeboModularScara3DOF-v2` is not pausing the Gazebo simulation and is tested in algorithms that solve continuous action space (PPO1 and ACKTR from baselines).This environment uses `slowness=1` and matches the delay between actions/observations to this value (slowness). In other words, actions are taken at "1/slowness" rate.| Box(10,) | Box(3,) | (-1, 1) [`if rmse<5 mm 1 - rmse else reward=-rmse`]|
| ![GazeboModularScara3DOF-v3.png](imgs/GazeboModularScara3DOF-v3.png)`GazeboModularScara3DOF-v3` | ROS | This environment present a modular SCARA robot with a range finder at its end pointing towards the workspace of the robot. The goal of this environment is defined to reach the center of the "O" from the "H-ROS" logo within the workspace. This environment compared to `GazeboModularScara3DOF-v2` is not pausing the Gazebo simulation and is tested in algorithms that solve continuous action space (PPO1 and ACKTR from baselines).This environment uses `slowness=1` and matches the delay between actions/observations to this value (slowness). In other words, actions are taken at "1/slowness" rate.| Box(9,) | Box(3,) | (-1, 1) [`if rmse<5 mm 1 - rmse else reward=-rmse`]|
| ![GazeboModularScara3DOF-v2.png](imgs/GazeboModularScara3DOF-v2.png)`GazeboModularScara3DOF-v2` | ROS | This environment present a modular SCARA robot with a range finder at its end pointing towards the workspace of the robot. The goal of this environment is defined to reach the center of the "O" from the "H-ROS" logo within the workspace. Reset function is implemented in a way that gives the robot 1 second to reach the "initial position".| Box(9,) | Box(3,) | (0, 1) [1 - rmse] |
| ![GazeboModularScara3DOF-v1.png](imgs/GazeboModularScara3DOF-v1.png)`GazeboModularScara3DOF-v1` | ROS | **Deprecated** | | | TBD |
| ![GazeboModularScara3DOF-v0.png](imgs/GazeboModularScara3DOF-v0.png)`GazeboModularScara3DOF-v0` | ROS | **Deprecated** | | | | TBD |
| ![ariac_pick.jpg](imgs/ariac_pick.jpg)`ARIACPick-v0` | ROS | | | |  |

## Installation
Refer to [INSTALL.md](INSTALL.md)

## Usage

### Build and install gym-gazebo

In the root directory of the repository:

```bash
sudo pip install -e .
```

### Running an environment

- Load the environment variables corresponding to the robot you want to launch. E.g. to load the Turtlebot:

```bash
cd gym_gazebo/envs/installation
bash turtlebot_setup.bash
```

Note: all the setup scripts are available in `gym_gazebo/envs/installation`

- Run any of the examples available in `examples/`. E.g.:

```bash
cd examples/turtlebot
python circuit2_turtlebot_lidar_qlearn.py
```

### Display the simulation

To see what's going on in Gazebo during a simulation, run gazebo client. In order to launch the `gzclient` and be able to connect it to the running `gzserver`:
1. Open a new terminal.
2. Source the corresponding setup script, which will update the _GAZEBO_MODEL_PATH_ variable: e.g. `source setup_turtlebot.bash`
3. Export the _GAZEBO_MASTER_URI_, provided by the [gazebo_env](https://github.com/erlerobot/gym-gazebo/blob/7c63c16532f0d8b9acf73663ba7a53f248021453/gym_gazebo/envs/gazebo_env.py#L33). You will see that variable printed at the beginning of every script execution. e.g. `export GAZEBO_MASTER_URI=http://localhost:13853`

**Note**: This instructions are needed now since `gazebo_env` creates a random port for the GAZEBO_MASTER_URI, which allows to run multiple instances of the simulation at the same time. You can remove the following two lines from the environment if you are not planning to launch multiple instances:

```bash
os.environ["ROS_MASTER_URI"] = "http://localhost:"+self.port
os.environ["GAZEBO_MASTER_URI"] = "http://localhost:"+self.port_gazebo
```

Finally, launch gzclient.
```bash
gzclient

```

### Display reward plot

Display a graph showing the current reward history by running the following script:

```bash
cd examples/utilities
python display_plot.py
```

HINT: use `--help` flag for more options.

### Killing background processes

Sometimes, after ending or killing the simulation `gzserver` and `rosmaster` stay on the background, make sure you end them before starting new tests.

We recommend creating an alias to kill those processes.

```bash
echo "alias killgazebogym='killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient'" >> ~/.bashrc
```
