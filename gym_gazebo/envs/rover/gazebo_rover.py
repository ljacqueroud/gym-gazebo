import gym
import rospy
import roslaunch
import time
import numpy as np
import collections
import copy
import torch
import matplotlib.pyplot as plt

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from std_srvs.srv import Empty

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Imu
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from rosgraph_msgs.msg import Clock

from gym.utils import seeding
from .utils import *

action_commands = [
        [1,0,0,0],      # forward
        [-1,0,0,0],     # backward
        [0,0,0,1],      # left
        [0,0,0,-1],     # right
        [0,0,0,0],      # stop
    ]

joint_state_datatypes = ["none", "chassis", "wheels"]


class GazeboRoverEnv(gazebo_env.GazeboEnv):


    #################### INIT FUNCTION ########################

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboRover.launch.xml")

        # initiate ros services
        # publishers
        self.vel_pub = rospy.Publisher('cmd_vel_chosen', Twist, queue_size=5)
        self.goal_pub = rospy.Publisher('goal_point', Point, queue_size=5)
        # subscribers
        #self.vel_sub = rospy.Subscriber('cmd_vel', Twist, self.vel_sub_callback)
        self.imu_sub = rospy.Subscriber('os1_cloud_node/imu', Imu, self.imu_sub_callback)
        self.joint_sub = rospy.Subscriber('/rover/joint_states', JointState, self.joint_sub_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_sub_callback)
        self.path_sub = rospy.Subscriber('/move_base/RAstarPlannerROS/plan', Path, self.path_sub_callback)
        self.clock_sub = rospy.Subscriber('/clock', Clock, self.clock_sub_callback)


        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        # environment parameters
        self.action_space = spaces.Discrete(1) #F,L,R
        self.reward_range = (-np.inf, np.inf)

        # network parameters
        self.N_params_imu = 10                      # number of imu data
        self.N_params_joint_state = 29              # number of joint state data
        self.N_params_sensor_state = self.N_params_imu + self.N_params_joint_state 
        self.N_params_odom = 3                      # number of odom data
        self.N_params_path = 5                      # number of path poses to keep
        self.N_params_path_state = self.N_params_odom + self.N_params_path*2
        self.N_actions = len(action_commands)       # number of action commands
        self.N_steps = 30       # number of time steps to keep in memory for network input
                                # when changing N_steps, also change it in rover_main.py (main script)

        # simulation parameters
        self.lin_speed = 5
        self.turn_speed = 7
        self.sim_time_step = 0.1

        # ros parameters
        self.new_joint_state = JointState()
        self.new_joint_state_datatype = "none"

        # parameters to keep in memory
        self.imu = collections.deque(maxlen=self.N_steps)
        self.joint_state = collections.deque(maxlen=self.N_steps)

        # flags for multi threads conflicts
        self.changing_joint_state = False
        self.changing_imu = False
        self.changing_odom = False
        self.changing_path = False
        self.changing_clock = False

        # initialize all variables
        self.initiliaze_vars()

        self._seed()

    #################### GET AND SET FUNCTIONS ########################

    def get_N_params_sensor_state(self):
        return self.N_params_sensor_state

    def get_N_params_path_state(self):
        return self.N_params_path_state

    def get_N_actions(self):
        return self.N_actions

    def get_N_steps(self):
        return self.N_steps

    def get_sensor_state(self):
        """
        sensor_state [1,39]:
        [1, 0:N_params_imu] imu data
        [1, N_params_imu:] joint data
        """
        # wait until other processes are done
        while(self.changing_joint_state or self.changing_imu):
            time.sleep(0.01)
        self.changing_joint_state = True
        self.changing_imu = True

        #print("imu {}".format(self.imu_counter))
        #print("joint_state {}".format(self.joint_counter))
        #print("imu {}: {}".format(self.imu_counter,self.imu))
        #print("joint_state {}: {}".format(self.joint_counter,self.joint_state))

        # save state from imu and joint_state data
        state = torch.Tensor([]).unsqueeze(1)
        for imu, joint in zip(self.imu, self.joint_state):
            state_t = torch.cat((imu, joint), 0)    # state at time t, of size [39]
            state_t = state_t.unsqueeze(1)      # expand dims: size [39,1]
            if state.nelement()==0:
                state = state_t                 # initialize tensor state
            else:
                state = torch.cat((state, state_t), 1)  # concatenate along dimension 1: size [39,N]

        # set flags
        self.changing_joint_state = False
        self.changing_imu = False

        return state.unsqueeze(0)                       # final tensor of size [1,39, N_steps]

    def get_path_state(self):
        """
        path_state [1,N]:
        [1, 0:N_params_odom]    odom data = pos.x, pos.y, orientation.z
        [1, N_params_odom:]     path data = [point.x, point.y] repeated N_params_path
        """
        # wait until other processes are done
        while(self.changing_odom or self.changing_path):
            time.sleep(0.01)
        self.changing_odom = True
        self.changing_path = True
        
        # save position from odom
        yaw = quaternion_to_euler(self.odom.orientation, only_yaw=True)
        pos = torch.Tensor([self.odom.position.x, self.odom.position.y, yaw])

        # save path
        path_pos = []
        for pose in self.path.poses[:self.N_params_path]:  # save first N_params_path from path
            path_pos.append([pose.pose.position.x, pose.pose.position.y])
        if len(path_pos) < self.N_params_path:          # repeat last path_pos to get to N_params_path
            if not len(path_pos):                           
                path_pos.append([0,0])                  # add zero element if list is empty
            for _ in range(self.N_params_path - len(path_pos)):
                path_pos.append(path_pos[-1])
        path_pos = torch.Tensor(path_pos)

        # concatenate position and path
        path_state = torch.cat((pos.view(-1), path_pos.view(-1)), 0)        # concatenate in [N]

        print("current pos: {}".format(pos))

        # set flags
        self.changing_odom = False
        self.changing_path = False

        return path_state.unsqueeze(0)                  # final tensor of size [1,N]

    def get_done(self):
        if self.steps_done >= self.max_steps:
            done = True
        else:
            done = False
        return done

    def set_max_steps(self, max_steps):
        self.max_steps = max_steps

    #################### ENVIRONMENT FUNCTIONS ########################

    def initialize_sensor_state(self):
        """ sets imu and joint_states to zero
        """
        for _ in range(self.N_steps):
            self.imu.append(torch.zeros([self.N_params_imu]))
            self.joint_state.append(torch.zeros([self.N_params_joint_state]))

        self.imu_counter=0
        self.joint_counter=0

#    def initialize_vel_cmd(self):
#        self.path_planner_cmd = self.get_vel_command(self.N_actions-1)     # init with stop command

    def initialize_odom(self):
        self.odom = Pose()
        self.last_pos = np.zeros(self.N_params_odom)

    def initialize_path(self):
        self.path = Path()

    def initialize_clock(self):
        self.clock = Clock()

    def initiliaze_vars(self):
        self.initialize_sensor_state()
        #self.initialize_vel_cmd()
        self.initialize_odom()
        self.initialize_path()
        self.initialize_clock()
        self.generate_new_goal()
        self.path_to_follow_array = []
        self.path_followed = []
        self.steps_done = 0

    def generate_new_goal(self, goal_point=None):
        """ generate goal_point randomly (if not given) and publish
        """
        if not goal_point:
            goal_point = Point()
            goal_point.x = np.random.uniform(low=-5, high=5)
            goal_point.y = np.random.uniform(low=-5, high=5)
        print("==================== new goal ======================\n{}".format(goal_point))
        self.goal_pub.publish(goal_point)

    def get_vel_command(self, action):
        #if action > 0:      # action [1:N_actions]
        vel_cmd = Twist()
        vel_cmd.linear.x = action_commands[action][0] * self.lin_speed
        vel_cmd.linear.y = action_commands[action][1] * self.lin_speed
        vel_cmd.linear.z = action_commands[action][2] * self.lin_speed
        vel_cmd.angular.x = 0
        vel_cmd.angular.y = 0
        vel_cmd.angular.z = action_commands[action][3] * self.turn_speed
        #else:           # action 0
        #    vel_cmd = self.path_planner_cmd

        return vel_cmd

    def wait_gazebo_sim(self, wait_time=0.2):
        """ Wait for wait_time using the simulation time
        """
        start_time = self.clock.clock.secs + self.clock.clock.nsecs*(10**(-9))
        done_waiting = False
        while not done_waiting:
            while self.changing_clock:
                time.sleep(0.01)
            self.changing_clock = True
            new_time = self.clock.clock.secs + self.clock.clock.nsecs*(10**(-9))
            self.changing_clock = False
            if (new_time - start_time) >= wait_time:
                break
            time.sleep(0.05)


    def compute_reward(self, done, action, path_state):
        """
        Compute reward based on reward function
        R = 
        """
        
        # convert to numpy
        path_state = path_state.squeeze(0).numpy()
        
        # get current pos of rover
        current_pos = path_state[:3]

        # find closest path points to rover's previous position
        path_dist = np.zeros(self.N_params_path)
        for i in range(self.N_params_path):
            path_dist[i] = np.linalg.norm(path_state[self.N_params_odom + i*2 : self.N_params_odom + (i+1)*2] - self.last_pos[:2])
        closest_points_idx = path_dist.argsort()[:2]        # get closest two points
        closest_points_idx.sort()                           # sort them to match path order
        closest_points_path_idx = self.N_params_odom + closest_points_idx*2     # convert to path_state idx
        closest_points = np.array([path_state[closest_points_path_idx[0]:closest_points_path_idx[0]+2],path_state[closest_points_path_idx[1]: closest_points_path_idx[1]+2]])       # save to numpy

        # find parallel and orthogonal projections of path traveled on path
        path_traveled = current_pos[:2] - self.last_pos[:2]
        path_real = closest_points[1] - closest_points[0]
        if np.linalg.norm(path_real) < 1e-10:       # if the rover didn't move
            proj_parallel = 0
            proj_orthogonal = 0
        else:
            proj_parallel = np.dot(path_traveled, path_real) / np.linalg.norm(path_real)
            proj_orthogonal = np.sqrt(np.dot(path_traveled,path_traveled) - proj_parallel**2)

        # find orientation alignment
        current_yaw = current_pos[2]
        last_yaw = self.last_pos[2]
        path_yaw = np.arctan2(path_real[1], path_real[0])
        current_yaw_distance = np.abs(path_yaw - current_yaw)
        if current_yaw_distance > np.pi:
            current_yaw_distance = 2*np.pi - current_yaw_distance
        last_yaw_distance = np.abs(path_yaw - last_yaw)
        if last_yaw_distance > np.pi:
            last_yaw_distance = 2*np.pi - last_yaw_distance 
        yaw_diff = last_yaw_distance - current_yaw_distance

        # get reward
        reward = proj_parallel - proj_orthogonal + yaw_diff
        #reward = 10 if action==0 else 0
        #print("rewards\nproj_parallel: {}\nproj_orthogonal: {} \nyaw_diff: {}".format(proj_parallel,proj_orthogonal,yaw_diff))
        #print("reward obtained: {}".format(reward))

        return reward

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        print("step: {}".format(self.steps_done))
        #print("imu data: {}".format(self.imu))
        #print("joint data: {}".format(self.joint_state))

        # define command based on action (if action = None: do nothing)
        if action is not None:
            vel_cmd = self.get_vel_command(action)
            self.vel_pub.publish(vel_cmd)

        #data = None
        #while data is None:
        #    try:
        #        data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
        #    except:
        #        pass


        self.wait_gazebo_sim(wait_time = self.sim_time_step)       # wait X seconds in the simulation

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        # update number of steps
        self.steps_done += 1

        # update state and done boolean
        sensor_state = self.get_sensor_state()
        path_state = self.get_path_state()
        done = self.get_done()

        # save path info
        self.path_followed.append(path_state[0,0:2].detach().cpu().tolist())

        # compute reward
        reward = self.compute_reward(done, action, path_state)

        # update last position
        self.last_pos = path_state.squeeze(0)[:self.N_params_odom].detach().cpu().numpy()

        return [sensor_state, path_state], reward, done, {}

    def reset(self):

        # Resets the state of the environment and returns an initial observation.
        print("============= RESETTING ENVIRONMENT =============")
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        #rospy.wait_for_service('/gazebo/unpause_physics')
        #try:
        #    #resp_pause = pause.call()
        #    self.unpause()
        #except (rospy.ServiceException) as e:
        #    print ("/gazebo/unpause_physics service call failed")

        # Pause simulation
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        # Reset everything
        self.initiliaze_vars()

        # update state and done boolean
        sensor_state = self.get_sensor_state()
        path_state = self.get_path_state()

        # update last position
        self.last_pos = path_state.squeeze(0)[:self.N_params_odom].numpy()

        return [sensor_state, path_state]



    #################### ROS CALLBACK FUNCTIONS ########################

#    def vel_sub_callback(self, twist):
#        self.path_planner_cmd = twist
#        print("================================== GOT VEL FROM PATH PLANNER: {}".format(twist))


    def imu_sub_callback(self, imu):
        """
        components of Imu() :
        imu.orientation.{x,y,z,w}
        imu.orientation_covariance
        imu.angular_velocity.{x,y,z}
        imu.angular_velocity_covariance
        imu.linear_acceleration.{x,y,z}
        imu.linear_acceleration_covariance
        """

        # wait until other processes are done
        while(self.changing_imu):
            time.sleep(0.01)
        self.changing_imu = True

        self.imu_counter+=1

        # save only interesting data (ignore covariances)
        new_imu = torch.cat((
            torch.Tensor([imu.orientation.x,imu.orientation.y,imu.orientation.z,imu.orientation.w]),
            torch.Tensor([imu.linear_acceleration.x,imu.linear_acceleration.y,imu.linear_acceleration.z]),
            torch.Tensor([imu.angular_velocity.x,imu.angular_velocity.y,imu.angular_velocity.z]),
            ), 0)                      
        self.imu.append(new_imu)

        # set flags
        self.changing_imu = False

    def joint_sub_callback(self, joint_state):
        """
        components of JointState() :
        joint_state.name
        joint_state.position [11]
        joint_state.velocity [11] 
        joint_state.effort [7]       WARNING: effort is empty for data "chassis[0]"

        data can be :
        "chassis": [Base_joint, BOGIE_LEFT, BOGIE_RIGHT, ROCKER_RIGHT]
        "wheels": [ROCKER_LEFT, WHEEL_LEFT_1, WHEEL_LEFT_2, WHEEL_LEFT_3, WHEEL_RIGHT_1, WHEEL_RIGHT_2, WHEEL_RIGHT_3]
        """

        # wait until other processes are done
        while(self.changing_joint_state):
            time.sleep(0.01)
        self.changing_joint_state = True

        self.joint_counter+=1

        # define data type (0 or 1)
        datatype = joint_state_datatypes[1] if len(joint_state.position) == 4 else joint_state_datatypes[2]
        
        # need to merge two types of joint states (chassis[1] + wheels[2]):
        # if new joint state is empty, or if receive same type of data as last: save data to new joint
        if self.new_joint_state_datatype != joint_state_datatypes[0]:    # if new joint state already has a value 
            if datatype == self.new_joint_state_datatype:        # save new data instead of previous one
                self.new_joint_state.position = joint_state.position
                self.new_joint_state.velocity = joint_state.velocity
                if self.new_joint_state_datatype == joint_state_datatypes[1]: 
                    self.new_joint_state.effort = joint_state.effort
            else:
                if datatype == joint_state_datatypes[2]:   # add new data at end
                    self.new_joint_state.position = self.new_joint_state.position + joint_state.position
                    self.new_joint_state.velocity = self.new_joint_state.velocity + joint_state.velocity
                    self.new_joint_state.effort = joint_state.effort
                else:                                           # add new data at beginning
                    self.new_joint_state.position = joint_state.position + self.new_joint_state.position
                    self.new_joint_state.velocity = joint_state.velocity + self.new_joint_state.velocity
                
                # apply mask here if want to remove some variables

                # add complete new joint state to list
                new_joint_state_tensor = torch.cat((
                    torch.Tensor(self.new_joint_state.position),
                    torch.Tensor(self.new_joint_state.velocity),
                    torch.Tensor(self.new_joint_state.effort)
                    ), 0)            
                self.joint_state.append(new_joint_state_tensor)

                # reset new joint state
                self.new_joint_state_datatype = "none"

        else:      # save to new joint state
            self.new_joint_state.position = joint_state.position
            self.new_joint_state.velocity = joint_state.velocity
            # if effort has a value, update it
            if datatype == joint_state_datatypes[2]: 
                self.new_joint_state.effort = joint_state.effort
            self.new_joint_state_datatype = datatype

        # set flags
        self.changing_joint_state = False


    def odom_sub_callback(self, odometry):
        """
        Odometry:
        pose:
            pose:
                position:
                    x,y,z
                orientation:
                    x,y,z,w
        twist:
            ...
        """

        # wait for other processes to be done
        while(self.changing_odom):
            time.sleep(0.01)
        self.changing_odom = True

        # save relevant info
        self.odom.position.x = odometry.pose.pose.position.x
        self.odom.position.y = odometry.pose.pose.position.y
        self.odom.orientation = odometry.pose.pose.orientation

        # set flag
        self.changing_odom = False


    def path_sub_callback(self, path):
        """
        Path:
        poses[N]:
            pose:
                position:
                    x,y,z
                orientation:
                    x,y,z,w = [0,0,0,1]
        """

        # wait for other processes to be done
        while(self.changing_path):
            time.sleep(0.01)
        self.changing_path = True

        # save relevant info
        self.path = path
        self.path_to_follow_array.append(path)

        # set flag
        self.changing_path = False

    def clock_sub_callback(self, clock):
        """
        Clock:
        secs
        nsecs
        """
        while self.changing_clock:
            time.sleep(0.01)
        self.changing_clock = True
        self.clock = clock
        self.changing_clock = False



    #################### DEBUG + PLOT FUNCTIONS ########################

    def plot_path(self, fig=None, ax=None):
        if not fig or not ax:
            fig, ax = plt.subplots()

        ax.clear()

        # plot paths saved (path supposed to follow)
        for i, path in enumerate(self.path_to_follow_array):
            path_points = []
            for pose in path.poses:
                path_points.append([pose.pose.position.x,pose.pose.position.y])
            path_points = np.array(path_points)
            ax.plot(path_points[:,0], path_points[:,1], label="path "+str(i))

        # plot path followed
        self.path_followed = np.array(self.path_followed)
        ax.plot(self.path_followed[:,0], self.path_followed[:,1], label="path followed")

        ax.legend()
        plt.show()
        


