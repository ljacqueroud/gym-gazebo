import gym
import rospy
import roslaunch
import time
import numpy as np
import collections
import copy

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Imu
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import JointState

from gym.utils import seeding

action_commands = [
        [1,0,0,0],      # forward
        [-1,0,0,0],     # backward
        [0,0,0,1],      # left
        [0,0,0,-1]      # right
    ]

joint_state_datatypes = ["none", "chassis", "wheels"]


class GazeboRoverEnv(gazebo_env.GazeboEnv):


    #################### CLASS FUNCTIONS ########################

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboRover.launch.xml")

        # initiate ros services
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.imu_sub = rospy.Subscriber('os1_cloud_node/imu', Imu, self.imu_sub_callback)
        self.joint_sub = rospy.Subscriber('/rover/joint_states', JointState, self.joint_sub_callback)

        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        # environment parameters
        self.action_space = spaces.Discrete(1) #F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.steps_done = 0
        self.max_steps = 50

        # network parameters
        self.N_steps = 20       # number of time steps to keep in memory for network input

        # simulation parameters
        self.lin_speed = 5
        self.turn_speed = 7

        # ros parameters
        self.imu = collections.deque(maxlen=self.N_steps)
        self.joint_state = collections.deque(maxlen=self.N_steps)
        self.new_joint_state = JointState()
        self.new_joint_state_datatype = "none"

        self._seed()

    def discretize_observation(self,data,new_ranges):
        discretized_ranges = []
        min_range = 0.2
        done = False
        mod = len(data.ranges)/new_ranges
        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if data.ranges[i] == float ('Inf'):
                    discretized_ranges.append(6)
                elif np.isnan(data.ranges[i]):
                    discretized_ranges.append(0)
                else:
                    discretized_ranges.append(int(data.ranges[i]))
            if (min_range > data.ranges[i] > 0):
                done = True
        return discretized_ranges,done

    def get_vel_command(self, action):
        vel_cmd = Twist()
        vel_cmd.linear.x = action_commands[action][0] * self.lin_speed
        vel_cmd.linear.y = action_commands[action][1] * self.lin_speed
        vel_cmd.linear.z = action_commands[action][2] * self.lin_speed
        vel_cmd.angular.x = 0
        vel_cmd.angular.y = 0
        vel_cmd.angular.z = action_commands[action][3] * self.turn_speed

        return vel_cmd

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

        # define command based on action
        vel_cmd = self.get_vel_command(action)
        self.vel_pub.publish(vel_cmd)

        #data = None
        #while data is None:
        #    try:
        #        data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
        #    except:
        #        pass

        time.sleep(0.2)

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        # update number of steps
        self.steps_done += 1

        #state,done = self.discretize_observation(data,5)
        state = 0
        if self.steps_done >= self.max_steps:
            done = True
        else:
            done = False

        # define rewards
        if not done:
            if action == 0:
                reward = 5
            else:
                reward = 1
        else:
            reward = -200

        return state, reward, done, {}

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
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        #read laser data
        #data = None
        #while data is None:
        #    try:
        #        data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
        #    except:
        #        pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        #state = self.discretize_observation(data,5)
        state = 0
        self.steps_done = 0

        return state



    #################### CALLBACK FUNCTIONS ########################

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

        # save only interesting data (ignore covariances)
        new_imu = Imu()
        new_imu.orientation = imu.orientation
        new_imu.angular_velocity = imu.angular_velocity
        new_imu.linear_acceleration = imu.linear_acceleration
        self.imu.append(new_imu)

    def joint_sub_callback(self, joint_state):
        """
        components of JointState() :
        joint_state.name
        joint_state.position
        joint_state.velocity
        joint_state.effort      WARNING: effort is empty for data "chassis[0]"

        data can be :
        "chassis": [Base_joint, BOGIE_LEFT, BOGIE_RIGHT, ROCKER_RIGHT]
        "wheels": [ROCKER_LEFT, WHEEL_LEFT_1, WHEEL_LEFT_2, WHEEL_LEFT_3, WHEEL_RIGHT_1, WHEEL_RIGHT_2, WHEEL_RIGHT_3]
        """

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
                self.joint_state.append(copy.copy(self.new_joint_state))
                # reset new joint state
                self.new_joint_state_datatype = "none"

        else:      # save to new joint state
            self.new_joint_state.position = joint_state.position
            self.new_joint_state.velocity = joint_state.velocity
            # if effort has a value, update it
            if datatype == joint_state_datatypes[2]: 
                self.new_joint_state.effort = joint_state.effort
            self.new_joint_state_datatype = datatype

