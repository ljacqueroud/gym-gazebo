import gym
import rospy
import roslaunch
import time
import numpy as np
import collections
import copy
import torch

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


    #################### INIT FUNCTION ########################

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

        # network parameters
        self.N_params_imu = 10                      # number of imu data
        self.N_params_joint_state = 29              # number of joint state data
        self.N_params = 10 + 29 
        self.N_actions = len(action_commands)       # number of action commands
        self.N_steps = 30       # number of time steps to keep in memory for network input
                                # when changing N_steps, also change it in rover_main.py (main script)

        # simulation parameters
        self.lin_speed = 5
        self.turn_speed = 7

        # ros parameters
        self.new_joint_state = JointState()
        self.new_joint_state_datatype = "none"

        # parameters to keep in memory
        self.imu = collections.deque(maxlen=self.N_steps)
        self.joint_state = collections.deque(maxlen=self.N_steps)
        self.initialize_state()

        self.imu_counter=0
        self.joint_counter=0

        # flags for multi threads conflicts
        self.changing_params = False
        self.saving_state = False

        self._seed()

    #################### GET AND SET FUNCTIONS ########################

    def get_N_params(self):
        return self.N_params

    def get_N_actions(self):
        return self.N_actions

    def get_N_steps(self):
        return self.N_steps

    def get_state(self):
        # wait until other processes are done
        while(self.changing_params):
            time.sleep(0.01)
        self.saving_state = True

        print("imu {}".format(self.imu_counter))
        print("joint_state {}".format(self.joint_counter))
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

        # set flag
        self.saving_state = False

        return state.unsqueeze(0)                       # final tensor of size [1,39, N_steps]

    def get_done(self):
        if self.steps_done >= self.max_steps:
            done = True
        else:
            done = False
        return done

    def set_max_steps(self, max_steps):
        self.max_steps = max_steps

    #################### ENVIRONMENT FUNCTIONS ########################

    def initialize_state(self):
        """ sets imu and joint_states to zero
        """
        for _ in range(self.N_steps):
            self.imu.append(torch.zeros([self.N_params_imu]))
            self.joint_state.append(torch.zeros([self.N_params_joint_state]))

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
        # update state and done boolean
        state = self.get_state()
        done = self.get_done()

        # define rewards
        if not done:
            if action == 0:
                reward = 5
            else:
                reward = 1
        else:
            reward = 10

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

        # reset and get state
        self.initialize_state()
        self.steps_done = 0

        return self.get_state()



    #################### ROS CALLBACK FUNCTIONS ########################

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
        while(self.changing_params or self.saving_state):
            time.sleep(0.01)
        self.changing_params = True

        self.imu_counter+=1

        # save only interesting data (ignore covariances)
        new_imu = torch.cat((
            torch.Tensor([imu.orientation.x,imu.orientation.y,imu.orientation.z,imu.orientation.w]),
            torch.Tensor([imu.linear_acceleration.x,imu.linear_acceleration.y,imu.linear_acceleration.z]),
            torch.Tensor([imu.angular_velocity.x,imu.angular_velocity.y,imu.angular_velocity.z]),
            ), 0)                      
        self.imu.append(new_imu)

        # set flags
        self.changing_params = False

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
        while(self.changing_params or self.saving_state):
            time.sleep(0.01)
        self.changing_params = True

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
        self.changing_params = False

