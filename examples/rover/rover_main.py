#!/usr/bin/env python
import gym
from gym import wrappers
import gym_gazebo
import time
import numpy
import random
import time
import liveplot
import torch
import torch.nn as nn

import dnn

from gym_gazebo.envs.rover.gazebo_rover import GazeboRoverEnv
 

def render():
    render_skip = 0 #Skip first X episodes.
    render_interval = 50 #Show render Every Y episodes.
    render_episodes = 10 #Show Z episodes every rendering.

    if (x%render_interval == 0) and (x != 0) and (x > render_skip):
        env.render()
    elif ((x-render_episodes)%render_interval == 0) and (x != 0) and (x > render_skip) and (render_episodes < x):
        env.render(close=True)

if __name__ == '__main__':

    # setup gym environment
    env = gym.make('GazeboRover-v0')

    outdir = '/tmp/gazebo_gym_experiments'
    env = gym.wrappers.Monitor(env, outdir, force=True)

    plotter = liveplot.LivePlot(outdir)

    last_time_steps = numpy.ndarray(0)

    # model dimensions
    N_params = env.get_N_params()       # number of input parameters (= N_channels)
    N_actions = env.get_N_actions()     # number of output actions (= N_output)
    N_steps = env.get_N_steps()         # when changing n_steps, also change it in gazebo_rover.py (environment)

    # define the model
    model = dnn.Model(N_channels = N_params, N_output = N_actions, N_steps = N_steps)

    # define device (gpu or cpu)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Device: ", device)
    model.to(device)

    # model parameters
    total_episodes = 10000
    max_episode_length = 100
    highest_reward = 0

    lr = 1e-3
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)

    start_time = time.time()


    for x in range(total_episodes):
        done = False

        cumulated_reward = 0 #Should going forward give more reward then L/R ?
        steps = 0

        observation = env.reset()

        #if qlearn.epsilon > 0.05:
        #    qlearn.epsilon *= epsilon_discount

        #render() #defined above, not env.render()

        #state = ''.join(map(str, observation))
        state = 0

        #for i in range(200):
        while True:
            steps += 1

            # Pick an action based on the current state
            #action = qlearn.chooseAction(state)
            action = 1

            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)
            cumulated_reward += reward

            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            #nextState = ''.join(map(str, observation))
            nextState = 0

            #qlearn.learn(state, action, reward, nextState)

            env._flush(force=True)

            if not(done):
                state = nextState
            else:
                last_time_steps = numpy.append(last_time_steps, [int(steps)])
                break

        if x % 100 == 0:
            plotter.plot(env)

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        #print ("EP: "+str(x+1)+" - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+" - epsilon: "+str(round(qlearn.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s))
        print ("EP: "+str(x+1)+" Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s))

    #Github table content
    print ("\n|"+str(total_episodes)+"|"+str(qlearn.alpha)+"|"+str(qlearn.gamma)+"|"+str(initial_epsilon)+"*"+str(epsilon_discount)+"|"+str(highest_reward)+"| PICTURE |")

    l = last_time_steps.tolist()
    l.sort()

    #print("Parameters: a="+str)
    print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
