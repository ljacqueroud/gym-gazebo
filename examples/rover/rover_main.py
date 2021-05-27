#!/usr/bin/env python
import gym
from gym import wrappers
import gym_gazebo
import time
import numpy as np
import random
import time
import liveplot
import torch
import torch.nn as nn
from torch import optim

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
    print("============== Setting up gym environment ===============")
    env = gym.make('GazeboRover-v0')

    outdir = '/tmp/gazebo_gym_experiments'
    env = gym.wrappers.Monitor(env, outdir, force=True)

    plotter = liveplot.LivePlot(outdir)

    last_time_steps = np.ndarray(0)

    # model dimensions
    N_params_sensor_state = env.get_N_params_sensor_state() # number of input params (= N_channels)
    N_params_path_state = env.get_N_params_path_state()     # number of input params on fc (= N_y)
    N_actions = env.get_N_actions()     # number of output actions (= N_output)
    N_steps = env.get_N_steps()         # number of time steps it remembers
    action_array = [i for i in range(N_actions)]    # create array of actions ID

    # define the model
    print("============== Creating neural network ===============")
    model = dnn.Model(N_channels = N_params_sensor_state, N_output = N_actions, N_steps = N_steps, N_y = N_params_path_state)

    # define device (gpu or cpu)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Device: ", device)
    model.to(device)

    # model parameters
    total_episodes = 10000
    max_episode_length = 50
    env.set_max_steps(max_episode_length)
    highest_reward = 0

    lr = 2e-2
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)

    score = []
    start_time = time.time()

    print("============== Starting training loop ===============")

    for x in range(total_episodes):
        done = False

        cumulated_reward = 0
        steps = 0
        transitions = []

        # reset environment
        state = env.reset()                 # state = [sensor_state, path_state]

        while True:
            steps += 1

            # Pick an action based on the current state
            action_prob = model(state[0], state[1]).squeeze(0)       # get action [1,N] and remove dim 0 [N,] 
            print("action probability: {}".format(action_prob[:]))
            action = np.random.choice(action_array, p=action_prob.data.numpy())
            print("action chosen: {}".format(action))

            #print("path: {}".format(env.get_path()))
            #print("odom: {}".format(env.get_odom()))

            # Execute the action and get feedback
            next_state, reward, done, info = env.step(action)
            #print("######## state: {}".format(next_state))
            transitions.append((state[0], state[1], action, reward, steps))
            cumulated_reward += reward
            state = next_state

            env._flush(force=True)

            # End episode when done
            if done:
                last_time_steps = np.append(last_time_steps, [int(steps)])
                break

        # save reward
        if highest_reward < cumulated_reward:
            highest_reward = cumulated_reward
        score.append(cumulated_reward)
        reward_batch = torch.Tensor([r for (s1,s2,a,r,n) in transitions])#.flip(dims=(0,))
        sensor_state_batch = torch.cat([s1 for (s1,s2,a,r,n) in transitions])
        path_state_batch = torch.cat([s2 for (s1,s2,a,r,n) in transitions])
        action_batch = torch.Tensor([a for (s1,s2,a,r,n) in transitions])

        # compute expected reward (= at each step, the reward that it gets after that step)
        batch_Gvals = []
        for i in range(len(transitions)):
            new_Gval = 0
            for j in range(i,len(transitions)):
                 new_Gval = new_Gval + reward_batch[j].numpy()
            #new_Gval = new_Gval + reward_batch[i].numpy()
            batch_Gvals.append(new_Gval)
        expected_returns_batch = torch.FloatTensor(batch_Gvals)
        expected_returns_batch -= expected_returns_batch.min()
        expected_returns_batch *= expected_returns_batch.max()

        # get the model output
        pred_batch = model(sensor_state_batch, path_state_batch)
        prob_batch = pred_batch.gather(dim=1, index=action_batch.long().view(-1,1)).squeeze()

        # compute the loss
        loss = -torch.sum(torch.log(prob_batch) * expected_returns_batch)

        # bakward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        print("Episode {}\tScore: {:.2f}\tTime: {}:{}:{}".format(x+1, score[-1],h,m,s))

    #Github table content
    print("\n|"+str(total_episodes)+"|"+str(highest_reward)+"| PICTURE |")

    scores_sorted = score.copy()
    scores_sorted.sort()

    #print("Parameters: a="+str)
    print("Mean score of last 50 episodes: {:0.2f}".format(np.mean(score[-50:-1])))
    print("Best 10 scores: {:0.2f}".format(reduce(lambda x, y: x + y, scores_sorted[-10:]) / len(scores_sorted[-10:])))

    env.close()
