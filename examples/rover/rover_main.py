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
import matplotlib.pyplot as plt

import dnn

from gym_gazebo.envs.rover.rover_gazebo_env import GazeboRoverEnv
 

def render(x, env, plotting, saving):
    render_skip = 0 #Skip first X episodes.
    render_interval = 5 #Show render Every Y episodes.

    if plotting or saving:
        if (x%render_interval == 0) and (x >= render_skip):
            env.plot_path(x, plotting=plotting, saving=saving)

if __name__ == '__main__':

    # parameters for plotting/saving the plots
    plotting = False
    saving = True

    # setup gym environment
    print("============== Setting up gym environment ===============")
    env = gym.make('GazeboRover-v0')

    outdir = '/tmp/gazebo_gym_experiments'
    env = gym.wrappers.Monitor(env, outdir, force=True)

    #plotter = liveplot.LivePlot(outdir)

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

    # training parameters
    total_episodes = 10000
    max_episode_length = 60
    env.set_max_steps(max_episode_length)
    highest_reward = 0

    lr = 2e-2
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)

    p_random_action = 0.2           # probability of discard model output and picking at random
    p_random_action_eps = 0.96      # epsilon reduction of the random probability after each episode
                                    # after 50 eps: p_random_action = 0.1*0.96^50 = 0.0130

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

            # convert state to correct device
            state[0] = state[0].to(device)
            state[1] = state[1].to(device)

            # get action probabilities from network output
            action_prob = model(state[0], state[1]).squeeze(0)       # get action [1,N] and remove dim 0 [N,] 
            print("action probability: {} with random action prob: {}".format(action_prob[:],p_random_action))

            # Pick an action based on the current state
            if np.random.random_sample() < p_random_action:
                action = np.random.randint(0, N_actions)
                print("random action chosen: {}".format(action))
            else:
                action = np.random.choice(action_array, p = (action_prob.data.numpy() if device == torch.device('cpu') else action_prob.cpu().data.numpy()))
                print("action chosen: {}".format(action))

            #print("path: {}".format(env.get_path()))
            #print("odom: {}".format(env.get_odom()))

            # Execute the action and get feedback
            next_state, reward, done, info = env.step(action)
            #print("######## state: {}".format(next_state))
            transitions.append((state[0], state[1], action, reward, steps))
            cumulated_reward += reward
            state = next_state

            #print("current reward: {}, cumulated reward: {}".format(reward, cumulated_reward))

            env._flush(force=True)

            # update random action probability
            p_random_action *= p_random_action_eps

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
        action_batch = torch.Tensor([a for (s1,s2,a,r,n) in transitions]).to(device)

        # compute expected reward (= at each step, the reward that it gets after that step)
        batch_Gvals = []
        for i in range(len(transitions)):
            new_Gval = 0
            #for j in range(i,len(transitions)):
            #     new_Gval = new_Gval + reward_batch[j].numpy()
            new_Gval = new_Gval + reward_batch[i].numpy()
            batch_Gvals.append(new_Gval)
        expected_returns_batch = torch.FloatTensor(batch_Gvals) if device==torch.device('cpu') else torch.cuda.FloatTensor(batch_Gvals)
        #expected_returns_batch -= expected_returns_batch.min()
        expected_returns_batch /= expected_returns_batch.abs().max()

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
        print("Episode {}\tScore: {:.2f}\tTime: {}:{}:{}".format(x, score[-1],h,m,s))

        # render plots
        render(x,env, plotting, saving)

    #Github table content
    print("\n|"+str(total_episodes)+"|"+str(highest_reward)+"| PICTURE |")

    scores_sorted = score.copy()
    scores_sorted.sort()

    #print("Parameters: a="+str)
    print("Mean score of last 50 episodes: {:0.2f}".format(np.mean(score[-50:-1])))
    print("Best 10 scores: {:0.2f}".format(reduce(lambda x, y: x + y, scores_sorted[-10:]) / len(scores_sorted[-10:])))

    env.close()
