#!/usr/bin/env python
# coding: utf-8

# # Project: Train a Quadcopter How to Fly
# 
# Design an agent to fly a quadcopter, and then train it using a reinforcement learning algorithm of your choice! 
# 
# Try to apply the techniques you have learnt, but also feel free to come up with innovative ideas and test them.

# ## Instructions
# 
# Take a look at the files in the directory to better understand the structure of the project. 
# 
# - `task.py`: Define your task (environment) in this file.
# - `agents/`: Folder containing reinforcement learning agents.
#     - `policy_search.py`: A sample agent has been provided here.
#     - `agent.py`: Develop your agent here.
# - `physics_sim.py`: This file contains the simulator for the quadcopter.  **DO NOT MODIFY THIS FILE**.
# 
# For this project, you will define your own task in `task.py`.  Although we have provided a example task to get you started, you are encouraged to change it.  Later in this notebook, you will learn more about how to amend this file.
# 
# You will also design a reinforcement learning agent in `agent.py` to complete your chosen task.  
# 
# You are welcome to create any additional files to help you to organize your code.  For instance, you may find it useful to define a `model.py` file defining any needed neural network architectures.
# 
# ## Controlling the Quadcopter
# 
# We provide a sample agent in the code cell below to show you how to use the sim to control the quadcopter.  This agent is even simpler than the sample agent that you'll examine (in `agents/policy_search.py`) later in this notebook!
# 
# The agent controls the quadcopter by setting the revolutions per second on each of its four rotors.  The provided agent in the `Basic_Agent` class below always selects a random action for each of the four rotors.  These four speeds are returned by the `act` method as a list of four floating-point numbers.  
# 
# For this project, the agent that you will implement in `agents/agent.py` will have a far more intelligent method for selecting actions!

# In[28]:


import random

class Basic_Agent():
    def __init__(self, task):
        self.task = task
    
    def act(self):
        new_thrust = random.gauss(450., 25.)
        return [new_thrust + random.gauss(0., 1.) for x in range(4)]


# Run the code cell below to have the agent select actions to control the quadcopter.  
# 
# Feel free to change the provided values of `runtime`, `init_pose`, `init_velocities`, and `init_angle_velocities` below to change the starting conditions of the quadcopter.
# 
# The `labels` list below annotates statistics that are saved while running the simulation.  All of this information is saved in a text file `data.txt` and stored in the dictionary `results`.  

# In[29]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import csv
import numpy as np
from task import Task

# Modify the values below to give the quadcopter a different starting position.
runtime = 5.                                     # time limit of the episode
init_pose = np.array([0., 0., 10., 0., 0., 0.])  # initial pose
init_velocities = np.array([0., 0., 0.])         # initial velocities
init_angle_velocities = np.array([0., 0., 0.])   # initial angle velocities
file_output = 'data.txt'                         # file name for saved results

# Setup
task = Task(init_pose, init_velocities, init_angle_velocities, runtime)
agent = Basic_Agent(task)
done = False
labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
          'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
          'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']
results = {x : [] for x in labels}

# Run the simulation, and save the results.
with open(file_output, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(labels)
    while True:
        rotor_speeds = agent.act()
        _, _, done = task.step(rotor_speeds)
        to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(rotor_speeds)
        for ii in range(len(labels)):
            results[labels[ii]].append(to_write[ii])
        writer.writerow(to_write)
        if done:
            break
results


# Run the code cell below to visualize how the position of the quadcopter evolved during the simulation.

# In[30]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(results['time'], results['x'], label='x')
plt.plot(results['time'], results['y'], label='y')
plt.plot(results['time'], results['z'], label='z')
plt.legend()
_ = plt.ylim()


# The next code cell visualizes the velocity of the quadcopter.

# In[31]:


plt.plot(results['time'], results['x_velocity'], label='x_hat')
plt.plot(results['time'], results['y_velocity'], label='y_hat')
plt.plot(results['time'], results['z_velocity'], label='z_hat')
plt.legend()
_ = plt.ylim()


# Next, you can plot the Euler angles (the rotation of the quadcopter over the $x$-, $y$-, and $z$-axes),

# In[32]:


plt.plot(results['time'], results['phi'], label='phi')
plt.plot(results['time'], results['theta'], label='theta')
plt.plot(results['time'], results['psi'], label='psi')
plt.legend()
_ = plt.ylim()


# before plotting the velocities (in radians per second) corresponding to each of the Euler angles.

# In[33]:


plt.plot(results['time'], results['phi_velocity'], label='phi_velocity')
plt.plot(results['time'], results['theta_velocity'], label='theta_velocity')
plt.plot(results['time'], results['psi_velocity'], label='psi_velocity')
plt.legend()
_ = plt.ylim()


# Finally, you can use the code cell below to print the agent's choice of actions.  

# In[34]:


plt.plot(results['time'], results['rotor_speed1'], label='Rotor 1 revolutions / second')
plt.plot(results['time'], results['rotor_speed2'], label='Rotor 2 revolutions / second')
plt.plot(results['time'], results['rotor_speed3'], label='Rotor 3 revolutions / second')
plt.plot(results['time'], results['rotor_speed4'], label='Rotor 4 revolutions / second')
plt.legend()
_ = plt.ylim()


# When specifying a task, you will derive the environment state from the simulator.  Run the code cell below to print the values of the following variables at the end of the simulation:
# - `task.sim.pose` (the position of the quadcopter in ($x,y,z$) dimensions and the Euler angles),
# - `task.sim.v` (the velocity of the quadcopter in ($x,y,z$) dimensions), and
# - `task.sim.angular_v` (radians/second for each of the three Euler angles).

# In[35]:


# the pose, velocity, and angular velocity of the quadcopter at the end of the episode
print(task.sim.pose)
print(task.sim.v)
print(task.sim.angular_v)


# In the sample task in `task.py`, we use the 6-dimensional pose of the quadcopter to construct the state of the environment at each timestep.  However, when amending the task for your purposes, you are welcome to expand the size of the state vector by including the velocity information.  You can use any combination of the pose, velocity, and angular velocity - feel free to tinker here, and construct the state to suit your task.
# 
# ## The Task
# 
# A sample task has been provided for you in `task.py`.  Open this file in a new window now. 
# 
# The `__init__()` method is used to initialize several variables that are needed to specify the task.  
# - The simulator is initialized as an instance of the `PhysicsSim` class (from `physics_sim.py`).  
# - Inspired by the methodology in the original DDPG paper, we make use of action repeats.  For each timestep of the agent, we step the simulation `action_repeats` timesteps.  If you are not familiar with action repeats, please read the **Results** section in [the DDPG paper](https://arxiv.org/abs/1509.02971).
# - We set the number of elements in the state vector.  For the sample task, we only work with the 6-dimensional pose information.  To set the size of the state (`state_size`), we must take action repeats into account.  
# - The environment will always have a 4-dimensional action space, with one entry for each rotor (`action_size=4`). You can set the minimum (`action_low`) and maximum (`action_high`) values of each entry here.
# - The sample task in this provided file is for the agent to reach a target position.  We specify that target position as a variable.
# 
# The `reset()` method resets the simulator.  The agent should call this method every time the episode ends.  You can see an example of this in the code cell below.
# 
# The `step()` method is perhaps the most important.  It accepts the agent's choice of action `rotor_speeds`, which is used to prepare the next state to pass on to the agent.  Then, the reward is computed from `get_reward()`.  The episode is considered done if the time limit has been exceeded, or the quadcopter has travelled outside of the bounds of the simulation.
# 
# In the next section, you will learn how to test the performance of an agent on this task.

# ## The Agent
# 
# The sample agent given in `agents/policy_search.py` uses a very simplistic linear policy to directly compute the action vector as a dot product of the state vector and a matrix of weights. Then, it randomly perturbs the parameters by adding some Gaussian noise, to produce a different policy. Based on the average reward obtained in each episode (`score`), it keeps track of the best set of parameters found so far, how the score is changing, and accordingly tweaks a scaling factor to widen or tighten the noise.
# 
# Run the code cell below to see how the agent performs on the sample task.

# In[36]:


import sys
import pandas as pd
from agents.policy_search import PolicySearch_Agent
from task import Task

num_episodes = 1000
target_pos = np.array([0., 0., 10.])
task = Task(target_pos=target_pos)
agent = PolicySearch_Agent(task) 

for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode() # start a new episode
    while True:
        action = agent.act(state) 
        next_state, reward, done = task.step(action)
        agent.step(reward, done)
        state = next_state
        if done:
            print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
                i_episode, agent.score, agent.best_score, agent.noise_scale), end="")  # [debug]
            break
    sys.stdout.flush()


# This agent should perform very poorly on this task.  And that's where you come in!

# ## Define the Task, Design the Agent, and Train Your Agent!
# 
# Amend `task.py` to specify a task of your choosing.  If you're unsure what kind of task to specify, you may like to teach your quadcopter to takeoff, hover in place, land softly, or reach a target pose.  
# 
# After specifying your task, use the sample agent in `agents/policy_search.py` as a template to define your own agent in `agents/agent.py`.  You can borrow whatever you need from the sample agent, including ideas on how you might modularize your code (using helper methods like `act()`, `learn()`, `reset_episode()`, etc.).
# 
# Note that it is **highly unlikely** that the first agent and task that you specify will learn well.  You will likely have to tweak various hyperparameters and the reward function for your task until you arrive at reasonably good behavior.
# 
# As you develop your agent, it's important to keep an eye on how it's performing. Use the code above as inspiration to build in a mechanism to log/save the total rewards obtained in each episode to file.  If the episode rewards are gradually increasing, this is an indication that your agent is learning.

# In[40]:


## TODO: Train your agent here.
# Now train with experiences

import sys
import pandas as pd
from agents.agent import critic_actor_agent
from task import Task

rewards_list = []

# agent task: Take off 

num_episodes = 1000
target_pos = np.array([0., 0., 10])
init_pose = np.array([0., 0., 1.,0,0,0])

init_velocities = np.array([0., 0., 0.])
init_angle_velocities = np.array([0., 0., 0.])
result_file = 'result.txt'
labels = ['episode', 'reward','best_reward']
rewards_list = {i : [] for i in labels}
eposide_score=[]
task = Task(init_pose=init_pose, target_pos=target_pos,init_velocities=init_velocities, init_angle_velocities=init_angle_velocities)  
agent = critic_actor_agent(task) 
step_number =0

with open(result_file, 'w') as file:
    for i_episode in range(1, num_episodes+1):
        state = agent.reset_episode() # start a new episode
        while True:
            action = agent.act(state) 
            next_state, reward, done = task.step(action)
            agent.step(action,reward, next_state, done)
            state = next_state
            step_number +=1
            if done:
                eposide_score.append(agent.mark)
                data= "\rEpisode = {:4d}, reward = {:7.3f} (best reward= {:7.3f}), step={:4d} agent_position = ({:7.3f},{:7.3f},{:7.3f}) \n".format(
                    i_episode, agent.mark , agent.best_mark, step_number, task.sim.pose[0], task.sim.pose[1], task.sim.pose[2])
                print(data, end="")  # [debug]
                file.write(data)
                file.flush()
                break
            
            rewards_list['episode'].append(i_episode)
            rewards_list['reward'].append(agent.mark)
            rewards_list['best_reward'].append(agent.best_mark)
        sys.stdout.flush()


# j## Plot the Rewards
# 
# Once you are satisfied with your performance, plot the episode rewards, either from a single run, or averaged over multiple runs. 

# In[41]:


## TODO: Plot the rewards.
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(rewards_list['episode'],rewards_list['reward'] )
plt.show()


# In[42]:


import matplotlib.pyplot as plt
plt.plot(eposide_score)
plt.show()


# ## Reflections
# 
# **Question 1**: Describe the task that you specified in `task.py`.  How did you design the reward function?
# 
# **Answer**:   
# - I choosed the takeoff task. I used the same reward code example suggested for this project which is reward = 1-(0.3*(abs(self.sim.pose[:3] - self.target_pos))).sum() but I repleaced the multiplyer 0.3 with smaller value (.003) becuase when I tested, I noticed that the agent learned better and faster with smaller value (reward = 1 - 0.003*abs(self.target_pos[2] - self.sim.pose[2]).sum()). and I specified the z position becuase is the most important position in take off task. Then I took the tanh of the reward to convert it in rang [-1,1]. It was the hardest part to specifying the best reward function to use.

# **Question 2**: Discuss your agent briefly, using the following questions as a guide:
# 
# - What learning algorithm(s) did you try? What worked best for you?
# - What was your final choice of hyperparameters (such as $\alpha$, $\gamma$, $\epsilon$, etc.)?
# - What neural network architecture did you use (if any)? Specify layers, sizes, activation functions, etc.
# 
# **Answer**:
# - I use Actor-critic method which is the suggested DDPG alogrithm, and the use the code provided with this project.
# - the $\gamma$ = 0.99, $\epsilon$ = 0.1 and $\alpha$= 0.1 the same as suggested code but I maximize the batch_size to 128 to get better result.
# - I use the same provided neural network archetecture becuase it's simple but I change some hyperparameter. The actor class uses three hidden lyers with relue activation method and increase the Units of hidden layer to 128, 256 and 300 respectively. the final lyer use segmid acrivation method. the cirtic uses four hidden layers with relu activation too and I use 64 and 128 units for the hidden layers. 

# **Question 3**: Using the episode rewards plot, discuss how the agent learned over time.
# 
# - Was it an easy task to learn or hard?
# - Was there a gradual learning curve, or an aha moment?
# - How good was the final performance of the agent? (e.g. mean rewards over the last 10 episodes)
# 
# **Answer**:
# - it was a hard task to make the agent learned, the agent learned arount 1000 episodes, the training took a lot of time. it goes dowm and up (a gradual learning curve) at the begining of episodes. but then the agent starts learning and the z position start changeing and become better, as shown in the above figure. So the agent learned over the time. 
# - the final performance of the agent is become beter after 100 eposides the agent start learning and its performance improved but I think we can imporve it more and more. 

# **Question 4**: Briefly summarize your experience working on this project. You can use the following prompts for ideas.
# 
# - What was the hardest part of the project? (e.g. getting started, plotting, specifying the task, etc.)
# - Did you find anything interesting in how the quadcopter or your agent behaved?
# 
# **Answer**:
# - Actually this project was the hardest one for me, I think the hardest part for me was when I was thinking how to spesify the reward at each step, I was trying many multipliers until I got the better one. Also it took long time to train the agent. 
# - I think it's interesting how the agent learned differently at each time I change the reward function, I belive that there are lots of ways to reward the agent thus behaves better than what I got now.  

# In[ ]:




