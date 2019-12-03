
# Collaboration and Competition

---

You are welcome to use this coding environment to train your agent for the project.  Follow the instructions below to get started!

### 1. Start the Environment

Run the next code cell to install a few packages.  This line will take a few minutes to run!


```python
!pip -q install ./python
```

    [31mtensorflow 1.7.1 has requirement numpy>=1.13.3, but you'll have numpy 1.12.1 which is incompatible.[0m
    [31mipython 6.5.0 has requirement prompt-toolkit<2.0.0,>=1.0.15, but you'll have prompt-toolkit 2.0.10 which is incompatible.[0m


The environment is already saved in the Workspace and can be accessed at the file path provided below. 


```python
from unityagents import UnityEnvironment
import numpy as np

env = UnityEnvironment(file_name="/data/Tennis_Linux_NoVis/Tennis")
```

    INFO:unityagents:
    'Academy' started successfully!
    Unity Academy name: Academy
            Number of Brains: 1
            Number of External Brains : 1
            Lesson number : 0
            Reset Parameters :
    		
    Unity brain name: TennisBrain
            Number of Visual Observations (per agent): 0
            Vector Observation space type: continuous
            Vector Observation space size (per agent): 8
            Number of stacked Vector Observation: 3
            Vector Action space type: continuous
            Vector Action space size (per agent): 2
            Vector Action descriptions: , 


Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.


```python
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
```

### 2. Examine the State and Action Spaces

Run the code cell below to print some information about the environment.


```python
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])
```

    Number of agents: 2
    Size of each action: 2
    There are 2 agents. Each observes a state with length: 24
    The state for the first agent looks like: [ 0.          0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.          0.
      0.          0.         -6.65278625 -1.5        -0.          0.
      6.83172083  6.         -0.          0.        ]


### 3. Take Random Actions in the Environment

In the next code cell, you will learn how to use the Python API to control the agent and receive feedback from the environment.

Note that **in this coding environment, you will not be able to watch the agents while they are training**, and you should set `train_mode=True` to restart the environment.


```python
for i in range(5):                                         # play game for 5 episodes
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
```

    Total score (averaged over agents) this episode: -0.004999999888241291
    Total score (averaged over agents) this episode: -0.004999999888241291
    Total score (averaged over agents) this episode: -0.004999999888241291
    Total score (averaged over agents) this episode: -0.004999999888241291
    Total score (averaged over agents) this episode: -0.004999999888241291


When finished, you can close the environment.


```python
# env.close()
```

### 4. It's Your Turn!

Now it's your turn to train your own agent to solve the environment!  A few **important notes**:
- When training the environment, set `train_mode=True`, so that the line for resetting the environment looks like the following:
```python
env_info = env.reset(train_mode=True)[brain_name]
```
- To structure your work, you're welcome to work directly in this Jupyter notebook, or you might like to start over with a new file!  You can see the list of files in the workspace by clicking on **_Jupyter_** in the top left corner of the notebook.
- In this coding environment, you will not be able to watch the agents while they are training.  However, **_after training the agents_**, you can download the saved model weights to watch the agents on your own machine! 


```python
import matplotlib.pyplot as plt
%matplotlib inline

import numpy as np
import random
import time
import torch

from collections import deque
from workspace_utils import active_session
from ddpgAgent import Agent
from unityagents import UnityEnvironment
```


```python
seed = 15
```

### The chosen hyperparameters Hyperparameters

BUFFER_SIZE = int(3e5)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 2e-4         # learning rate for actor
LR_CRITIC = 2e-4        # learning rate fot critic
WEIGHT_DECAY = 0        # L2 weight decay

LEARN_EVERY = 20
LEARN_NUM = 10
GRAD_CLIPPING = 1.0

OU_SIGMA = 0.02
OU_THETA = 0.1

EPSILON = 1.0
EPSILON_DECAY = 1e-6

### Model architecture
Similar to single-agent Actor Critic architecture, each agent has it’s own actor and critic network. The actor network takes in the current state of agent and output a recommended action for that agent. However the critic part is slightly different from ordinary single-agent DDPG. Here the critic network of each agent has full visibility on the environment. It not only takes in the observation and action of that particular agent, but also observations and actions of all other agents as well. Critic network has much higher visibility on what is happening while actor network can only access to the observation information of the respective agent. The output of the critic network is, nevertheless, still the Q value estimated given a full observation input(all agents) and a full action input(all agents). The output of the actor network is a recommended action for that particular agent.

class Actor(nn.Module):
    
    def __init__(self, state_size, action_size, seed):
        super(Actor,self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 300)
        self.fc3 = nn.Linear(300, action_size)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self,x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

    
class Critic(nn.Module):
    
    def __init__(self, state_size, action_size,seed):
        
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500 + action_size,300)
        self.fc3 = nn.Linear(300, 1)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self,state, action):
        x_state = F.relu(self.bn1(self.fc1(state)))
        x = torch.cat((x_state, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
        


```python
# DDPG Function

def ddpg(n_episodes = 2500, max_t = 1000, print_every = 10):
    
    mean_scores = []
    moving_avgs = []
    best_score = -np.inf
    scores_window = deque(maxlen=100)
    
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode= True)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        agent.reset()
        start_time = time.time()
        for t in range(max_t):
            actions = agent.act(states, add_noise=True)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state,action,reward, next_state,done,t)
            states = next_states
            scores += rewards
            if np.any(dones):
                break
        duration = time.time() - start_time
        mean_scores.append(np.mean(scores))
        scores_window.append(mean_scores[-1])
        moving_avgs.append(np.mean(scores_window))
        
        if i_episode % print_every == 0:
            print('\rEpisode {} ({}s)\tMean: {:.1f}\tMoving Avg: {:.1f}'.format(\
                  i_episode, round(duration), mean_scores[-1], moving_avgs[-1]))
                  
        if moving_avgs[-1] >= 0.5 and i_episode >= 100:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, moving_avgs[-1]))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
        
    return mean_scores, moving_avgs
```


```python
start = time.time()
```


```python
agent = Agent(state_size = state_size, action_size = action_size, seed = seed)
with active_session():
    scores, avgs = ddpg()
```

    Episode 10 (0s)	Mean: 0.0	Moving Avg: 0.0
    Episode 20 (1s)	Mean: -0.0	Moving Avg: 0.0
    Episode 30 (1s)	Mean: -0.0	Moving Avg: 0.0
    Episode 40 (0s)	Mean: -0.0	Moving Avg: -0.0
    Episode 50 (0s)	Mean: -0.0	Moving Avg: -0.0
    Episode 60 (0s)	Mean: -0.0	Moving Avg: -0.0
    Episode 70 (0s)	Mean: -0.0	Moving Avg: -0.0
    Episode 80 (0s)	Mean: -0.0	Moving Avg: -0.0
    Episode 90 (0s)	Mean: -0.0	Moving Avg: -0.0
    Episode 100 (1s)	Mean: -0.0	Moving Avg: -0.0
    Episode 110 (1s)	Mean: -0.0	Moving Avg: -0.0
    Episode 120 (1s)	Mean: -0.0	Moving Avg: -0.0
    Episode 130 (1s)	Mean: -0.0	Moving Avg: -0.0
    Episode 140 (1s)	Mean: -0.0	Moving Avg: -0.0
    Episode 150 (1s)	Mean: -0.0	Moving Avg: -0.0
    Episode 160 (1s)	Mean: -0.0	Moving Avg: -0.0
    Episode 170 (1s)	Mean: -0.0	Moving Avg: -0.0
    Episode 180 (2s)	Mean: 0.1	Moving Avg: -0.0
    Episode 190 (1s)	Mean: -0.0	Moving Avg: -0.0
    Episode 200 (1s)	Mean: -0.0	Moving Avg: -0.0
    Episode 210 (1s)	Mean: -0.0	Moving Avg: -0.0
    Episode 220 (1s)	Mean: -0.0	Moving Avg: -0.0
    Episode 230 (1s)	Mean: -0.0	Moving Avg: -0.0
    Episode 240 (1s)	Mean: -0.0	Moving Avg: -0.0
    Episode 250 (1s)	Mean: 0.0	Moving Avg: -0.0
    Episode 260 (1s)	Mean: 0.0	Moving Avg: 0.0
    Episode 270 (1s)	Mean: 0.0	Moving Avg: 0.0
    Episode 280 (1s)	Mean: 0.0	Moving Avg: 0.0
    Episode 290 (1s)	Mean: 0.0	Moving Avg: 0.0
    Episode 300 (1s)	Mean: 0.0	Moving Avg: 0.0
    Episode 310 (1s)	Mean: -0.0	Moving Avg: 0.0
    Episode 320 (1s)	Mean: 0.0	Moving Avg: 0.0
    Episode 330 (1s)	Mean: -0.0	Moving Avg: 0.0
    Episode 340 (1s)	Mean: 0.0	Moving Avg: 0.0
    Episode 350 (1s)	Mean: 0.0	Moving Avg: 0.0
    Episode 360 (1s)	Mean: 0.0	Moving Avg: 0.0
    Episode 370 (1s)	Mean: 0.0	Moving Avg: 0.0
    Episode 380 (1s)	Mean: 0.0	Moving Avg: 0.0
    Episode 390 (1s)	Mean: 0.0	Moving Avg: 0.0
    Episode 400 (1s)	Mean: 0.0	Moving Avg: 0.0
    Episode 410 (2s)	Mean: 0.1	Moving Avg: 0.0
    Episode 420 (1s)	Mean: 0.0	Moving Avg: 0.0
    Episode 430 (2s)	Mean: 0.1	Moving Avg: 0.0
    Episode 440 (1s)	Mean: 0.0	Moving Avg: 0.0
    Episode 450 (1s)	Mean: 0.0	Moving Avg: 0.1
    Episode 460 (2s)	Mean: 0.1	Moving Avg: 0.1
    Episode 470 (3s)	Mean: 0.2	Moving Avg: 0.1
    Episode 480 (2s)	Mean: 0.1	Moving Avg: 0.1
    Episode 490 (2s)	Mean: 0.1	Moving Avg: 0.1
    Episode 500 (1s)	Mean: -0.0	Moving Avg: 0.1
    Episode 510 (1s)	Mean: 0.0	Moving Avg: 0.1
    Episode 520 (1s)	Mean: 0.0	Moving Avg: 0.1
    Episode 530 (1s)	Mean: 0.0	Moving Avg: 0.1
    Episode 540 (1s)	Mean: -0.0	Moving Avg: 0.1
    Episode 550 (1s)	Mean: 0.0	Moving Avg: 0.1
    Episode 560 (1s)	Mean: -0.0	Moving Avg: 0.1
    Episode 570 (1s)	Mean: 0.0	Moving Avg: 0.1
    Episode 580 (1s)	Mean: 0.0	Moving Avg: 0.1
    Episode 590 (15s)	Mean: 1.3	Moving Avg: 0.1
    Episode 600 (2s)	Mean: 0.1	Moving Avg: 0.1
    Episode 610 (4s)	Mean: 0.3	Moving Avg: 0.1
    Episode 620 (2s)	Mean: 0.1	Moving Avg: 0.2
    Episode 630 (28s)	Mean: 2.7	Moving Avg: 0.4
    Episode 640 (9s)	Mean: 0.8	Moving Avg: 0.4
    
    Environment solved in 549 episodes!	Average Score: 0.51



```python
end = time.time()
elapsed = (end - start) / 60.0 # in minutes
print("\nElapsed Time: {0:3.2f} mins.".format(elapsed)) 
```

    
    Elapsed Time: 19.70 mins.



```python
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores, label='DDPG')
plt.plot(np.arange(len(scores)), avgs, c='r', label='Moving AVG')
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.legend(loc='upper left');
plt.show()
```


![png](output_20_0.png)



```python
env.close()
```

### Ideas for improving the agent's performance
Batch Normalization Neural network enhancement for a better performance.


```python

```
