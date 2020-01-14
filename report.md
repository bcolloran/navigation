[//]: # "Image References"
[plot1]: ./rewards.png "Trained Agent"

# Banana Navigation Report

## Learning algorithm

For this project, the DDQN algorithm was used.

### Hyperparameters

The hyperparameters for this project were set at:

```
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network
```

### Network architecture

The neural network consists of three hidden layers of width 128, 64, 64. ReLU activation functions are used throughout.

## Solution and plot of rewards

![Plot of rewards][plot1]

This project is considered solved when the agent gets an average score of +13 over 100 consecutive episodes. In our training, this was achieved at episode 2900

## Next steps

If computational power was no obstacle, it would be interesting to perform a more rigourous search over the hyperparameter space, as well as searching a bit over network depth and width. It would also be interesting to see how performance improves with more advanced DQN variants (prioritized experience replay, dueling DQN, etc).
