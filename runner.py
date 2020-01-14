from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
import time


def runner(
    env,
    agent,
    agent_name,
    solution_score,
    train=True,
    n_episodes=2000,
    max_t=1000,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
    round_num=0,
    update_plot_every=10
):
    t0 = time.time()

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    scores_rolling_avg = []
    steps_in_episode = []

    eps = eps_start  # initialize epsilon
    elapsed_time = []
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=train)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0 
        for t in range(max_t):
            action = agent.act(state, eps)                 # select an action
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished 
            score += reward                                # update the score
            
            if train:
                agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
                

        steps_in_episode.append(t)

        print("episode lasted {} steps".format(t))
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        rolling_avg = np.mean(scores_window)
        scores_rolling_avg.append(rolling_avg)
        elapsed_time.append(time.time() - t0)

        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        if i_episode % update_plot_every == 0:
            # plot it
            plt.cla()
            titleStr = "{} -- round {}\nEpisode {}, ({:.2f}s, {:.2f}s/episode)\nAverage Score: {:.2f}".format(
                agent_name,
                round_num,
                i_episode,
                time.time() - t0,
                (time.time() - t0) / i_episode,
                rolling_avg,
            )
            plt.title(titleStr)
            plt.plot(scores)
            plt.plot(scores_rolling_avg)
            plt.ylabel("Score")
            plt.xlabel("Episode #")
            plt.xlim(0, n_episodes)  # consistent scale
            display.clear_output(wait=True)
            display.display(plt.gcf())
    if np.mean(scores_window) >= solution_score:
        print(
            "\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                i_episode - 100, np.mean(scores_window)
            )
        )
        torch.save(agent.qnetwork_local.state_dict(), "bananas_ddqn_naive-prioritized-replay.pth")
    return (scores, elapsed_time, steps_in_episode, agent.qnetwork_local.state_dict())
