import time, random
import numpy as np
from itertools import product

from route_env import OneLayer


env = OneLayer(8, 8)

env.reset()

q_table = np.zeros(((env.n*env.m), env.action_space.n))

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

training_steps = 100000

for iter in range(training_steps):
    state = env.reset()

    epoch = 0
    done = False
    
    while not done:
        agent = state['agent']
        idx = env._get_agent_idx()

        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[idx]) # Exploit learned values

        next_state, reward, done, _ = env.step(action) 
        next_agent = next_state['agent']
        next_idx = env._get_agent_idx()
        
        old_value = q_table[idx, action]
        next_max = np.max(q_table[next_idx])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[idx, action] = new_value

        state = next_state
        epoch += 1
        
    if iter % 100 == 0:
        print('\033c', end='')
        print(f"Episode: {iter}")
        for i in range(env.n):
            for j in range(env.m):
                idx = i * env.m + j
                print(f'{env._actions[np.argmax(q_table[idx])]}', end='')
            print()



print("Training finished.\n")

"""Evaluate agent's performance after Q-learning"""

for i, j in product(range(env.n), range(env.m)):
    idx = env._build_idx(i, j)
    print(f'i,j: {i},{j}', end='')
    for action_idx, q in enumerate(q_table[idx]):
        print(f' {env._actions[action_idx]}: {q:.6f}', end='')
    print(f' => {env._actions[np.argmax(q_table[idx])]}', end='')
    lb = min(q_table[idx][env._action_to_int['R']],q_table[idx][env._action_to_int['D']])
    ub = max(q_table[idx][env._action_to_int['U']],q_table[idx][env._action_to_int['L']])
    print(f' {ub-lb:.6f} {(ub-lb)/lb*100:.2f}%', end='')
    print()

for i in range(env.n):
    for j in range(env.m):
        idx = env._build_idx(i, j)
        print(f'{env._actions[np.argmax(q_table[idx])]}', end='')
    print()

total_epochs = 0
episodes = 10000

for _ in range(episodes):
    state = env.reset()
    epoch = 0
    
    done = False
    
    agent = state['agent']
    expected_epochs = agent[0] + agent[1]

    #print( f'starting at: {agent} expected_epochs: {expected_epochs}')

    while not done:
        action = np.argmax(q_table[env._get_agent_idx()])
        state, reward, done, _ = env.step(action)
        #print( f'epoch: {epoch} state: {state} reward: {reward} done: {done}')
        epoch += 1

    assert expected_epochs == 0 or expected_epochs == epoch, (expected_epochs, epoch)

    total_epochs += epoch

print(f"Results after {episodes} episodes:")
if episodes > 0:
    print(f"Average timesteps per episode: {total_epochs / episodes}")
