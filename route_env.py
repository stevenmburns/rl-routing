import time, random
import numpy as np
from gym import Env, spaces




class OneLayer(Env):


    def __init__(self, n, m, tgt=(0,0)):
        self.n = n
        self.m = m

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Dict(
            { 'agent': spaces.MultiDiscrete((self.n, self.m)),
              'target': spaces.MultiDiscrete((self.n, self.m))
            }
        )

        self._actions = ['U', 'R', 'D', 'L']
        
        self._action_to_int = { k : idx for idx, k in enumerate(self._actions) }


        self._delta_tbl = {'U': np.array((-1, 0), dtype=np.int8),
                           'R': np.array((0, 1), dtype=np.int8),
                           'D': np.array((1, 0), dtype=np.int8),
                           'L': np.array((0, -1), dtype=np.int8)}
        
        self.set_agent_location((self.n-1, self.m-1))
        self.set_target_location(tgt)



    def _get_obs(self):
        return {'agent': self._agent_location, 'target': self._target_location}

    def _get_info(self):
        return None


    def _build_idx(self, i, j):
        return i * self.m + j

    def _get_agent_idx(self):
        return self._build_idx(self._agent_location[0], self._agent_location[1])

    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)

        while True:
            self.set_agent_location((self.np_random.randint(self.n), self.np_random.randint(self.m)))
            if self.dist_to_target() > 0:
                break

        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation


    def dist_to_target(self):
        return np.sum(np.abs(self._agent_location-self._target_location))

    def step(self, action):
        agent_location = self._agent_location
        delta = self._delta_tbl[self._actions[action]]

        new_agent_location = agent_location + delta

        i, j = agent_location[0], agent_location[1]
        di, dj = delta[0], delta[1]
        ii, jj = new_agent_location[0], new_agent_location[1]

        if i == 0 and di == -1 or i == self.n-1 and di == 1:
            ii = i

        if j == 0 and dj == -1 or j == self.m-1 and dj == 1:
            jj = j

        self.set_agent_location((ii, jj))
        done = (new_agent_location == self._target_location).all()
        reward = 0 if done else -1

        #print(f'agent_location: {agent_location} action: {self._actions[action]} new_agent_location: {self._agent_location} done: {done} reward: {reward}')

        return self._get_obs(), reward, done, self._get_info
        

    def set_agent_location(self, p):
        self._agent_location = np.array(p, dtype=np.int8)

    def set_target_location(self, p):
        self._target_location = np.array(p, dtype=np.int8)

def test_A():

    env = OneLayer(4, 4)

    def t( state, action, new_state, new_reward, new_done):
        env.set_agent_location(state)
        next_state, reward, done, _ = env.step(env._action_to_int[action])
        assert (next_state['agent'] == new_state).all() and reward == new_reward and done == new_done

    t((3, 3), 'R', (3, 3), -1, False)
    t((3, 3), 'U', (2, 3), -1, False)
    t((3, 3), 'L', (3, 2), -1, False)
    t((3, 3), 'D', (3, 3), -1, False)
    
    t((1, 0), 'R', (1, 1), -1, False)
    t((1, 0), 'U', (0, 0),  0, True)
    t((1, 0), 'L', (1, 0), -1, False)
    t((1, 0), 'D', (2, 0), -1, False)

def test_B():
    env = OneLayer(4, 4)

    env.set_agent_location((3,3))
    
    assert env.dist_to_target() == 6
