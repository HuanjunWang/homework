import time

import gym

env = gym.make("HalfCheetah-v2")
env.reset()

start = time.time()

steps = 0
while steps < 100000:
    _, _, done, _ = env.step(env.action_space.sample())
    if done:
        env.reset()

    steps += 1

print(time.time() - start)