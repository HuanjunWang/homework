import time

import gym

from multiprocessing import Queue, Pool, Process


class FastEnv(object):
    def __init__(self, env_name, env_num=100, process_num=8):
        self.envs = []

        self.obsevations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.dones = []
        self.env_num = env_num

        for _ in range(env_num):
            self.envs.append(gym.make(env_name))

        self.epochs = 0
        self.steps = 0

        self.steps_queue = Queue()
        self.info_queue = Queue()

        for _ in range(process_num):
            p = Process(target=self.env_running_process)
            p.daemon = True
            p.start()


    def env_running_process(self):
        while True:
            env, action = self.steps_queue.get()
            obs = env.env.state
            next_obs, reward, done, _ = env.step(action)
            self.info_queue.put((obs, action, reward, next_obs, done))

    def random_actions(self):
        actions = []
        for env in self.envs:
            actions.append(env.action_space.sample())
        return actions

    def reset(self):
        self.obsevations = []
        for env in self.envs:
            self.obsevations.append(env.reset())
        return self.obsevations

    def step(self, actions):
        for env, action in zip(self.envs, actions):
            self.steps_queue.put((env, action))

        self.next_observations = []
        self.rewards = []
        self.dones = []

        for i in range(self.env_num):
            obs, action, reward, next_obs, done = self.info_queue.get()
            self.next_observations.append(next_obs)
            self.rewards.append(reward)
            self.dones.append(done)
            self.steps += 1

        self.obsevations = []
        for env, done, next_obs in zip(self.envs, self.dones, self.next_observations):
            if done:
                self.obsevations.append(env.reset())
                self.epochs += 1
            else:
                self.obsevations.append(next_obs)

        return self.next_observations, self.rewards, self.dones

    def get_observation(self):
        return self.obsevations


if __name__ == "__main__":
    env = FastEnv("HalfCheetah-v2")
    env.reset()

    start = time.time()
    while env.steps < 100000:
        env.step(env.random_actions())
    print(env.epochs)
    print(time.time() - start)
