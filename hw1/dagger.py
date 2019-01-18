import pickle
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import gym
from load_policy import *

ENVS = 'Ant-V2'


class ARGS(object):
    def __init__(self):
        self.envname = ENVS
        self.expert_policy_file = 'experts/%s.pkl' % ENVS
        self.max_timesteps = 1000
        self.num_rollouts = 10
        self.render = True


def test_policy(policy_fun, args):
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(args.num_rollouts):
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy_fun(obs[None, :])
            observations.append(obs)
            actions.append(action)

            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    return observations, actions


def make_model(input_shape, action_shape):
    model = Sequential()
    model.add(Dense(units=128, activation='relu', input_shape=input_shape))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=action_shape, activation='linear'))

    adam = Adam(lr=1e-3)
    model.compile(optimizer=adam, loss='mse', metrics='mae')
    return model


def dagger(epoch):
    env = gym.make(ENVS)
    input_shape = env.observation_space
    action_shape = env.action_space
    model = make_model(input_shape, action_shape)

    def policy_fun(obs):
        return model.predict(obs)

    # Now the model is totally random
    dagger_data = []
    dagger_label = []

    args = ARGS()
    args.num_rollouts = 1
    for i in range(epoch):
        # run the env to get the dagger data
        new_data, _ = test_policy(policy_fun, args)
        expert_fn = load_policy.load_policy(args.expert_policy_file)
        _, new_label = test_policy(expert_fn, args)

        # run the expert to get the dagger label
        dagger_data.extend(new_data)
        dagger_label.extend(new_label)
        model.fit(dagger_data, dagger_label, batch_size=128, epochs=40, verbose=1)

    test_policy(policy_fun)


if __name__ == '__main__':
    dagger()
