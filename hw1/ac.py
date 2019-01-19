import pickle
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import gym

import tensorflow as tf

ENVS = 'Humanoid-v2'
SIGMA = 5.


def verify_policy(policy_fun):
    class ARGS(object):
        pass

    args = ARGS()
    args.envname = ENVS
    args.max_timesteps = 1000
    args.num_rollouts = 10
    args.render = False

    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    for i in range(args.num_rollouts):
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy_fun(obs[None, :])
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


def make_gauss_model(input_shape, action_num):
    model = Sequential()
    model.add(Dense(units=128, activation='relu', input_shape=input_shape))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=action_num, activation='linear'))

    adam = Adam(lr=1e-2)

    def maximum_likehood(y_true, mu):
        normal_d = tf.distributions.Normal(loc=mu, scale=SIGMA)
        return -tf.reduce_mean(tf.multiply(normal_d.log_prob(y_true[:, :-1]), tf.expand_dims(y_true[:, -1], 1)))

    model.compile(optimizer=adam, loss=maximum_likehood)
    return model


def make_value_model(input_shape):
    model = Sequential()
    model.add(Dense(units=128, activation='relu', input_shape=input_shape))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=1, activation='linear'))

    adam = Adam(lr=1e-3)
    model.compile(optimizer=adam, loss='mse')
    return model


def ac():
    env = gym.make(ENVS)
    input_shape = env.observation_space.shape
    # input_shape= (input_shape[0])
    action_num = env.action_space.shape[0]
    policy_model = make_gauss_model(input_shape, action_num)
    value_model = make_value_model(input_shape)

    def policy_fun_gauss(obs):
        mu = policy_model.predict(obs)
        return np.random.normal(mu, SIGMA)

    for epoch in range(1000):
        print("epoch:", epoch)
        # Step 1, Sample
        observations = []
        actions = []
        rewards_to_go = []
        action_with_advance = []

        for i in range(100):
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            rewards = []
            while not done:
                action = policy_fun_gauss(obs[None, :])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                rewards.append(r)
                totalr += r
                steps += 1

            for j in range(len(rewards)):
                rewards_to_go.append(sum(rewards[j:]))

        # Step 2, Fit the value model
        value_model.fit(np.array(observations), np.array(rewards_to_go), batch_size=128, epochs=10, verbose=0)

        # Step 3, improve the policy model
        for (obs, action, reward) in zip(observations, actions, rewards_to_go):
            baseline = value_model.predict(obs[np.newaxis])[0, 0]
            advance = reward - baseline
            action_with_advance.append(np.append(action, advance))

        policy_model.fit(np.array(observations), np.array(action_with_advance), batch_size=128, epochs=1, verbose=2)

        if epoch % 10 == 0:
            verify_policy(policy_fun_gauss)


if __name__ == '__main__':
    ac()
