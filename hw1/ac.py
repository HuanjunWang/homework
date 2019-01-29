import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam,SGD
import gym
import tensorflow as tf

ENVS = 'Humanoid-v2'
SIGMA = 5.
LEARNING_RATE = 1e-3

def verify_policy(env, policy_fun):
    class ARGS(object):
        pass

    args = ARGS()
    args.envname = ENVS
    args.max_timesteps = 2000
    args.num_rollouts = 10
    args.render = False

    #env = gym.make(args.envname)
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
    model.add(Dense(units=256, activation='relu', input_shape=input_shape))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=action_num, activation='linear'))

    adam = Adam(lr=LEARNING_RATE)
    sgd = SGD(lr=LEARNING_RATE)

    def maximum_likehood(y_true, mu):
        normal_d = tf.distributions.Normal(loc=mu, scale=SIGMA)
        return -tf.reduce_mean(tf.multiply(normal_d.log_prob(y_true[:, :-1]), tf.expand_dims(y_true[:, -1], 1)))

    model.compile(optimizer=adam, loss=maximum_likehood)
    return model


def make_value_model(input_shape):
    model = Sequential()
    model.add(Dense(units=256, activation='relu', input_shape=input_shape))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=1, activation='linear'))

    adam = Adam(lr=LEARNING_RATE)
    sgd = SGD(lr=LEARNING_RATE)
    model.compile(optimizer=adam, loss='mse')
    return model


def step1_sample(env, policy_fun, min_steps=500):
    states = []
    actions = []
    rewards = []
    next_states = []
    final_state = []

    steps = 0
    while steps < min_steps:
        s = env.reset()
        done = False
        while not done:
            action = policy_fun(s[None, :])
            n_s, r, done, _ = env.step(action)
            steps += 1

            states.append(s)
            actions.append(action)
            rewards.append(r)
            next_states.append(n_s)
            if done:
                final_state.append(0)
            else:
                final_state.append(1)

            s = n_s

    return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(final_state)


def ac():
    env = gym.make(ENVS)
    random_seed = 1
    env.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_random_seed(random_seed)

    input_shape = env.observation_space.shape
    action_num = env.action_space.shape[0]

    print("Input Shape:", input_shape)
    print("Action Number:", action_num)

    policy_model = make_gauss_model(input_shape, action_num)
    value_model = make_value_model(input_shape)

    def policy_fun_gauss(obs):
        mu = policy_model.predict(obs)
        return np.random.normal(mu, SIGMA)[0]

    for epoch in range(10000):
        print("epoch:", epoch)

        # Step 1, Sample
        current_states, actions, rewards, next_states, final_state = step1_sample(env, policy_fun_gauss, min_steps=50000)

        # Step 2, Fit the value model
        for _ in range(10):
            next_value = value_model.predict(np.array(next_states))
            target_value = np.array(rewards) + next_value.ravel() * final_state * .99
            value_model.fit(np.array(current_states), target_value, batch_size=128, epochs=1, verbose=0)

        # Step 3, improve the policy model
        next_v = value_model.predict(next_states).ravel()
        this_v = value_model.predict(current_states).ravel()
        advances = rewards + next_v * final_state - this_v
        advances = (advances - np.mean(advances)) / np.std(advances)

        policy_model.fit(current_states, np.hstack((actions, advances[:, None])), batch_size=128, epochs=1, verbose=2)

        def policy_fun_gauss2(obs):
            mu = policy_model.predict(obs)
            return np.random.normal(mu, .01)[0]

        if epoch % 100 == 0:
            verify_policy(env, policy_fun_gauss2)


if __name__ == '__main__':
    ac()
