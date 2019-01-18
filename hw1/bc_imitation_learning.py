import pickle
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

ENVS = 'Humanoid-v2'


def verify_policy(policy_fun):
    class ARGS(object):
        pass

    args = ARGS()
    args.envname = ENVS
    args.max_timesteps = 1000
    args.num_rollouts = 10
    args.render = True

    import gym
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    for i in range(args.num_rollouts):
        print('iter', i)
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


def load_expert_data():
    data_file = 'expert_data/%s.pkl' % ENVS
    with open(data_file, 'rb') as f:
        data = pickle.loads(f.read())
    return data['observations'], data['actions']


def make_model(input_shape, action_num):
    model = Sequential()
    model.add(Dense(units=128, activation='relu', input_shape=input_shape))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=action_num, activation='linear'))

    adam = Adam(lr=1e-3)
    model.compile(optimizer=adam, loss='mse', metrics=['mae'])
    return model


def imitation():
    test_data, test_label = load_expert_data()
    test_label = test_label.reshape((test_label.shape[0], test_label.shape[-1]))
    input_shape = test_data.shape[1:]
    action_num = test_label.shape[-1]
    model = make_model(input_shape, action_num)

    model.fit(test_data, test_label, batch_size=128, epochs=100, verbose=2)

    def policy_fun(obs):
        return model.predict(obs)

    verify_policy(policy_fun)


if __name__ == '__main__':
    imitation()
