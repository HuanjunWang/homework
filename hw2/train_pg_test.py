from train_pg_f18 import *

import tensorflow as tf


def main():
    input_placeholder = tf.placeholder(tf.float32, shape=[None, 11])
    output = build_mlp(input_placeholder, 8, "FULLCONNECTED", 3, 128, activation=tf.tanh, output_activation=None)
    writer = tf.summary.FileWriter('/tmp/tf')
    writer.add_graph(tf.get_default_graph())


def sample():
    dist = tf.distributions.Categorical(probs=[[.4, .5, .1], [0., 0., 1.]])
    s = dist.sample()
    print(s)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(s))


def random_sample():
    r = tf.random_normal(shape=[2, 3])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(r))


def log_prob():
    mu = [[2., 1., 1.], [3., 4., 1.]]
    sigma = [[5., 5., 5.]]
    dist = tf.distributions.Normal(mu, sigma)
    log_p = dist.log_prob([[1., 1., 1.], [1., 1., 1.]])
    log_p = tf.reduce_sum(log_p, axis=-1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(log_p))


def multiply_broadcast():
    a = tf.constant([[1., 1.], [2., 2.], [4., 4.]])
    b = tf.constant([2., 2.])
    c = a * b + 1

    return c


def rewards():
    r = np.ones([10])
    print(r)

    result = []
    result.append(r[-1])
    for i in r[-2::-1]:
        result.append(i + result[-1] * .9)

    print(result)


def run(func):
    target = func()
    print(target)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(target))

if __name__ == '__main__':
    rewards()
