import tensorflow as tf
import numpy as np

def model(scope):
    with tf.name_scope("ns_%s"%scope):
        tf.placeholder(tf.float32, shape=[None], name="Placeholder")
        with tf.variable_scope(scope):
            a = tf.get_variable("a", shape=[3,4])
            b = tf.get_variable("b", shape=[2,2])

def main():
    model('new')
    model('old')

    n = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'new')
    o = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'old')

    new2old = [tf.assign(nv, ov) for nv,ov in zip(n, o)]

    writer = tf.summary.FileWriter("/tmp/tftry")
    writer.add_graph(tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print(sess.run(n))
        print(sess.run(o))
        sess.run(new2old)


        print(sess.run(n))
        print(sess.run(o))

if __name__ == '__main__':
    main()