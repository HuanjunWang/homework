import tensorflow as tf
import numpy as np



def main():
    a = tf.constant(shape=[3,4], value=np.arange(0,12))
    b = tf.reduce_max(a, axis=1)



    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print(sess.run(b))





if __name__ == '__main__':
    main()