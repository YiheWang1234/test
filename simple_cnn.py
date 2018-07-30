#!/anaconda3/bin/python3.6
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./tmp/data", one_hot=True)

learning_rate = 0.01
batch_size = 200
num_steps = 500

display_step = 500
examples_to_show = 10

num_hiden = 256
num_input = 784

X = tf.placeholder("float", [None, num_input])
y = tf.placeholder("float", [None, 10])

w = {
    'w1': tf.Variable(tf.random_normal([num_input, num_hiden])),
    'w2': tf.Variable(tf.random_normal([num_hiden, num_hiden])),
    'w3': tf.Variable(tf.random_normal([num_hiden, num_hiden])),
    'w4': tf.Variable(tf.random_normal([num_hiden, 10])),

    'b1': tf.Variable(tf.random_normal([num_hiden])),
    'b2': tf.Variable(tf.random_normal([num_hiden])),
    'b3': tf.Variable(tf.random_normal([num_hiden])),
    'b4': tf.Variable(tf.random_normal([10]))
}


def layer(X):
    l1 = tf.nn.sigmoid(tf.add(tf.matmul(X, w['w1']), w['b1']))
    l2 = tf.nn.sigmoid(tf.add(tf.matmul(l1, w['w2']), w['b2']))
    l3 = tf.nn.sigmoid(tf.add(tf.matmul(l2, w['w3']), w['b3']))
    l4 = tf.nn.sigmoid(tf.add(tf.matmul(l3, w['w4']), w['b4']))
    return l4


y_pred = layer(X)
loss = tf.reduce_sum(tf.pow(y - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
init = tf.global_variables_initializer()

print("start training and testing")

with tf.Session() as sess:
    sess.run(init)

    #train
    for i in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x,
                                                      y: batch_y})

        if i % display_step == 0 or i == 1:
            print("Step %i: Minibatch Loss: %f" % (i, l))

    #test
    acc = 0
    for i in range(1, num_steps+1):
        batch_x, batch_y = mnist.test.next_batch(batch_size)
        g = sess.run(y_pred, feed_dict={X: batch_x})
        acc += sum(sum(batch_y - g))/batch_size
    print("the accuracy of cnn is %s" % (acc))
    print(g)
    print(np.sum(g, axis=0))
        

