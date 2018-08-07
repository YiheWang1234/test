#!/anaconda3/bin/python3.6
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:43:29 2016

@author: Rob Romijnders
"""

import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.contrib.tensorboard.plugins import projector
from AE_ts_model_re import Model, open_data

"""Hyperparameters"""
direc = './'
LOG_DIR = './'
config = {}  # Put all configuration information into the dict
config['num_layers'] = 2  # number of layers of stacked RNN's
config['hidden_size'] = 90  # memory cells in a layer
config['max_grad_norm'] = 5  # maximum gradient norm during training
config['batch_size'] = batch_size = 64
config['learning_rate'] = .005
config['crd'] = 1  # Hyperparameter for future generalization
config['num_l'] = 1  # number of units in the latent space
lag = 10

plot_every = 10  # after _plot_every_ GD steps, there's console output
max_iterations = 10  # maximum number of iterations
dropout = 0.8  # Dropout rate
"""Load the data"""
X_train1, X_val1, y_train1, y_val1 = open_data('./UCR_TS_Archive_2015')
X_train1, X_train_out = X_train1[:, :-1], X_train1[:, -1]
X_val1, X_val_out = X_val1[:, :-1], X_val1[:, -1]

N = X_train1.shape[0]
Nval = X_val1.shape[0]
D1 = X_train1.shape[1]
config['sl'] = D = sl = lag  # sequence length
print('We have %s observations with %s dimensions' % (N, D1))

z_train = np.zeros((N-N%batch_size, D1-D))
z_test = np.zeros((Nval-Nval%batch_size, D1-D))
# sig_train = np.zeros((N-N%batch_size, D1-D))
# sig_test = np.zeros((Nval-Nval%batch_size, D1-D))

for j in range(D1 - D):

    """Training time!"""
    print("Current running ", j + 1, "AE of total ", D1 - D, " tasks")
    X_train = X_train1[:, j:(j + lag)]
    X_val = X_val1[:, j:(j + lag)]

    # Proclaim the epochs
    epochs = np.floor(batch_size * max_iterations / N)
    print('Train with approximately %d epochs' % epochs)

    model = Model(config)
    print(model)
    sess = tf.Session()
    perf_collect = np.zeros((2, int(np.floor(max_iterations / plot_every))))

    if True:
        # sess.run(model.init_op)
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(LOG_DIR, sess.graph)  # writer for Tensorboard

        step = 0  # Step is a counter for filling the numpy array perf_collect
        for i in range(max_iterations):
            batch_ind = np.random.choice(N, batch_size, replace=False)
            result = sess.run([model.loss, model.loss_seq, model.loss_lat_batch, model.train_step],
                              feed_dict={model.x: X_train[batch_ind], model.keep_prob: dropout})

            # if i % plot_every == 0:
            #     # Save train performances
            #     perf_collect[0, step] = loss_train = result[0]
            #     loss_train_seq, lost_train_lat = result[1], result[2]
            #
            #     # Calculate and save validation performance
            #     batch_ind_val = np.random.choice(Nval, batch_size, replace=False)
            #
            #     result = sess.run([model.loss, model.loss_seq, model.loss_lat_batch, model.merged],
            #                       feed_dict={model.x: X_val[batch_ind_val], model.keep_prob: 1.0})
            #     perf_collect[1, step] = loss_val = result[0]
            #     loss_val_seq, lost_val_lat = result[1], result[2]
            #     # and save to Tensorboard
            #     summary_str = result[3]
            #     writer.add_summary(summary_str, i)
            #     writer.flush()
            #
            #     print(
            #         "At %6s / %6s train (%5.3f, %5.3f, %5.3f), val (%5.3f, %5.3f,%5.3f) in order (total, seq, lat)" % (
            #             i, max_iterations, loss_train, loss_train_seq, lost_train_lat, loss_val, loss_val_seq,
            #             lost_val_lat))
            #     step += 1

    if True:
        ##Extract the latent space coordinates of the validation set
        start = 0
        label = []  # The label to save to visualize the latent space
        z_run = np.empty([0])
        x_out = []
        sigma_out = []

        while start + batch_size < Nval:
            run_ind = range(start, start + batch_size)
            z_mu_fetch = sess.run(model.z_mu, feed_dict={model.x: X_val[run_ind], model.keep_prob: 1.0})
            z_run = np.append(z_run, z_mu_fetch)

            decode_input = np.concatenate([z_mu_fetch, X_val[run_ind]], axis=1)
            x_out_fetch = (
                sess.run(model.h_mu, feed_dict={model.z_tmp: decode_input, model.keep_prob: 1.0})).transpose()
            x_out.append(x_out_fetch)

            sigma_out_fetch = (
                sess.run(model.h_sigma, feed_dict={model.z_tmp: decode_input, model.keep_prob: 1.0})).transpose()
            sigma_out.append(sigma_out_fetch)

            start += batch_size
        # np.concatenate(z_run, axis=0)
        z_test[:, j] = np.array(z_run)
        sigma_out = np.concatenate(sigma_out, axis=0)
        x_out = np.concatenate(x_out, axis=0)

    if True:
        ##Extract the latent space coordinates of the validation set
        start = 0
        z_t = np.empty([0])
        x_t = []
        sig_t = []

        while start + batch_size < N:
            # print(start)
            run_ind = range(start, start + batch_size)
            z_mu_fetch = sess.run(model.z_mu, feed_dict={model.x: X_train[run_ind], model.keep_prob: 1.0})
            z_t = np.append(z_t, z_mu_fetch)

            decode_input = np.concatenate([z_mu_fetch, X_train[run_ind]], axis=1)
            x_out_fetch = (
                sess.run(model.h_mu, feed_dict={model.z_tmp: decode_input, model.keep_prob: 1.0})).transpose()
            x_t.append(x_out_fetch)

            sigma_out_fetch = (
                sess.run(model.h_sigma, feed_dict={model.z_tmp: decode_input, model.keep_prob: 1.0})).transpose()
            sig_t.append(sigma_out_fetch)

            start += batch_size

        z_train[:, j] = z_t
        sig_t = np.concatenate(sig_t, axis=0)
        x_t = np.concatenate(x_t, axis=0)

    sess.close()


np.savez('output_rec_z', z_train=z_train, z_test=z_test,
         x_train=X_train1, x_test=X_val1,
         x_train_out=X_train_out, x_test_out=X_val_out,
         )
