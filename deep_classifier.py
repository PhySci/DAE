import pandas as pd
import numpy as np
import tensorflow as tf
import os
import time
from sklearn.metrics import roc_auc_score
import logging


class DeepClassifier:

    def __init__(self, dae_lr=0.001, nn_lr=0.01, dae_decay=0.99, nn_decay=0.99, L2 = 0.005, batch_size=512,
                 features=345, dae_hidden = [1500, 1500, 1500], clf_hidden = [1000, 1000],
                 restart=False, verbose=True, keep_prob=1, name = './model/DAE_model',
                 cat_features=[], bin_features = [], num_features=[], ohe = None):
        """
        Create an instance
        :param dae_lr:
        :param nn_lr:
        :param features:
        :param dae_hidden:
        :param clf_hidden:
        :param decay:
        :param momentum:
        :param restart:
        :param verbose:
        :param dropout:
        """

        # Training Parameters
        self.dae_learning_rate = dae_lr
        self.nn_learning_rate = nn_lr
        self.batch_size = batch_size
        self.dae_decay = dae_decay
        self.nn_decay = nn_decay
        self.momentum = 0
        self.L2 = L2
        self.verbose = True
        self.keep_prob = keep_prob
        self.dae_epoch = 12
        self.clf_epoch = 12

        self.cat_features = cat_features
        self.num_features = num_features
        self.bin_features = bin_features
        self.ohe = ohe

        self.pth = name

        self.dae_size= [features]+ dae_hidden + [features]
        self.nn_size = [np.array(self.dae_size[1:-1]).sum()] + clf_hidden + [1]

        tf.reset_default_graph()
        self.inp_features = tf.placeholder(tf.float32, [None, self.dae_size[0]], name='inp_features')
        self.ref_features = tf.placeholder(tf.float32, [None, self.dae_size[0]], name='ref_features')
        self.labels = tf.placeholder(tf.int8, [None, 1], name='labels')
        self.keep_prob_ph = tf.placeholder(tf.float32, [], name='keep_prob_ph')

        self.graph = self.build_graph(self.inp_features)

        self.dae_loss = self.get_dae_loss(self.ref_features)
        self.dae_optimizer = self.get_dae_optimizer()

        self.clf_loss = self.get_clf_loss(self.labels)
        self.clf_optimizer = self.get_clf_optimizer()

        self.saver = tf.train.Saver()
        logging.basicConfig(filename='DAE.log', level=logging.DEBUG, format='%(message)s')
        self.logger = logging.getLogger('DAE_classifier')

        if restart:
            init_op = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init_op)
                self.saver.save(sess, self.pth)
                self.logger.debug('Init and save the model')

    def get_dae_loss(self, ref_x):
        with tf.name_scope('dae_loss'):
            x_dae = tf.get_default_graph().get_tensor_by_name("DAE/layer3/output:0")
            loss = tf.reduce_mean(tf.pow(ref_x - x_dae, 2, name='pow2'), name='loss')
        return loss

    def get_dae_optimizer(self):
        with tf.name_scope('dae_optimizer'):
            return tf.train.RMSPropOptimizer(self.dae_learning_rate, decay=self.dae_decay, momentum=self.momentum,
                                             name='dae_optimizer_op'). \
                minimize(self.dae_loss, var_list=tf.get_collection('DAE'))

    def train_dae(self, x, noise=0.1):
        """
        Train Denoising Auto Encoder
        :param x: dataset
        :param noise: noise level
        :return:
        """
        noise_list = np.linspace(0.001, 0.1, self.dae_epoch)
        #np.random.seed(seed=int(time.time()))
        batch_num = np.floor(x.shape[0] / self.batch_size).astype(int)

        with tf.Session() as sess:
            self.saver.restore(sess, self.pth)
            train_writer = tf.summary.FileWriter('./train/dae', sess.graph)

            for epoch in range(self.dae_epoch): #enumerate(noise_list):
                print('Epoch: {epoch}'.format(epoch=epoch))
                loss = list()
                for i, ids in enumerate(np.array_split(x.index.tolist(), batch_num)):
                    print(i)
                    batch, noise_batch = self.get_corrupted_data(x.loc[ids, :], noise_level=noise)
                    noise_std = np.mean(np.abs(batch-noise_batch))
                    feed_dict = {self.ref_features: batch,
                                 self.inp_features: noise_batch,
                                 self.keep_prob_ph: 1}
                    _, t = sess.run([self.dae_optimizer, self.dae_loss], feed_dict=feed_dict)
                    loss.append(t)

                # add info
                dae_summary = tf.Summary()
                dae_summary.value.add(tag='DAE_train/loss_mean', simple_value=np.array(loss).mean())
                dae_summary.value.add(tag='DAE_train/loss_std', simple_value=np.array(loss).std())
                dae_summary.value.add(tag='DAE_train/noise', simple_value=noise)
                dae_summary.value.add(tag='DAE_train/noise_diff', simple_value=noise_std)

                train_writer.add_summary(sess.run(tf.summary.merge_all(), feed_dict=feed_dict), epoch)
                train_writer.add_summary(dae_summary, epoch)
                
                if epoch % 10 == 0:
                    self.logger.debug(self.saver.save(sess, self.pth))

            print(self.saver.save(sess, self.pth))

    def get_clf_loss(self, y):
        with tf.name_scope('clf_loss'):
            y_pr = tf.get_default_graph().get_tensor_by_name("NN/layer2/output:0")
            loss_sample =  tf.reduce_mean(tf.losses.log_loss(y, y_pr), name='clf_loss')
            loss_reg = tf.nn.l2_loss(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            return loss_sample+self.L2*loss_reg

    def get_clf_optimizer(self):
        with tf.name_scope('clf_optimizer'):
            return tf.train.RMSPropOptimizer(self.nn_learning_rate, decay=self.nn_decay, momentum=self.momentum,
                                             name='clf_optimizer_op'). \
                minimize(self.clf_loss, var_list=tf.get_collection('clf'))

    def train_clf(self, X, y, X_val=None, y_val=None, restart=False, L2=0.005):
        """
        Train classification subpart of the model
        :param X: train dataset
        :param y: train labels
        :param X_val: validation dataset
        :param y_val: validation labels
        :param restart: Should the classifier be reinitialized?
        :param L2: regularization term
        :return:
        """
        self.logger.debug('Classifier training')
        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter('./train/clf', sess.graph)
            self.saver.restore(sess, self.pth)
            if restart:
                print('Reinit classifier')
                tf.initialize_variables(tf.get_collection('clf'))
            for epoch in range(self.clf_epoch):
                loss = []
                for i in range(np.ceil(X.shape[0] / self.batch_size).astype(int)):
                    batch_x, batch_y = self.get_train_batch(X, y, batch_size=self.batch_size)
                    feed_dict = {self.inp_features: batch_x,
                                 self.labels: batch_y,
                                 self.keep_prob_ph: self.keep_prob}
                    _, t = sess.run([self.clf_optimizer, self.clf_loss], feed_dict=feed_dict)
                    loss.append(t)
                # add info
                clf_summary = tf.Summary()
                clf_summary.value.add(tag='clf_train/loss_mean', simple_value=np.array(loss).mean())
                clf_summary.value.add(tag='clf_train/loss_std', simple_value=np.array(loss).std())
                clf_summary.value.add(tag='clf_train/keep_prob', simple_value=self.keep_prob)
                train_writer.add_summary(sess.run(tf.summary.merge_all(), feed_dict=feed_dict), epoch)
                train_writer.add_summary(clf_summary, epoch)

                if True: #(epoch % 2 ==0) or epoch==(self.clf_epoch-1):
                    val_set, val_label = self.get_train_batch(X_val, y_val)

                    y_pr = sess.run(self.graph, feed_dict={self.inp_features: val_set, self.keep_prob_ph: 1})
                    msg = 'ROC-AUC is {v: 2.3f}'.format(v=roc_auc_score(y_val, y_pr))
                    self.show_msg(msg)
                    self.logger.debug(self.saver.save(sess, self.pth))

            print(self.saver.save(sess, self.pth))

    def build_graph(self, inp):
        # Define DAE
        with tf.name_scope('DAE'):
            with tf.name_scope('layer0'):
                dae_w0 = tf.get_variable(shape=[self.dae_size[0], self.dae_size[1]], name='dae_w0')
                dae_b0 = tf.get_variable(shape=[self.dae_size[1]], name='dae_b0')
                dae_layer0 = tf.nn.relu(tf.add(tf.matmul(inp, dae_w0), dae_b0), name='output')
                tf.add_to_collection("DAE", dae_w0)
                tf.add_to_collection("DAE", dae_b0)

                tf.summary.histogram('dae_w0', dae_w0)
                tf.summary.histogram('dae_b0', dae_b0)
                tf.summary.histogram('dae_layer0', dae_layer0)

            with tf.name_scope('layer1'):
                dae_w1 = tf.get_variable(shape=[self.dae_size[1], self.dae_size[2]], name='dae_w1')
                dae_b1 = tf.get_variable(shape=[self.dae_size[2]], name='dae_b1')
                dae_layer1 = tf.nn.relu(tf.add(tf.matmul(dae_layer0, dae_w1), dae_b1), name='output')
                tf.add_to_collection("DAE", dae_w1)
                tf.add_to_collection("DAE", dae_b1)

                tf.summary.histogram('dae_w1', dae_w1)
                tf.summary.histogram('dae_b1', dae_b1)
                tf.summary.histogram('dae_layer1', dae_layer1)

            with tf.name_scope('layer2'):
                dae_w2 = tf.get_variable(shape=[self.dae_size[2], self.dae_size[3]], name='dae_w2')
                dae_b2 = tf.get_variable(shape=[self.dae_size[3]], name='dae_b2')
                dae_layer2 = tf.nn.relu(tf.add(tf.matmul(dae_layer1, dae_w2), dae_b2), name='output')
                tf.add_to_collection("DAE", dae_w2)
                tf.add_to_collection("DAE", dae_b2)

                tf.summary.histogram('dae_w2', dae_w2)
                tf.summary.histogram('dae_b2', dae_b2)
                tf.summary.histogram('dae_layer2', dae_layer2)

            with tf.name_scope('layer3'):
                dae_w3 = tf.get_variable(shape=[self.dae_size[3], self.dae_size[4]], name='dae_w3')
                dae_b3 = tf.get_variable(shape=[self.dae_size[4]], name='dae_b3')
                dae_layer3 = tf.add(tf.matmul(dae_layer2, dae_w3), dae_b3, name='output')
                tf.add_to_collection("DAE", dae_w3)
                tf.add_to_collection("DAE", dae_b3)

                tf.summary.histogram('dae_w3', dae_w3)
                tf.summary.histogram('dae_b3', dae_b3)
                tf.summary.histogram('dae_layer3', dae_layer3)

        # Define Neural Network Classifier
        with tf.name_scope('NN'):
            nn_inp = tf.concat([dae_layer0, dae_layer1, dae_layer2], axis=1, name='concat')
            regularizer = tf.contrib.layers.l2_regularizer(scale=1.0)
            with tf.name_scope('layer0'):
                nn_b0 = tf.get_variable(shape=[self.nn_size[1]], name='clf_b0')
                nn_w0 = tf.get_variable(shape=[self.nn_size[0], self.nn_size[1]], name='clf_w0', regularizer=regularizer)
                a0 = tf.nn.relu(tf.add(tf.matmul(nn_inp, nn_w0), nn_b0), name='output')
                nn_layer0 = tf.nn.dropout(a0, self.keep_prob_ph, name='dropout_0')
                tf.add_to_collection("clf", nn_w0)
                tf.add_to_collection("clf", nn_b0)
                tf.summary.histogram('nn_w0', nn_w0)
                tf.summary.histogram('nn_b0', nn_b0)
                tf.summary.histogram('nn_layer0', a0)
            with tf.name_scope('layer1'):
                nn_b1 = tf.get_variable(shape=[self.nn_size[2]], name='clf_b1')
                nn_w1 = tf.get_variable(shape=[self.nn_size[1], self.nn_size[2]], name='clf_w1', regularizer=regularizer)
                a1 = tf.nn.relu(tf.add(tf.matmul(nn_layer0, nn_w1), nn_b1), name='output')
                nn_layer1 = tf.nn.dropout(a1, keep_prob=self.keep_prob_ph, name='dropout_1')
                tf.add_to_collection("clf", nn_w1)
                tf.add_to_collection("clf", nn_b1)
                tf.summary.histogram('nn_w1', nn_w1)
                tf.summary.histogram('nn_b1', nn_b1)
                tf.summary.histogram('nn_layer1', a1)
            with tf.name_scope('layer2'):
                nn_b2 = tf.get_variable(shape=[self.nn_size[3]], name='clf_b2')
                nn_w2 = tf.get_variable(shape=[self.nn_size[2], self.nn_size[3]], name='clf_w2', regularizer=regularizer)
                a2 = tf.nn.sigmoid(tf.add(tf.matmul(nn_layer1, nn_w2), nn_b2), name='output')
                y_pr = tf.nn.dropout(a2, keep_prob=self.keep_prob_ph, name='dropout_2')
                tf.add_to_collection("clf", nn_w2)
                tf.add_to_collection("clf", nn_b2)
                tf.summary.histogram('nn_w2', nn_w2)
                tf.summary.histogram('nn_b2', nn_b2)
                tf.summary.histogram('nn_layer2', a2)
        return y_pr

    def get_corrupted_data(self, df, noise_level=0.1):
        """
        Return corrupted training minibatch
        :param df: original dataset
        :param noise_level: corruption degree
        :return:
        """

        # set random seed
        np.random.seed(seed=int(time.time()))

        # Get random subset
        arr_size = df.shape
        original_num_features = df[self.num_features].values
        original_bin_features = df[self.bin_features].values
        original_batch = np.concatenate((original_num_features,
                                         original_bin_features,
                                         self.ohe.transform(df[self.cat_features].values)),
                                        axis=1)

        # add noise to numerical features
        noisy_num_features = original_num_features.copy() + noise_level*np.random.rand(arr_size[0],len(self.num_features))

        # add noise to binary features
        noisy_bin_features = original_bin_features.copy() + np.random.choice(a=[0, 1],
                                                     size=(arr_size[0], len(self.bin_features)),
                                                     p=[1-noise_level, noise_level])
        noisy_bin_features = noisy_bin_features % 2

        # add noise to categorical features
        noisy_cat_features = df[self.cat_features].values

        n_cat_features = len(self.cat_features)
        n_perm = int(noise_level*self.batch_size*n_cat_features)
        cols = list(np.random.randint(0, n_cat_features, n_perm))
        rows = list(np.random.randint(0, arr_size[0], n_perm))
        new_rows = list(np.random.randint(0, arr_size[0], n_perm))
        noisy_cat_features[new_rows, cols] = noisy_cat_features[rows, cols]

        noisy_batch = np.concatenate((noisy_num_features,
                                      noisy_bin_features,
                                      self.ohe.transform(noisy_cat_features)),
                                     axis=1)

        return original_batch, noisy_batch

    def predict(self, X):
        with tf.Session() as sess:
            self.saver.restore(sess, self.pth)
            y_pr = sess.run(self.graph, feed_dict={self.inp_features: X, self.keep_prob_ph: 1})
        return y_pr

    def show_msg(self, msg):
        self.logger.debug(msg)
        if self.verbose:
            print(msg)

    def get_batch(self, x, y, i):
        """

        :param x: features dataset
        :param y: labels
        :param i: chunk number
        :return:
        """


    def get_train_batch(self, df, labels, batch_size=None):

        if batch_size is not None:
            idx = np.random.randint(0, df.shape[0], batch_size)
            X = df.iloc[idx]
            y = labels.iloc[idx]
        else:
            X = df
            y = labels

        val_num_features = X[self.num_features]
        val_bin_features = X[self.bin_features]
        val_cat_features = self.ohe.transform(X[self.cat_features])
        batch = np.concatenate((val_num_features, val_bin_features, val_cat_features), axis=1)

        return batch, y[:, np.newaxis]




