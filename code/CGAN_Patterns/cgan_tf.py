import tensorflow as tf
import numpy as np
import pandas as pd
import sys

from CGAN_Patterns.hyperparam import Hyperparam


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def weights_init(size):
    if Hyperparam.WEIGHTS_INIT == 'xavier':
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)
    elif Hyperparam.WEIGHTS_INIT == 'normal':
        return tf.random_normal(shape=size, stddev=Hyperparam.STDDEV)
    else:
        exit('ERROR initializing weights!')

def xe(data_in, loss_target='maximize'):
    y = 0 if loss_target == 'maximize' else 1
    F_loss = -tf.reduce_mean(y * tf.log(data_in) + (1 - y) * tf.log(1 - data_in))
    return F_loss


def xe_reg(data_in, lamda, theta_D, loss_target='maximize'):
    MODE=Hyperparam.MODE
    y = 0 if loss_target == 'maximize' else 1
    f_loss = -tf.reduce_mean(y * tf.log(data_in) + (1 - y) * tf.log(1 - data_in))
    # f_reg=np.zeros(4)
    if MODE == 'lip-log':
        f_reg = (lamda / 2) * (tf.log(tf.reduce_sum(theta_D[0] ** 2)) + tf.log(tf.reduce_sum(theta_D[1] ** 2)) +
                               tf.log(tf.reduce_sum(theta_D[2] ** 2)) + tf.log(tf.reduce_sum(theta_D[3] ** 2)))
    elif MODE == 'lip-mul':
        f_reg = lamda * tf.sqrt(tf.reduce_sum(theta_D[0] ** 2)) * tf.sqrt(tf.reduce_sum(theta_D[1] ** 2)) * tf.sqrt(
            tf.reduce_sum(theta_D[2] ** 2)) * tf.sqrt(tf.reduce_sum(theta_D[3] ** 2))

    elif MODE == 'lip-sum':
        f_reg = lamda * (tf.reduce_sum(theta_D[0] ** 2) + tf.reduce_sum(theta_D[1] ** 2) +
                         tf.reduce_sum(theta_D[2] ** 2) + tf.reduce_sum(theta_D[3] ** 2))

    else:
        sys.exit('Error in selecting MODE!!')

    net_loss = f_loss + f_reg
    return net_loss


class Cgan_tf():
    def __init__(self,paths, preprocessor):
        self.paths=paths
        X_dim=preprocessor.X_dim
        y_dim=preprocessor.y_dim
        D_H_DIM=Hyperparam.D_H_DIM
        Z_DIM=Hyperparam.Z_DIM
        G_H_DIM=Hyperparam.G_H_DIM
        self.X = tf.placeholder(tf.float32, shape=[None, X_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, y_dim])
        self.Z = tf.placeholder(tf.float32, shape=[None, Z_DIM])

        """ Discriminator parameters """
        D_W1 = tf.Variable(weights_init([X_dim + y_dim, D_H_DIM[0]]))
        D_b1 = tf.Variable(tf.zeros(shape=[D_H_DIM[0]]))
        D_W2 = tf.Variable(weights_init([D_H_DIM[0], D_H_DIM[1]]))
        D_b2 = tf.Variable(tf.zeros(shape=[D_H_DIM[1]]))
        D_W3 = tf.Variable(weights_init([D_H_DIM[1], D_H_DIM[0]]))
        D_b3 = tf.Variable(tf.zeros(shape=[D_H_DIM[0]]))
        D_W4 = tf.Variable(weights_init([D_H_DIM[0], 1]))
        D_b4 = tf.Variable(tf.zeros(shape=[1]))
        self.theta_D = [D_W1, D_W2, D_W3, D_W4, D_b1, D_b2, D_b3, D_b4]

        """ Generator parameters """
        G_W1 = tf.Variable(weights_init([Z_DIM + y_dim, G_H_DIM[0]]))
        G_b1 = tf.Variable(tf.zeros(shape=[G_H_DIM[0]]))
        G_W2 = tf.Variable(weights_init([G_H_DIM[0],G_H_DIM[1]]))
        G_b2 = tf.Variable(tf.zeros(shape=[G_H_DIM[1]]))
        G_W3 = tf.Variable(weights_init([G_H_DIM[1], G_H_DIM[0]]))
        G_b3 = tf.Variable(tf.zeros(shape=[G_H_DIM[0]]))
        G_W4 = tf.Variable(weights_init([G_H_DIM[0], X_dim]))
        G_b4 = tf.Variable(tf.zeros(shape=[X_dim]))
        self.theta_G = [G_W1, G_W2, G_W3, G_W4, G_b1, G_b2, G_b3, G_b4]

        """ Losses and Solvers """
        G_sample = self.generator(self.Z, self.y)
        D_real, D_logit_real = self.discriminator(self.X, self.y)
        D_fake, D_logit_fake = self.discriminator(G_sample, self.y, reuse=True)
        self.solver(self.X,self.y,G_sample,D_real,D_logit_real, D_fake, D_logit_fake,self.theta_D, self.theta_G)

        """ Model Saver """
        self.saver = tf.train.Saver()

        """MMD Sigmas """
        self.sigmas = [
            1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
            1e3, 1e4, 1e5, 1e6
        ]

        """lists for plotting purposes"""
        self.saved_loss = []
        self.scores_synth_history = []
        self.mmds_all_itr = []

    def discriminator(self, x, y, reuse=None):
        theta_D=self.theta_D
        with tf.variable_scope('dis', reuse=reuse):
            inputs = tf.concat(axis=1, values=[x, y])
            D_h1 = tf.nn.leaky_relu(tf.add(tf.matmul(inputs, theta_D[0]), theta_D[4]))
            D_h1_dropout = tf.nn.dropout(D_h1, Hyperparam.D_KEEP_PERC)
            D_h2 = tf.nn.leaky_relu(tf.add(tf.matmul(D_h1_dropout, theta_D[1]), theta_D[5]))
            D_h2_dropout = tf.nn.dropout(D_h2, Hyperparam.D_KEEP_PERC)
            D_h3 = tf.nn.leaky_relu(tf.add(tf.matmul(D_h2_dropout, theta_D[2]), theta_D[6]))
            D_h3_dropout = tf.nn.dropout(D_h3, Hyperparam.D_KEEP_PERC)
            D_logit = tf.add(tf.matmul(D_h3_dropout, theta_D[3]),theta_D[7])
            D_prob = tf.nn.sigmoid(D_logit)
            return D_prob, D_logit

    def generator(self, z, y, reuse=None):
        theta_G = self.theta_G
        with tf.variable_scope('gen', reuse=reuse):
            inputs = tf.concat(axis=1, values=[z, y])
            G_h1 = tf.nn.leaky_relu(tf.matmul(inputs, theta_G[0]) + theta_G[4])
            G_h1_dropout = tf.nn.dropout(G_h1, Hyperparam.G_KEEP_PERC)
            G_h2 = tf.nn.leaky_relu(tf.matmul(G_h1_dropout, theta_G[1]) + theta_G[5])
            G_h2_dropout = tf.nn.dropout(G_h2, Hyperparam.G_KEEP_PERC)
            G_h3 = tf.nn.leaky_relu(tf.matmul(G_h2_dropout, theta_G[2]) + theta_G[6])
            G_h3_dropout = tf.nn.dropout(G_h3, Hyperparam.G_KEEP_PERC)
            G_log_prob = tf.matmul(G_h3_dropout, theta_G[3]) + theta_G[7]
            G_prob = tf.nn.tanh(G_log_prob)
            return G_prob


    def solver(self,X,y,G_sample,D_real,D_logit_real, D_fake, D_logit_fake,theta_D, theta_G):

        MODE=Hyperparam.MODE
        """ IAN-VANILLA """
        if MODE == 'ian_vanilla':
            G_loss = -xe(D_fake, 'maximize')
            D_loss_real = xe(D_real, 'minimize')
            D_loss_fake = xe(D_fake, 'maximize')
            D_loss = D_loss_real + D_loss_fake

            self.D_solver = tf.train.AdamOptimizer(learning_rate=0.0005, beta1=0.3).minimize(D_loss,
                                                                                        var_list=theta_D)  # theta_D#d_vars
            self.G_solver = tf.train.AdamOptimizer(learning_rate=0.0005, beta1=0.3).minimize(G_loss,
                                                                                        var_list=theta_G)  # theta_G#g_vars

            self.clip_disc_weights = None

        """ GAN-VANILLA """
        if MODE == 'vanilla':
            G_loss = xe(D_fake, 'maximize')
            D_loss_real = xe(D_real, 'maximize')
            D_loss_fake = xe(D_fake, 'minimize')
            D_loss = D_loss_real + D_loss_fake

            self.D_solver = tf.train.AdamOptimizer(learning_rate=0.0005, beta1=0.3).minimize(D_loss,
                                                                                        var_list=theta_D)
            self.G_solver = tf.train.AdamOptimizer(learning_rate=0.0005, beta1=0.3).minimize(G_loss,
                                                                                        var_list=theta_G)

            self.clip_disc_weights = None

        """ WGAN """
        if MODE == 'wgan':

            G_loss = tf.reduce_mean(D_logit_fake)
            D_loss = tf.reduce_mean(D_logit_real-D_logit_fake)

            self.G_solver = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(G_loss,
                                                                              var_list=theta_G)
            self.D_solver = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(D_loss,
                                                                              var_list=theta_D)

            clip_ops = []
            for var in theta_D:
                clip_bounds = [-0.01, 0.01]
                clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))

            self.clip_disc_weights = tf.group(*clip_ops)

        """ Logit GAN """
        if MODE == 'logit':
            G_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
            D_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
            D_loss += tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_real)))
            D_loss /= 2.

            self.D_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(D_loss,
                                                                                      var_list=theta_D)
            self.G_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(G_loss,
                                                                                      var_list=theta_G)

            self.clip_disc_weights = None

        """ WGAN-GP """
        if MODE == 'wgan-gp':
            G_loss = -tf.reduce_mean(D_fake)
            D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)

            alpha = tf.random_uniform(shape=[Hyperparam.MB_SIZE, 1], minval=0., maxval=1.)

            real_data = X
            fake_data = G_sample

            differences = fake_data - real_data
            interpolates = real_data + (alpha * differences)
            gradients = tf.gradients(self.discriminator(interpolates, y), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            D_loss += Hyperparam.LAMBDA * gradient_penalty

            self.G_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(G_loss,
                                                                                                 var_list=theta_G)
            self.D_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(D_loss,
                                                                                                 var_list=theta_D)

            self.clip_disc_weights = None

        """ WGAN-GAN-GP-SMK added log to wgan-gp """
        if MODE == 'wgan-gp-smk':
            G_loss = xe(D_fake, 'maximize')
            D_loss = xe(D_real, 'maximize') + xe(D_fake, 'minimize')

            alpha = tf.random_uniform(shape=[Hyperparam.MB_SIZE, 1], minval=0., maxval=1.)

            real_data = X
            fake_data = G_sample

            differences = fake_data - real_data
            interpolates = real_data + (alpha * differences)
            gradients = tf.gradients(self.discriminator(interpolates, y), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            D_loss += Hyperparam.LAMBDA * gradient_penalty

            self.G_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(G_loss,
                                                                                                 var_list=theta_G)
            self.D_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(D_loss,
                                                                                                 var_list=theta_D)

            self.clip_disc_weights = None

        """ LIP regularization """
        if MODE == 'lip-mul' or MODE == 'lip-log' or MODE == 'lip-sum':
            D_loss_real = xe_reg(D_real, Hyperparam.LAMBDA,self.theta_D, 'maximize')
            D_loss_fake = xe_reg(D_fake, Hyperparam.LAMBDA,self.theta_D, 'minimize')
            D_loss = D_loss_real + D_loss_fake
            G_loss = xe_reg(D_fake, Hyperparam.LAMBDA_G,self.theta_D, 'maximize')

            self.D_solver = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(D_loss,
                                                                                        var_list=theta_D)
            self.G_solver = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(G_loss,
                                                                                        var_list=theta_G)

            self.clip_disc_weights = None

        self.D_real=D_real
        self.D_fake=D_fake
        self.D_logit_real=D_logit_real
        self.D_logit_fake=D_logit_fake
        self.D_loss=D_loss
        self.G_loss=G_loss




    def train(self,datasets,preprocessor,loader,evaluator,distance):
        mmds_one_itr = []
        sess = tf.Session()
        tf.global_variables_initializer().run(session=sess)
        for epoch in range(Hyperparam.EPOCHS):

            X_mb, y_mb = loader.shuffle_x(Hyperparam.MB_SIZE)

            noise = np.random.normal(0, 1, (Hyperparam.MB_SIZE, Hyperparam.Z_DIM)) if Hyperparam.NOISE == 'normal' else np.random.uniform(-1, 1, (
            Hyperparam.MB_SIZE, Hyperparam.Z_DIM))

            for i in range(Hyperparam.D_STEPS):
                _, D_loss_curr, D_logit_real_curr, D_real_curr, D_logit_fake_curr, D_fake_curr = sess.run(
                    [self.D_solver, self.D_loss, self.D_logit_real, self.D_real, self.D_logit_fake, self.D_fake],
                    feed_dict={self.X: X_mb, self.Z: noise, self.y: y_mb})

            if self.clip_disc_weights is not None:
                _ = sess.run(self.clip_disc_weights)

            for i in range(Hyperparam.G_STEPS):
                _, G_loss_curr, D_fake_curr = sess.run([self.G_solver, self.G_loss, self.D_fake], feed_dict={self.Z: noise, self.y: y_mb})

            losses = np.array([epoch, D_loss_curr, G_loss_curr]).reshape(1, 3)
            self.saved_loss.append(losses)

            if epoch % 200 == 0:
                X_samples_real, y_samples_real = loader.shuffle_x(preprocessor.scaled_X.shape[0])
                synth_samples, real_samples = self.get_intraining_samples(sess,preprocessor, X_samples_real, y_samples_real)

                scores_synth =evaluator.test_model(preprocessor.scaler.transform(synth_samples), y_samples_real)
                scores_synth = np.concatenate(([epoch], scores_synth))
                self.scores_synth_history.append(scores_synth)

                for load in datasets.all_short_names:
                    load_synth_samples = synth_samples.loc[load]
                    load_real_samples = real_samples.loc[load]

                    ## MMD ###
                    mmd_val = sess.run(distance.maximum_mean_discrepancy_gauss(tf.constant(load_real_samples.values.astype(np.float32)),tf.constant(load_synth_samples.values.astype(np.float32)),self.sigmas))
                    mmds_one_itr.append(mmd_val)


                self.mmds_all_itr.append(mmds_one_itr)
                mmds_one_itr = []


                print('Iter: {}'.format(epoch))
                print('D_loss: {:.4}'.format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                print()
                print()

        """save model"""
        save_path = self.saver.save(sess, self.paths.current_path + 'pattern_model/pattern.ckpt') #Save mdel snapshot after training
        print("Model saved in path: %s" % save_path)

        """save evaluator's results of synthetic data at various epochs"""
        scores_synth_history = np.asarray(self.scores_synth_history)
        scores_synth_history_df = pd.DataFrame(scores_synth_history, columns=['itr'] + evaluator.model.metrics_names)
        scores_synth_history_df.to_csv(self.paths.current_path+ "scores_synth_history.csv", index=False)

        """save loss of both D and G during training """
        saved_loss = np.concatenate(self.saved_loss, axis=0)
        saved_loss = pd.DataFrame(saved_loss, columns=['epoch', 'D_Loss', 'G_Loss'])
        saved_loss.to_csv(self.paths.current_path + "loss_patterns.csv", index=False)
        saved_loss = pd.read_csv(self.paths.current_path + 'loss_patterns.csv')

        """save MMD results"""
        mmds_all_itr = np.vstack(self.mmds_all_itr)
        mmds = pd.DataFrame(mmds_all_itr, columns=datasets.all_short_names)
        mmds.to_csv(self.paths.current_path + "mmd.csv", index=False)

        """close session """
        sess.close()

    def get_intraining_samples(self, sess,preprocessor, X_mb_real, y_mb_real):

        noise = np.random.normal(0, 1, (len(y_mb_real), Hyperparam.Z_DIM)) if Hyperparam.NOISE == 'normal' else np.random.uniform(-1, 1, (
        len(y_mb_real), Hyperparam.Z_DIM))

        gen_samples = sess.run(self.generator(self.Z, self.y, reuse=True), feed_dict={self.Z: noise, self.y: y_mb_real})

        X_mb_real = preprocessor.scaler.inverse_transform(X_mb_real)
        gen_samples = preprocessor.scaler.inverse_transform(gen_samples)

        y_mb_real = preprocessor.encoder.inverse_transform(y_mb_real)

        X_mb_real = pd.DataFrame(X_mb_real, index=y_mb_real.ravel())
        X_mb_synth = pd.DataFrame(gen_samples, index=y_mb_real.ravel())

        return X_mb_synth, X_mb_real

    def generate_pattern(self,input_path,output_path, preprocessor,nof_samples, sampled_labels):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        sess = tf.Session()
        sess.run(init)

        """Restore Model"""
        saver.restore(sess, input_path + '/pattern_model/pattern.ckpt')

        """ Generate Patterns"""
        trained_output=self.get_trained_signatures( sess,preprocessor, nof_samples, sampled_labels)
        trained_output = pd.DataFrame(trained_output)
        trained_output.to_csv(output_path + "/synth_patterns.csv", index=False)
        trained_output_round = pd.read_csv(output_path + "/synth_patterns.csv")
        trained_output_round=trained_output_round.iloc[:,0:-3]
        trained_output_round.iloc[:, 2:] = (trained_output_round.iloc[:, 2:].astype(float)).round(0)
        trained_output_round.to_csv(output_path + "/synth_patterns_rounded.csv", index=False)

        """close session """
        sess.close()

        return trained_output.values

    def get_trained_signatures(self,sess,preprocessor, nof_samples=100, sampled_labels=['CDE', 'FRE']):

        stacked_samples = []


        sampled_labels = np.array(sampled_labels).reshape(-1, 1)
        sampled_labels = preprocessor.encoder.transform(sampled_labels).toarray()

        for i in range(nof_samples):
            samples = []

            noise = np.random.normal(0, 1, (len(sampled_labels), Hyperparam.Z_DIM)) if \
                Hyperparam.NOISE == 'normal' else np.random.uniform(-1,1, (len(sampled_labels), Hyperparam.Z_DIM))


            gen_signatures = sess.run(self.generator(self.Z,self.y, reuse=True), feed_dict={self.Z: noise, self.y: sampled_labels})

            iteration = i * np.ones(len(sampled_labels)).reshape(-1, 1)
            samples = np.append(sampled_labels, gen_signatures, axis=1)
            samples = np.append(iteration, samples, axis=1)
            stacked_samples.append(samples)

        stacked_samples = np.concatenate(stacked_samples, axis=0)

        stacked_samples[:, preprocessor.y_dim + 1:] = preprocessor.scaler.inverse_transform(stacked_samples[:, preprocessor.y_dim + 1:])

        stacked_samples_labels = stacked_samples[:, 1:preprocessor.y_dim + 1].astype(int)
        stacked_samples_labels = preprocessor.encoder.inverse_transform(stacked_samples_labels)
        stacked_samples = np.concatenate(
            (stacked_samples[:, 0].reshape(-1, 1), stacked_samples_labels, stacked_samples[:, preprocessor.y_dim + 1:]),
            axis=1)

        return stacked_samples