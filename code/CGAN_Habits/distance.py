import tensorflow as tf

class Distance():
    def compute_pairwise_distances(self,x, y):
        if not len(x.get_shape()) == len(y.get_shape()) == 2:
            raise ValueError('Both inputs should be matrices.')

        if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
            raise ValueError('The number of features should be the same.')

        norm = lambda x: tf.reduce_sum(tf.square(x), 1)

        return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))

    def gaussian_kernel_matrix(self,x, y, sigmas):
        beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

        dist = self.compute_pairwise_distances(x, y)

        s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

        return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))

    def maximum_mean_discrepancy_gauss(self, x, y, gaussian_sigmas):

        with tf.name_scope('MaximumMeanDiscrepancy'):
            cost = tf.reduce_mean(self.gaussian_kernel_matrix(x, x, gaussian_sigmas))
            cost += tf.reduce_mean(self.gaussian_kernel_matrix(y, y, gaussian_sigmas))
            cost -= 2 * tf.reduce_mean(self.gaussian_kernel_matrix(x, y, gaussian_sigmas))
            cost = tf.where(cost > 0, cost, 0, name='value')
        return cost
