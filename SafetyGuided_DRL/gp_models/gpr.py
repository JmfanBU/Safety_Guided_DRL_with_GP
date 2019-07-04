# Modified from gpflow


import tensorflow as tf

from gpflow import likelihoods
from gpflow import settings

from gpflow.conditionals import base_conditional
from gpflow.params import DataHolder
from gpflow.decors import params_as_tensors
from gpflow.decors import name_scope
from gpflow.logdensities import multivariate_normal

from SafetyGuided_DRL.gp_models.model import GPModel


class GPR(GPModel):
    """
    Gaussian Process Regression.

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.

    The log likelihood of this model is sometimes referred to as the
    'marginal log likelihood', and is given by

    .. math::

       \log p(\mathbf y | \mathbf f) = \mathcal N(\mathbf y | 0, \mathbf K + \sigma_n \mathbf I)
    """
    def __init__(self, X, Y, kern, noise_sigma=0.1, tol=0.05, mean_function=None, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects
        """
        likelihood = likelihoods.Gaussian()
        X = DataHolder(X)
        Y = DataHolder(Y)
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)

        self.noise_sigma = noise_sigma  # observation noise bound
        self.tol = tol  # probability tolerance

    @name_scope('likelihood')
    @params_as_tensors
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.

            \log p(Y | theta).

        """
        K = self.kern.K(self.X) + tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) * self.likelihood.variance
        L = tf.cholesky(K)
        m = self.mean_function(self.X)
        logpdf = multivariate_normal(self.Y, m, L)  # (R,) log-likelihoods for each independent dimension of Y

        return tf.reduce_sum(logpdf)

    @name_scope('predict')
    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, the points at which we want to predict.

        This method computes

            p(F* | Y)

        where F* are points on the GP at Xnew, Y are noisy observations at X.

        """
        y = self.Y - self.mean_function(self.X)
        Kmn = self.kern.K(self.X, Xnew)
        Kmm = self.kern.K(self.X)
        Kmm_sigma = self.kern.K(self.X) + tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) * self.likelihood.variance
        Knn = self.kern.K(Xnew) if full_cov else self.kern.Kdiag(Xnew)
        f_mean, f_var = base_conditional(Kmn, Kmm_sigma, Knn, y, full_cov=full_cov, white=False)  # N x P, N x P or P x N x N

        # Compute confidence interval
        B = tf.matmul(tf.matmul(self.Y, Kmm, transpose_a=True), self.Y)
        gamma_n = 1/2 * tf.log(tf.clip_by_value(
            tf.linalg.det(tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) + Kmm/self.noise_sigma**2), 1.0, 1e10))
        # gamma_n = tf.log(1 + tf.linalg.trace(Kmm)/self.noise_sigma**2)
        beta = tf.sqrt(B) + 4 * self.noise_sigma * tf.sqrt(1 + gamma_n + tf.log(1/self.tol))
        return f_mean + self.mean_function(Xnew), f_var, beta

    @name_scope('test')
    @params_as_tensors
    def _build_predict_test(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, the points at which we want to predict.

        This method computes

            p(F* | Y)

        where F* are points on the GP at Xnew, Y are noisy observations at X.

        """
        y = self.Y - self.mean_function(self.X)
        Kmn = self.kern.K(self.X, Xnew)
        Kmm = self.kern.K(self.X)
        Kmm_sigma = self.kern.K(self.X) + tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) * self.likelihood.variance
        Knn = self.kern.K(Xnew) if full_cov else self.kern.Kdiag(Xnew)
        f_mean, f_var = base_conditional(Kmn, Kmm_sigma, Knn, y, full_cov=full_cov, white=False)  # N x P, N x P or P x N x N

        # Compute confidence interval
        B = tf.matmul(tf.matmul(self.Y, Kmm, transpose_a=True), self.Y)
        # gamma_n = 1/2 * tf.log(tf.linalg.det(tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) + Kmm/self.noise_sigma**2))
        # gamma_n = tf.log(1 + tf.linalg.trace(Kmm)/self.noise_sigma**2)
        gamma_n = 1/2 * tf.log(tf.clip_by_value(
            tf.linalg.det(tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) + Kmm/self.noise_sigma**2), 1.0, 1e10))
        beta = tf.sqrt(B) + 4 * self.noise_sigma * tf.sqrt(1 + gamma_n + tf.log(1/self.tol))
        return f_mean + self.mean_function(Xnew), f_var, beta

    def update_feed_dict(self):
        feed_dict = {}
        feed_dict.update(self.initializable_feeds)
        return feed_dict
