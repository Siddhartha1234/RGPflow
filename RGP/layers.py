import tensorflow as tf
import numpy as np

from gpflow import Parameterized, settings, transforms
from gpflow import params_as_tensors

from gpflow.params import Parameter, Parameterized
from gpflow.features import InducingPoints

from scipy.cluster.vq import kmeans2

class Layer(Parameterized):
    """
        SVGP unit in RGP
    """

    def __init__(self, layer_id, U, num_outputs,
                 **kwargs):
        Parameterized.__init__(self, **kwargs)
        self.layer_id = layer_id
        self.U = U

        self.num_outputs = num_outputs

    def conditional_ND(self, X, full_cov=False):
        raise NotImplementedError

    def KL(self):
        return tf.cast(0., dtype=settings.float_type)

    def propagate_inp(self, X, U, full_cov=False):
        X_pred_mean = []
        X_pred_var = []
        for ind in range(U.shape[0]):
            print(ind)
            if self.layer_id == 0:
                if ind == 0:
                    X_in = tf.reshape(U[ind], [1, -1])
                else:
                    X_in = tf.concat([tf.reshape(X_pred_mean[ind - 1], [1, -1]), tf.reshape(U[ind], [1, -1])], axis=1)
                X_out_mean, X_out_var = self.conditional_ND(X_in, full_cov=full_cov)
            else:
                if ind == 0:
                    X_in = tf.reshape(X[ind], [1, -1])
                else:
                    X_in = tf.concat([tf.reshape(X_pred_mean[ind - 1], [1, -1]), tf.reshape(X[ind], [1, -1])], axis=1)
                X_out_mean, X_out_var = self.conditional_ND(X_in, full_cov=full_cov)
            X_pred_mean.append(tf.reshape(X_out_mean, [1, -1]))
            X_pred_var.append(tf.reshape(X_out_var, [1, -1]))
        return X_pred_mean, X_pred_var


class SVGP_Layer(Layer):
    def __init__(self, layer_id, kern, U, Z, num_outputs, mean_function,
                 white=False, **kwargs):
        """
        A sparse variational GP layer in whitened representation. This layer holds the kernel,
        variational parameters, inducing points and mean function.
        The underlying model at inputs X is
        f = Lv + mean_function(X), where v \sim N(0, I) and LL^T = kern.K(X)
        The variational distribution over the inducing points is
        q(v) = N(q_mu, q_sqrt q_sqrt^T)
        The layer holds D_out independent GPs with the same kernel and inducing points.
        :param kern: The kernel for the layer (input_dim = D_in)
        :param Z: Inducing points (M, D_in)
        :param num_outputs: The number of GP outputs (q_mu is shape (M, num_outputs))
        :param mean_function: The mean function
        :return:
        """
        Layer.__init__(self, layer_id, U, num_outputs, **kwargs)

        #Initialize using kmeans

        self.dim_in  = U[0].shape[1] if layer_id == 0 else num_outputs
        self.Z = Z if Z is not None else np.random.normal(0, 0.01, (100, self.dim_in))

        self.num_inducing = self.Z.shape[0]

        q_mu = np.zeros((self.num_inducing, num_outputs))
        self.q_mu = Parameter(q_mu)

        q_sqrt = np.tile(np.eye(self.num_inducing)[None, :, :], [num_outputs, 1, 1])
        transform = transforms.LowerTriangular(self.num_inducing, num_matrices=num_outputs)
        self.q_sqrt = Parameter(q_sqrt, transform=transform)

        self.feature = InducingPoints(self.Z)
        self.kern = kern
        self.mean_function = mean_function

        self.num_outputs = num_outputs
        self.white = white

        if not self.white:  # initialize to prior
            Ku = self.kern.compute_K_symm(self.Z)
            Lu = np.linalg.cholesky(Ku + np.eye(self.Z.shape[0])*settings.jitter)
            self.q_sqrt = np.tile(Lu[None, :, :], [num_outputs, 1, 1])

        self.needs_build_cholesky = True

    @params_as_tensors
    def build_cholesky_if_needed(self):
        # make sure we only compute this once
        if self.needs_build_cholesky:
            self.Ku = self.feature.Kuu(self.kern, jitter=settings.jitter)
            self.Lu = tf.cholesky(self.Ku)
            self.Ku_tiled = tf.tile(self.Ku[None, :, :], [self.num_outputs, 1, 1])
            self.Lu_tiled = tf.tile(self.Lu[None, :, :], [self.num_outputs, 1, 1])
            self.needs_build_cholesky = False


    def conditional_ND(self, X, full_cov=False):
        self.build_cholesky_if_needed()

        # mmean, vvar = conditional(X, self.feature.Z, self.kern,
        #             self.q_mu, q_sqrt=self.q_sqrt,
        #             full_cov=full_cov, white=self.white)
        Kuf = self.feature.Kuf(self.kern, X)

        A = tf.matrix_triangular_solve(self.Lu, Kuf, lower=True)
        if not self.white:
            A = tf.matrix_triangular_solve(tf.transpose(self.Lu), A, lower=False)

        mean = tf.matmul(A, self.q_mu, transpose_a=True)

        A_tiled = tf.tile(A[None, :, :], [self.num_outputs, 1, 1])
        I = tf.eye(self.num_inducing, dtype=settings.float_type)[None, :, :]

        if self.white:
            SK = -I
        else:
            SK = -self.Ku_tiled

        if self.q_sqrt is not None:
            SK += tf.matmul(self.q_sqrt, self.q_sqrt, transpose_b=True)


        B = tf.matmul(SK, A_tiled)

        if full_cov:
            # (num_latent, num_X, num_X)
            delta_cov = tf.matmul(A_tiled, B, transpose_a=True)
            Kff = self.kern.K(X)
        else:
            # (num_latent, num_X)
            delta_cov = tf.reduce_sum(A_tiled * B, 1)
            Kff = self.kern.Kdiag(X)

        # either (1, num_X) + (num_latent, num_X) or (1, num_X, num_X) + (num_latent, num_X, num_X)
        var = tf.expand_dims(Kff, 0) + delta_cov
        var = tf.transpose(var)

        return mean + self.mean_function(X), var

    def KL(self):
        """
        The KL divergence from the variational distribution to the prior
        :return: KL divergence from N(q_mu, q_sqrt) to N(0, I), independently for each GP
        """
        # if self.white:
        #     return gauss_kl(self.q_mu, self.q_sqrt)
        # else:
        #     return gauss_kl(self.q_mu, self.q_sqrt, self.Ku)

        self.build_cholesky_if_needed()

        KL = -0.5 * self.num_outputs * self.num_inducing
        KL -= 0.5 * tf.reduce_sum(tf.log(tf.matrix_diag_part(self.q_sqrt) ** 2))

        if not self.white:
            KL += tf.reduce_sum(tf.log(tf.matrix_diag_part(self.Lu))) * self.num_outputs
            KL += 0.5 * tf.reduce_sum(tf.square(tf.matrix_triangular_solve(self.Lu_tiled, self.q_sqrt, lower=True)))
            Kinv_m = tf.cholesky_solve(self.Lu, self.q_mu)
            KL += 0.5 * tf.reduce_sum(self.q_mu * Kinv_m)
        else:
            KL += 0.5 * tf.reduce_sum(tf.square(self.q_sqrt))
            KL += 0.5 * tf.reduce_sum(self.q_mu**2)

        return KL

