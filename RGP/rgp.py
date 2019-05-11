from .layers import SVGP_Layer

from .latent_store import Latent_Store
import tensorflow as tf
import numpy as np

from gpflow import autoflow, params_as_tensors, ParamList, DataHolder
from gpflow.models.model import Model
from gpflow import settings

from gpflow.mean_functions import Zero
float_type=settings.float_type


class RGP_Base(Model):
    """
      This is the base class for RGP

      - U is input data matrix, size nSeq x N x D
      - Y is label data matrix, size nSeq x N x C
    """

    def __init__(self,
                 U,
                 Y,
                 likelihood,
                 kernels,
                 num_classes=None,
                 init='random',
                 X_var=0.01,
                 **kwargs):
        Model.__init__(self, **kwargs)

        # U and Y are list of ndarrays
        if isinstance(U, np.ndarray):
            U = [U]
        if isinstance(Y, np.ndarray):
            Y = [Y]

        self.nSeq = len(Y)
        self.kernels = kernels
        self.nLayers = len(self.kernels)
        self.init = init
        self.X_var = X_var

        self.likelihood = likelihood

        if num_classes is None:
            self.num_classes = self.Y[0].shape[1]
        else:
            self.num_classes = num_classes

        self._init_data(U, Y)
        self._init_layers()

    def _init_data(self, U, Y):
        self.U, self.Y = U, Y

        # for i in range(self.nSeq):
        #    self.U[i] = DataHolder(self.U[i])
        #    self.Y[i] = DataHolder(self.Y[i])

    def _init_layers(self):
        self.layers = []

        for i in range(self.nLayers):
            self.layers.append(
                SVGP_Layer(layer_id=i,
                           Z=None,
                           U=self.U,
                           kern=self.kernels[i],
                           num_outputs=self.num_classes,
                           mean_function=Zero()))

        self.layers = ParamList(self.layers)

    @params_as_tensors
    def _build_predict(self, U, full_cov=False):
        Fmeans, Fvars = [], []
        for i_seq in range(self.nSeq):
            Fs, Fvars = [], []
            X_mean = np.random.normal(0, self.X_var, U[i_seq].shape)
            for layer in self.layers:
                X_mean, X_var = layer.propagate_inp(X_mean, U[i_seq], full_cov=full_cov)
                Fs.append(X_mean)
                Fvars.append(X_var)
            print("unit ", Fs[-1][0].shape)
            mstacked = tf.concat(Fs[-1], axis=0)
            varstacked = tf.concat(Fvars[-1], axis=0)
            print("stacked ", mstacked.shape, varstacked.shape)
            Fmeans.append(mstacked)
            Fvars.append(varstacked)
        return Fmeans, Fvars

    def E_log_p_Y(self, U, Y):
        """
        Calculate the expectation of the data log likelihood under the variational distribution
         with MC samples
        """
        Fmeans, Fvars = self._build_predict(U, full_cov=False)
        var_exps = []
        for i in range(self.nSeq):
            var_exp = self.likelihood.variational_expectations(Fmeans[i], Fvars[i],
                                                           Y[i])  # S, N, D
            var_exps.append(tf.reduce_mean(var_exp, 0))

        var_exp_total = tf.reduce_sum(var_exps, 0)
        return var_exp_total

    @params_as_tensors
    def _build_likelihood(self):
        L = tf.reduce_sum(self.E_log_p_Y(self.U, self.Y))
        KL = tf.reduce_sum([layer.KL() for layer in self.layers])
        scale = tf.cast(self.nSeq, float_type)
        return L * scale - KL
