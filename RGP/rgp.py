from layers import SVGP_Layer

import tensorflow as tf
import numpy as np

from gpflow.params import DataHolder
from gpflow import autoflow, params_as_tensors, ParamList
from gpflow.models.model import Model
from gpflow.likelihoods import Gaussian
from gpflow import settings, features
float_type = settings.float_type


class RGP_Base(Model):
    """
      This is the base class for RGP

      - U is input data matrix, size N x D
      - Y is label data matrix, size N x C
    """

    def __init__(self, U, Y, likelihood, kernels,
                 window_lengths, dimensions,
                 U_win, init='random', X_var=0.01,
                 ** kwargs):
        Model.__init__(self, **kwargs)

        self.U_win = U_win
        self.nLayers = len(self.window_lengths)
        self.window_lengths = window_lengths
        self.dimensions = dimensions
        self.likelihood = likelihood
        self.kernels = kernels

        self._init_data(U, Y)
        self._init_params(init, X_var)
        self._init_layers()

    def _init_data(self, U, Y, init):
        self.U = DataHolder(U)

        self.Y = []
        for i in range(len(Y)):
            self.Y.append(Y[self.U_win:].copy())
        self.Y = DataHolder(np.array(Y))

    def _init_params(self, init, X_var):
        self.X = []
        for i_l in range(1, self.n_layers):
            xx = []
            C_win, C_dims = self.window_lengths[i_l], self.dimensions[i_l]
            for i_n in range(len(self.Y)):
                xx.append(tf.random.normal(
                    (C_win + self.Y.shape[0], C_dims), 0, X_var))
        self.X.append(xx)
        self.X = ParamList(np.array(self.X))

    def _init_layers(self):
        self.layers = []

        for i in range(self.nLayers - 1, -1, -1):
            if i == self.nLayers - 1:
                self.layers.append(SVGP_Layer(
                    X=self.X[i - 1], X_win=self.window_lengths[i], U=self.U, U_win=self.U_win))
            elif i == 0:
                self.layers.append(SVGP_Layer(
                    X=self.Y, X_win=self.window_lengths[i], U=self.X[i], U_win=self.window_lengths[i + 1]))
            else:
                self.layers.append(SVGP_Layer(
                    X=self.X[i - 1], X_win=self.window_lengths[i], U=self.X[i], U_win=self.window_lengths[i + 1]))

        self.layers = ParamList(self.layers)

    @params_as_tensors
    def _build_likelihood(self):
        return sum([self.layers[i]._build_likelihood() for i in range(len(self.layers))])

    @params_as_tensors
    def _build_predict(self, X_new):
        # TODO
        pass
