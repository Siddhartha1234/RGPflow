#!/usr/bin/env python
# coding: utf-8

# In[27]:

from RGP.rgp import RGP_Base
from scipy import io
from gpflow.likelihoods import Gaussian
from gpflow.training import ScipyOptimizer
import os
os.environ['CUDA_VISIBLE_DEVICES']=""

import gpflow.training.monitor as mon

import gpflow
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

# In[28]:

data = io.loadmat('./notebooks/identificationExample.mat')
data_in = data['u'][:, None]
data_out = data['y'][:, None]
win_in = int(data['lu'])
win_out = int(data['ly'])

data_in_train = data_in[:150]
data_out_train = data_out[:150]
data_in_test = data_in[150:]
data_out_test = data_out[150:]

# In[29]:

print(data_in_train.shape)
print(data_in_test.shape)

# In[30]:

likelihood = Gaussian()
kern = gpflow.kernels.Matern32(input_dim=1, variance=10., lengthscales=2.)

kernels = [kern] * 2
m = RGP_Base(U=data_in_train,
             Y=data_out_train,
             likelihood=likelihood,
             kernels=kernels,
             num_classes=1)

print_task = mon.PrintTimingsTask().with_name('print')\
    .with_condition(mon.PeriodicIterationCondition(10))\
    .with_exit_condition(True)

session = m.enquire_session()
global_step = mon.create_global_step(session)

with mon.Monitor([print_task],session, global_step, print_summary=True) as monitor:
    ScipyOptimizer().minimize(m, step_callback=monitor, maxiter=2000, global_step=global_step)



# In[ ]:
