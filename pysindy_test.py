import warnings
from contextlib import contextmanager
from copy import copy
from pathlib import Path

# try:
#     import gurobipy
#     run_miosr = True
#     GurobiError = gurobipy.GurobiError
# except ImportError:
#     run_miosr = False

import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
from pysindy.utils import enzyme
from pysindy.utils import lorenz
from pysindy.utils import lorenz_control
from scipy.integrate import solve_ivp
from scipy.linalg import LinAlgWarning
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

if __name__ != "testing":
    t_end_train = 10
    t_end_test = 15
else:
    t_end_train = .04
    t_end_test = .04

data = (Path() / "../data").resolve()

@contextmanager
def ignore_specific_warnings():
    filters = copy(warnings.filters)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=LinAlgWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    yield
    warnings.filters = filters

train_data = np.loadtxt("out.csv", delimiter=',', skiprows=1)
t_train    = train_data[140:,0]
x_train    = train_data[140:,1]
u_train    = train_data[140:,2]
library_functions = [
    lambda steer, u: ((steer - u >= 0.002) * (steer - u - 0.002)) +  ((steer - u <= -0.002) * (steer - u - (-0.002))),
]
library_function_names = [
    lambda x,u : 'dz_0.002(' + x + ',' + u + ')',
]
custom_library = ps.CustomLibrary(
    library_functions=library_functions, function_names=library_function_names
) + ps.IdentityLibrary() + ps.PolynomialLibrary() + ps.FourierLibrary()

ps.ParameterizedLibrary()
smoother_kws = {"polyorder":1,"window_length":11}
smoothed_fd = ps.SmoothedFiniteDifference(smoother_kws=smoother_kws)
ensemble_optimizer = ps.EnsembleOptimizer(
    ps.STLSQ(threshold=0.2,normalize_columns=False),
    bagging=False,
    library_ensemble=True,
    n_subset=int(x_train.shape[0])
)
# Instantiate and fit the SINDYc model
model = ps.SINDy(feature_library=custom_library, optimizer=ensemble_optimizer, differentiation_method=smoothed_fd)
model.fit(x_train, u=u_train, t=t_train)
model.print()

# Predict derivatives using the learned model
x_dot_train_predicted = model.predict(x_train, u=u_train)  

# Compute derivatives with a finite difference method, for comparison
x_dot_train_computed = model.differentiate(x_train, t=t_train)

x_dot_int = np.zeros_like(x_dot_train_computed)
x_dot_int[0] = x_train[0]

dt = np.diff(t_train)
for i in range(1, x_dot_train_computed.shape[0]):
    x_dot_int[i] = x_dot_int[i-1] + x_dot_train_predicted[i-1] * dt[i-1]


plt.plot(x_train, 'r', label='true_steer')
plt.plot(x_dot_int, 'g', label='pred_steer')
plt.show()

plt.plot(x_dot_train_computed, 'r', label='true_steer_dt')
plt.plot(x_dot_train_predicted, 'g', label='pred_steer_dt')
plt.show()