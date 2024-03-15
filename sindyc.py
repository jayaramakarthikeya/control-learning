import pysindy as ps
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


def load_state_control_matrix(data_path: str) -> np.ndarray:
    data = np.load(data_path)
    return data

def normalized_error(matrix_true, matrix_pred):
    return np.linalg.norm(matrix_true - matrix_pred) / np.linalg.norm(matrix_true)
    



if __name__ == "__main__":
   omega = load_state_control_matrix('observations2.npy') 
   state_lifted_space_dim = 4
   omega = omega.T
#    for i in range(omega.shape[1]):
#        if omega[-1][i] == 1:
#            omega[-1][i] = 10
#        else:
#            omega[-1][i] = -10
   X = omega[:,:20001]
   omega_test = omega[:,20001:]
   scaler = StandardScaler()
   X[:4,:] = scaler.fit_transform(X[:4,:].T).T
   omega_test[:4,:] = scaler.transform(omega_test[:4,:].T).T
   feature_lib = ps.PolynomialLibrary(degree=5,include_bias=True)
   parameter_lib = ps.PolynomialLibrary(degree=5,include_bias=True)
   lib = ps.ParameterizedLibrary(parameter_library=parameter_lib,feature_library=feature_lib,num_features=4,num_parameters=1)
   fourier_lib = ps.FourierLibrary(n_frequencies=2)
   lib_generalized = ps.GeneralizedLibrary([lib,fourier_lib])
   optimizer = ps.optimizers.STLSQ(max_iter=1000)
   model = ps.SINDy(optimizer=optimizer,feature_library=lib_generalized,discrete_time=True)
   model.fit(x=X[:4,:].T,u=X[-1,:].T)
   
   S = model.coefficients()
   X_test = omega_test[:,:-1]
   y_test = omega_test[:,1:]

   sim = model.simulate(x0=omega_test[:-1,1],u=omega_test[-1,:],t=1000)

   fig = plt.figure(1)
   plt.plot(sim[:,0])
   plt.show()


   


   
   