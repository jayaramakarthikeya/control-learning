import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from typing import Tuple
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

def load_state_control_matrix(data_path: str) -> np.ndarray:
    data = np.load(data_path)
    return data

def rbf(X: np.ndarray,cluster_centers):
    euclidist = euclidean_distances(X,cluster_centers)
    Phi = np.power(euclidist,2)*np.log(euclidist)
    return Phi

def find_cluster_centers(X:np.ndarray,n_clusters: int):
    kmeans_ = KMeans(n_clusters=n_clusters)
    kmeans_.fit(X)
    cluster_centers = kmeans_.cluster_centers_

    return cluster_centers

def lift_observable_space(X: np.ndarray, n_clusters: int) -> np.ndarray:
    
    cluster_centers = find_cluster_centers(X,n_clusters=n_clusters)
    Phi = rbf(X,cluster_centers)
    return Phi.T

def get_data(Phi: np.ndarray, ini_state_control_space: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    Omega = Phi[:,:-1]
    Omega_dash = Phi[:-1,1:]
    #Omega = np.vstack((Omega,ini_state_control_space[-1,:-1].reshape((1,-1))))
    #Omega_dash = np.hstack((Omega_dash,ini_state_control_space[1:,-1].reshape((-1,1))))
    return Omega , Omega_dash

def compute_unlift_operator(X: np.ndarray,Phi: np.ndarray) -> np.ndarray:

    return X @ np.linalg.pinv(Phi)

def compute_svd(X:np.ndarray)-> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    U , s, V = np.linalg.svd(X,full_matrices=False)
    V = V.conj().T
    return (U,s,V)

def get_finite_koopman_operator(Phi_omega:np.ndarray,Phi_omega_dash: np.ndarray,state_lifted_space_dim: int) -> np.ndarray:
    U_tilde , sigma_tilde , V_tilde_star = compute_svd(Phi_omega)

    U_tilde_1 = U_tilde[:state_lifted_space_dim,:]
    U_tilde_2 = U_tilde[state_lifted_space_dim:,:]
    U_cap , _ , _ = compute_svd(Phi_omega_dash)

    A_tilde = np.linalg.multi_dot(
       [U_cap.T.conj(),Phi_omega_dash,V_tilde_star,np.diag(np.reciprocal(sigma_tilde)),U_tilde_1.T.conj(),U_cap]
    )

    B_tilde = np.linalg.multi_dot([
       U_cap.T.conj(),Phi_omega_dash,V_tilde_star,np.diag(np.reciprocal(sigma_tilde)),U_tilde_2.T.conj()
    ])

    return A_tilde,B_tilde



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
   #cluster_centers = find_cluster_centers(X.T,n_clusters=state_lifted_space_dim)
   #Phi = rbf(X.T,cluster_centers=cluster_centers).T
   #Phi_test = rbf(omega_test.T,cluster_centers=cluster_centers).T
   #C = compute_unlift_operator(X,Phi)
   Phi_omega , Phi_omega_dash = get_data(Phi=X,ini_state_control_space=omega[:,:20001])
   A_tilde , B_tilde = get_finite_koopman_operator(Phi_omega=Phi_omega,Phi_omega_dash=Phi_omega_dash,state_lifted_space_dim=state_lifted_space_dim)
   
   
   #x_k = omega_test[:4,0]
   #u_k = omega_test[-1,0]
#    omega = load_state_control_matrix('observations.npy') 
#    omega_test = omega.T
#    for i in range(omega_test.shape[1]):
#        if omega_test[-1][i] == 1:
#            omega_test[-1][i] = 10
#        else:
#            omega_test[-1][i] = -10
   x_lift = []
   x_k = omega_test[:4,0].reshape((-1,1))
   for t in range(omega_test.shape[1]):
       x_k_plus_1 = A_tilde @ x_k + B_tilde @ omega_test[-1,t].reshape((-1,1))
       x_k = omega_test[:4,t].reshape((-1,1))
       #x_k = x_k_plus_1
       #print(x_k_plus_1.shape)
       x_lift.append(x_k_plus_1.T)
   x_lift = np.concatenate(x_lift,axis=0)

   #x_koop = C @ x_lift.T
   #x_koop = x_koop.T
   x_koop = x_lift

   for i in range(4):
       rms = mean_squared_error(x_koop[:,i], omega_test.T[:,i], squared=True)
       print(f"RMSE of {i}th variable = ",rms)
   

   fig = plt.figure(1)
   plt.subplot(411)
   plt.plot(x_koop[:,0],'r-',label="koopman predictions")
   plt.plot(omega_test[0,:],'g-',label ="True data")
   plt.legend()
   plt.subplot(412)
   plt.plot(x_koop[:,1],'r-',label="koopman predictions")
   plt.plot(omega_test[1,:],'g-',label ="True data")
   plt.legend()
   plt.subplot(413)
   plt.plot(x_koop[:,2],'r-',label="koopman predictions")
   plt.plot(omega_test[2,:],'g-',label ="True data")
   plt.legend()
   plt.subplot(414)
   plt.plot(x_koop[:,3],'r-',label="koopman predictions")
   plt.plot(omega_test[3,:],'g-',label ="True data")
   plt.legend()
   plt.show()

#    fig = plt.figure(2)
#    plt.plot(x_koop[:,1],'r-')
#    plt.plot(omega_test[1,:],'g-')
   
#    plt.show()