import osqp
import numpy as np
import scipy as sp
from scipy import sparse
from typing import Tuple
import koopman
from scipy.sparse.linalg import eigs
from sklearn.preprocessing import StandardScaler
import gym
import numpy as np
import matplotlib.pyplot as plt

class Koopman_MPC:

    def __init__(self,x_dim,u_min,u_max,x_min,x_max,Q,QN,R,
                 xr,x0,N,A,B,scaler: StandardScaler) -> None:
        #self.cluster_centers = cluster_centers
        self.x_dim = x_dim
        self.u_min = u_min
        self.u_max = u_max
        self.z_min = np.array([-np.inf]*z_dim)
        self.z_max = np.array([np.inf]*z_dim)
        #print(self.z_min,self.z_max)
        self.Q = Q
        self.QN = QN
        self.R = R
        self.zr = scaler.transform(xr.reshape((1,-1))).squeeze()
        self.z0 = scaler.transform(x0.reshape((1,-1))).squeeze()
        self.N = N
        self.A = sparse.csc_matrix(A)
        self.B = sparse.csc_matrix(B)
        self.z_dim = A.shape[0]
        self.u_dim = B.shape[1]
        #print(self.z_min,self.z_max)

    def cast_mpc_to_qp(self):
        self.P = sparse.block_diag([sparse.kron(sparse.eye(self.N), self.Q), self.QN,
                       sparse.kron(sparse.eye(self.N), self.R)], format='csc')
        # - linear objective
        self.q_ = np.hstack([np.kron(np.ones(self.N), -self.Q.dot(self.zr)), -self.QN.dot(self.zr),
                    np.zeros(N*self.u_dim)])
        # - linear dynamics
        Ax = sparse.kron(sparse.eye(self.N+1),-sparse.eye(self.z_dim)) + sparse.kron(sparse.eye(self.N+1, k=-1), self.A)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, self.N)), sparse.eye(self.N)]), self.B)
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-self.z0, np.zeros(N*self.z_dim)])
        ueq = leq
        # - input and state constraints
        Aineq = sparse.eye((N+1)*self.z_dim + N*self.u_dim)
        lineq = np.hstack([np.kron(np.ones(self.N+1), self.z_min), np.kron(np.ones(self.N), self.u_min)])
        uineq = np.hstack([np.kron(np.ones(self.N+1), self.z_max), np.kron(np.ones(self.N), self.u_max)])
        # - OSQP constraints
        self.A_ = sparse.vstack([Aeq, Aineq], format='csc')
        self.l = np.hstack([leq, lineq])
        self.u_ = np.hstack([ueq, uineq])

    def setup_qp(self):

        prob = osqp.OSQP()
        prob.setup(self.P,self.q_,self.A_,self.l,self.u_,warm_start=True)

        return prob

    
    def mpc_loop(self,prob: osqp.OSQP):

        res = prob.solve()

        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')
        
        ctrl = res.x[-self.N*self.u_dim:-(self.N-1)*self.u_dim]
        self.z0 = self.A.dot(self.z0) + self.B.dot(ctrl)

        self.l[:self.z_dim] = -self.z0
        self.u_[:self.z_dim] = -self.z0
        prob.update(l=self.l,u = self.u_)

        return ctrl[0]



def compute_state_dynamics(data_path: str,state_lifted_space_dim: int,scaler: StandardScaler):
    omega = koopman.load_state_control_matrix(data_path=data_path)
    x0 = omega[0][:-1]
    omega = omega.T
    # for i in range(omega.shape[1]):
    #    if omega[-1][i] == 1:
    #        omega[-1][i] = 10
    #    else:
    #        omega[-1][i] = -10
    x0 = omega.T[0][:-1]
    X = omega[:,:]
    X[:4,:] = scaler.fit_transform(X[:4,:].T).T
    #cluster_centers = koopman.find_cluster_centers(X.T,n_clusters=state_lifted_space_dim)
    #Phi = koopman.rbf(X.T,cluster_centers).T
    Phi_omega = X[:,:-1]
    Phi_omega_dash = X[:-1,1:]
    #Phi_omega , Phi_omega_dash = koopman.get_data(Phi=Phi,ini_state_control_space=omega)
    A , B = koopman.get_finite_koopman_operator(Phi_omega=Phi_omega,Phi_omega_dash=Phi_omega_dash,state_lifted_space_dim=state_lifted_space_dim)

    return A, B , x0




if __name__ == "__main__":
    state_lifted_space_dim = 4
    scaler = StandardScaler()
    A , B , x0 = compute_state_dynamics("observations2.npy",state_lifted_space_dim,scaler=scaler)

    #Constraints
    x_dim = 4
    z_dim = A.shape[0]
    u_dim = B.shape[1]
    umin = np.array([-10])
    umax = np.array([10])
    xmin = np.array([-10,-0.2,-np.pi,-0.001])
    xmax = np.array([10,0.2,np.pi,0.001])

    env = gym.make('CartPole-v1',render_mode="human")
    observation, info = env.reset()
    env.env.state = np.zeros(4)
    

    #Objective function
    Q = np.eye(4, dtype=int)*0.3        # choose Q (weight for state)
    Q[3, 3] = 0.8
    Q[1, 1] = 0.9
    R = np.eye(1, dtype=int) *0.01  
    QN = Q

    #initial
    xr = np.array([0.0,0.0,np.pi,0.0])
    #z_r = koopman.rbf(xr.reshape((1,-1)),cluster_centers=cluster_centers)
    #z_r = z_r.squeeze()
    #z_0 = koopman.rbf(np.zeros(x_dim).reshape((1,-1)),cluster_centers=cluster_centers)

    #Prediction Horizon
    N = 50

    koopman_mpc = Koopman_MPC(x_dim,umin,umax,xmin,xmax,Q,QN,R,xr,x0,N,A,B,scaler=scaler)
    koopman_mpc.cast_mpc_to_qp()
    prob = koopman_mpc.setup_qp()

    nsim = 200
    #u_stack = []
    obs = []
    for i in range(nsim):
        env.render()
        u_dash = koopman_mpc.mpc_loop(prob=prob)
        print(u_dash)
        if u_dash > 0:
            action = 1
        else:
            action = 0 # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        obs.append(observation)
    
    #print(obs.shape)
    obs = np.array(obs)
    #obs = np.concatenate(obs,axis=0)
    plt.figure(1)
    plt.subplot(411)
    plt.plot(obs[:,0],label="x")
    plt.plot([0]*len(obs),label="x_ref")
    plt.legend()
    plt.subplot(412)
    plt.plot(obs[:,1],label="x_dot")
    plt.plot([0]*len(obs),label="x_dot_ref")
    plt.legend()
    plt.subplot(413)
    plt.plot(obs[:,2],label="theta")
    plt.plot([np.pi]*len(obs),label="theta_ref")
    plt.legend()
    plt.subplot(414)
    plt.plot(obs[:,3],label="theta_dot")
    plt.plot([0]*len(obs),label="thetat_dot_ref")
    plt.legend()
    plt.show()

    env.close()
    
    
    #print(u_stack)
    
    

