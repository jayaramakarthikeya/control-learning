
import numpy as np
from inverted_pendulum import CustomInvertedPendulumEnv
from scipy import linalg
import time

env = CustomInvertedPendulumEnv(render_mode="human")
env.init_qpos = np.array([0, np.pi])
observation , _ = env.reset()
env.render()
time.sleep(1)
observations = np.hstack((observation,0))
print(observation)

#env.height
np.random.seed(42)

class energy_shaping_controller:

  def __init__(self):
    # k_p and k_d gains for swing up controller
    
    # choose appropriate k_p and k_d gains
    self.kp = (0.6*3)*400
    self.kd = (0.125*3)*700
    
    self.g  = -9.81

    # mass of the pole and mass of the part
    self.mc = 1.0
    self.mp = 0.1
    self.L  = 0.5
    #print(self.mc,self.mp,self.L)

    # Computes the lqr gains for the final stabilizing controller
    a = self.g / (self.L*(4.0/3 - self.mp/(self.mp + self.mc)))
    A = np.array([[0, 1, 0, 0], 
                  [0, 0, a, 0], 
                  [0, 0, 0, 1], 
                  [0, 0, a, 0]])
                  
    
    b = -1/(self.L*(4.0/3 - self.mp/(self.mp + self.mc)))
    B = np.array([[0, 1/(self.mp + self.mc), 0, b]]).T
    Q = np.eye(4, dtype=int)        # choose Q (weight for state)
    Q[3, 3] = 10
    R = np.eye(1, dtype=int)          # choose R (weight for input)

    # solve ricatti equation
    self.s = linalg.solve_continuous_are(A, B, Q, R)

    # calculate optimal controller gain
    self.K  = np.dot(np.linalg.inv(R),
                     np.dot(B.T, self.s))
                     
    self.x_des = np.array([0, 0, np.pi, 0])


  def swingup_policy(self, obs):
    # FIX THETA REFERENCE, NECESSARY FOR CONTROLLER
    obs[1] = (obs[1] + 2*np.pi) % (2 * np.pi) # Convert -pi to pi to 0 to 2pi
    
    q = obs[[0,1]]
    qdot = obs[[2,-1]]

    e_tilde = 0.5 * qdot[1] ** 2 - self.g * np.cos(q[1]) - self.g
    angular_distance  = obs[3]**2 + (obs[1]-np.pi)**2

    u = 0
    if np.abs(e_tilde) < 1 and angular_distance < 1:
      u = self.compute_lqr_input(obs)
    else:
      u = self.compute_energy_shaping_input(obs)
    #print(u)
    return u
    


  def compute_energy_shaping_input(self, obs):
    '''
    Computes the energy shaping inputs to stabilize the cartpole
    '''
    q = obs[[0,1]]
    qdot = obs[[2,-1]]
    u = 0
    q1_ddot_des = 0
    q1_ddot_pd_term = - self.kd * qdot[0] - self.kp * q[0]

    q1dot = qdot[0]
    q2dot = qdot[1]

    q1 = q[0]
    q2 = q[1]
    
    # compute q1_ddot_des 
    e_tilde = 0.5 * qdot[1] ** 2 - self.g * np.cos(q[1]) - self.g
    q1_ddot_des = q2dot * np.cos(q2) * e_tilde

    # Add pd terms to q1_ddot_des
    q1_ddot_des = q1_ddot_des + q1_ddot_pd_term

    # compute the optimal input u according to q1_ddot_des and partial feedback linearization
    m1 = self.mc + self.mp
    m2 = self.mp * self.L * np.cos(q2)
    m3 = self.mp * (self.L)**2
    
    c1 = -self.mp * self.L * np.sin(q2) * q2dot**2
    c2 = self.mp * self.g * self.L * np.sin(q2)
     
    u = (m1 - (m2**2)/m3) * q1_ddot_des - m2 * (c2 / m3) + c1

    return u

  def compute_lqr_input(self, obs):
    '''
    Stabilizes the cartpole at the final location using lqr
    '''
    return -self.K @ (obs - self.x_des)

           
i = 0
control = energy_shaping_controller()
for _ in range(23000):
    env.render()
    action = np.array([float(np.clip(control.swingup_policy(observation),-10,10))])
    #action = np.array([-10]) #np.random.uniform(-10,10,1)
    #print(observation,action)
    # if i < len(u_stack):
    #     action = np.array([float(np.clip(u_stack[i],-10,10))])
    #     print(action)
    # else:
    #     action = np.random.uniform(-10,10,1)
    observation, reward, terminated, _, info = env.step(action)
    state_control_space = np.hstack((observation,action))
    observations = np.vstack((observations,state_control_space))
    i += 1

print(observations.shape)
np.save('observations.npy',observations)
env.close()
