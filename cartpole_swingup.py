import gymnasium as gym
import numpy as np
import os

# Common imports
import numpy as np
import random
import os
import collections
import gym
import sys
from cartpole import CartPoleEnv

np.set_printoptions(threshold=sys.maxsize)
#np.random.seed(1)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# To get smooth animations
import matplotlib.animation as animation
mpl.rc('animation', html='jshtml')

class energy_shaping_controller:

  def __init__(self, env):
    # k_p and k_d gains for swing up controller
    
    # choose appropriate k_p and k_d gains
    self.kp = (0.6*3)*400
    self.kd = (0.125*3)*700
    
    self.g  = env.gravity

    # mass of the pole and mass of the part
    self.mc = env.masscart
    self.mp = env.masspole
    self.L  = env.length
    print(self.mc,self.mp,self.L)

    # get riccati solver
    from scipy import linalg

    # Computes the lqr gains for the final stabilizing controller
    a = self.g / (self.L*(4.0/3 - self.mp/(self.mp + self.mc)))
    A = np.array([[0, 1, 0, 0], 
                  [0, 0, a, 0], 
                  [0, 0, 0, 1], 
                  [0, 0, a, 0]])
                  
    
    b = -1/(self.L*(4.0/3 - self.mp/(self.mp + self.mc)))
    B = np.array([[0, 1/(self.mp + self.mc), 0, b]]).T
    Q = np.eye(4, dtype=int)*20     # choose Q (weight for state)
    Q[3, 3] = 30
    Q[1, 1] = 50
    R = np.eye(1, dtype=int) *0.01         # choose R (weight for input)

    # solve ricatti equation
    self.s = linalg.solve_continuous_are(A, B, Q, R)

    # calculate optimal controller gain
    self.K  = np.dot(np.linalg.inv(R),
                     np.dot(B.T, self.s))
                     
    self.x_des = np.array([0, 0, np.pi, 0])


  def swingup_policy(self, obs):
    # FIX THETA REFERENCE, NECESSARY FOR CONTROLLER
    obs[2] = obs[2] + np.pi

    q = obs[[0,2]]
    qdot = obs[[1,-1]]

    e_tilde = 0.5 * qdot[1] ** 2 - self.g * np.cos(q[1]) - self.g
    angular_distance  = obs[3]**2 + (obs[2]-np.pi)**2

    u = 0
    if np.abs(e_tilde) < 1 and angular_distance < 2:
      u = self.compute_lqr_input(obs)
      u = u[0]
      print(u)
    else:
      u = self.compute_energy_shaping_input(obs)
      print("**",u)
    #print(u)
    if u > 0:
        return 1 , u    # if force_dem > 0 -> move cart right
    else:
        return 0 , u     # if force_dem <= 0 -> move cart left


  def compute_energy_shaping_input(self, obs):
    '''
    Computes the energy shaping inputs to stabilize the cartpole
    '''
    q = obs[[0,2]]
    qdot = obs[[1,-1]]
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

def random_policy(obs):
    return bool(random.getrandbits(1))

def record_scenario(env, policy, obs_init=None, num_frames=100) -> dict:
    frames = []
    obs_mat = np.empty((num_frames, 4))
    actions = np.empty((num_frames,))
    rewards = np.empty((num_frames,))
    dones = np.empty((num_frames,), dtype=int)
    
    first_done_info = ''
    if obs_init is None:
        obs = env.reset()  # initial observation
    else:
        obs = env.reset()  # initialize
        obs = obs_init
        env.state = obs_init
        observations = np.hstack((obs,0))
    for i in range(num_frames):
        if i % 1200 == 0:
          cart_vel = np.random.uniform(-0.05,0.05,1)[0]
          angle = np.random.uniform(-np.pi,np.pi,1)[0]
          print([0,cart_vel,angle,0])
          env.state = np.array([0,cart_vel,angle,0])
          
        action , u = policy(obs)
        #action = env.action_space.sample()
        #print(env.step(action))
        obs, reward, done, _,info = env.step(action)
        state_control_space = np.hstack((obs,u))
        observations = np.vstack((observations,state_control_space))
        #print(obs,action)
        #img = env.render()
        #frames.append(img)
        obs_mat[i,:] = obs
        actions[i] = action
        rewards[i] = reward
        dones[i] = int(done)
        if done and first_done_info == '':
            first_done_info = info

    np.save('observations2.npy',observations)
    record = {'frames': frames, 'obs': obs_mat, 'actions': actions, 'rewards': 
              rewards, 'dones': dones, 'first_done_info':first_done_info}
    return record

def update_scene(num, frames, patch, time_text, obs_mat, actions, cum_rewards, dones):
    patch.set_data(frames[num])
    text = f"frame: {num}"
    text += ", Obs: ({:.3f}, {:.3f}, {:.3f}, {:.3f})\n".format(*obs_mat[num,:])
    text += f"Action: {actions[num]}"
    text += f", Cumulative Reward: {cum_rewards[num]}"
    text += f", Done: {dones[num]}"
    time_text.set_text(text)
    return patch, time_text

def plot_animation(record, repeat=False, interval=40):
    '''record should contain
    frames: list of N frames
    obs: (N, 4) array of observations
    actions: (N, ) array of actions {0, 1}
    rewards: (N, ) array of rewards at each step {0, 1}
    dones: (N, 1) array of dones {0, 1}
    '''
    cum_rewards = np.cumsum(record['rewards'])
    frames = record['frames']
    fig = plt.figure()
    patch = plt.imshow(record['frames'][0])
    ax = plt.gca()
    time_text = ax.text(0., 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch, time_text, record['obs'], record['actions'], cum_rewards, record['dones']),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.show()
    plt.close()
    return anim

if __name__ == '__main__':
    env = CartPoleEnv(render_mode="human")
    #env.screen_width = 3000

    # Initial state
    x0 = np.zeros(4)
    x0[2] = np.pi

    # Initialize controller
    swingup_controller = energy_shaping_controller(env)

    # Simulate and create animation
    swingup_record = record_scenario(env, swingup_controller.swingup_policy, x0, 23000)
    #plot_animation(swingup_record)