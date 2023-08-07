#! /usr/bin/env python3
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Normal
import math
import random
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from geometry_msgs.msg import Point32
from time import sleep
import rospy
import tf2_ros
from math import pi
from geometry_msgs.msg import Twist, Point32, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import time
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float64MultiArray
import os
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


# region define TD3 Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, maxaction):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)

        self.maxaction = maxaction

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))
        a = torch.tanh(self.l3(a)) * self.maxaction
        return a


class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Q_Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, net_width)
        self.l5 = nn.Linear(net_width, net_width)
        self.l6 = nn.Linear(net_width, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
            self,
            env_with_Dead,
            state_dim,
            action_dim,
            max_action,
            gamma=0.99,
            net_width=128,
            a_lr=1e-4,
            c_lr=1e-4,
            Q_batchsize=256
    ):

        self.actor = Actor(state_dim, action_dim, net_width, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
        self.actor_target = copy.deepcopy(self.actor)

        self.q_critic = Q_Critic(state_dim, action_dim, net_width).to(device)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=c_lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)

        self.env_with_Dead = env_with_Dead
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.policy_noise = 0.2 * max_action
        self.noise_clip = 0.5 * max_action
        self.tau = 0.005
        self.Q_batchsize = Q_batchsize
        self.delay_counter = -1
        self.delay_freq = 1

    def select_action(self, state):  # only used when interact with the env
        with torch.no_grad():
            state = torch.FloatTensor(np.array(state).reshape(1, -1)).to(device)
            a = self.actor(state)
        return a.cpu().numpy().flatten()

    def train(self, replay_buffer):
        self.delay_counter += 1
        with torch.no_grad():
            s, a, r, s_prime, dead_mask = replay_buffer.sample(self.Q_batchsize)
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            smoothed_target_a = (
                    self.actor_target(s_prime) + noise  # Noisy on target action
            ).clamp(-self.max_action, self.max_action)

        # Compute the target Q value
        target_Q1, target_Q2 = self.q_critic_target(s_prime, smoothed_target_a)
        target_Q = torch.min(target_Q1, target_Q2)
        '''DEAD OR NOT'''
        if self.env_with_Dead:
            target_Q = r + (1 - dead_mask) * self.gamma * target_Q  # env with dead
        else:
            target_Q = r + self.gamma * target_Q  # env without dead

        # Get current Q estimates
        current_Q1, current_Q2 = self.q_critic(s, a)

        # Compute critic loss
        q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the q_critic
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        if self.delay_counter == self.delay_freq:
            # Update Actor
            a_loss = -self.q_critic.Q1(s, self.actor(s)).mean()
            self.actor_optimizer.zero_grad()
            a_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.delay_counter = -1

    def save(self, episode):
        torch.save(self.actor.state_dict(), "ppo_actor{}.pth".format(episode))
        torch.save(self.q_critic.state_dict(), "ppo_q_critic{}.pth".format(episode))

    def load(self, episode):

        self.actor.load_state_dict(torch.load("ppo_actor{}.pth".format(episode)))
        self.q_critic.load_state_dict(torch.load("ppo_q_critic{}.pth".format(episode)))
        print("model has been loaded...")


# endregion

# region ReplayBuffer
class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.dead = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, reward, next_state, dead):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.dead[self.ptr] = dead  # 0,0,0，...，1

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.dead[ind]).to(self.device)
        )


# endregion

class Env():
    def __init__(self):
        # 用于判断机器人与终点和障碍物的相对位置关系，若为Ture，表示机器人到达终点或撞上障碍物，若为False，表示机器人未到达终点并且没有撞上障碍物
        self.done = False
        # self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.pub_cmd_vel = rospy.Publisher('speed', Twist, queue_size=5)
        self.state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)

    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_world service call failed")
        self.done = False
        time.sleep(2)  # 等待gazebo中的话题发布完成

    def get_state(self):
        # data = rospy.wait_for_message('point', Point32, timeout=5)
        data_min = None
        data_points = None
        try:
            data_min = rospy.wait_for_message("cloud", Point32, timeout=20)
            data_points = rospy.wait_for_message("chatter", Float64MultiArray, timeout=20)
        except:
            pass
        P_x = data_min.x
        P_y = data_min.y
        P_z = data_min.z
        s_state = [data_points.data[0], data_points.data[1], data_points.data[2], data_points.data[3],
                   data_points.data[4], data_points.data[5], data_points.data[6], data_points.data[7],
                   data_points.data[8],
                   data_points.data[9], data_points.data[10], data_points.data[11], data_points.data[12],
                   data_points.data[13], data_points.data[14], data_points.data[15], data_points.data[16],
                   data_points.data[17], data_points.data[18]]
        # rospy.loginfo("P_x:{:.3f},P_y:{:.3f},P_z:{:.3f}".format(P_x,P_y,P_z))
        return s_state, P_x, P_y, P_z

    def setReward(self, P_z, d_des):
        reward = 0
        if P_z > 0.2:
            reward = -0.1
            self.done = False
        elif P_z > 0.15:
            reward = -0.5
            self.done = False
        else:
            reward = -20
            self.done = True

        if (d_des < 0.2):
            self.done = True
        return reward

    def vetor(self, rho, sigma, dire, P_x, P_y, P_z, s, s_stop):
        tfs = self.buffer.lookup_transform("base_footprint", "des", rospy.Time(0))
        d_des = math.sqrt(math.pow(tfs.transform.translation.x, 2) + math.pow(tfs.transform.translation.y, 2))
        v_des_x = tfs.transform.translation.x * 0.15 / d_des
        v_des_y = tfs.transform.translation.y * 0.15 / d_des
        m_vector = math.pow(P_z, 2) * math.exp(P_z / rho)
        n_vector = pow(P_z, 2) * math.exp(P_z / sigma)

        if s == s_stop:
            v_vec_y = v_des_y
            v_vec_x = v_des_x
        else:
            M1 = 1 - P_x * P_x / m_vector + dire * P_x * P_y / n_vector
            M2 = -P_x * P_y / m_vector + dire * P_y * P_y / n_vector
            M3 = -P_x * P_y / m_vector - dire * P_x * P_x / n_vector
            M4 = 1 - P_y * P_y / m_vector - dire * P_x * P_y / n_vector
            v_vec_x = M1 * v_des_x + M2 * v_des_y
            v_vec_y = M3 * v_des_x + M4 * v_des_y

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.1
        vel_cmd.angular.z = 0.3 * math.atan2(v_vec_y, v_vec_x)
        self.pub_cmd_vel.publish(vel_cmd)
        return d_des

    def analyse_action(self, action):
        action_0 = np.round(action[0], 2)
        action_1 = np.round(action[1], 2)

        rho = sigma = (action_0 + 1) * 4
        if action_1 > 0:
            dire = 1
        else:
            dire = -1

        return rho, sigma, dire


if __name__ == '__main__':
    random_seed = 1  # 固定随机初始化

    # Whether the Env has dead state. True for Env like BipedalWalkerHardcore-v3, CartPole-v0.
    # False for Env like Pendulum-v0
    env_with_Dead = True

    rospy.init_node('turlebot3_sac')  # 初始化ros节点，并命名为turtlebot3_dqn
    env = Env()  # 创建初始环境
    # s_stop = [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5]

    state_dim = 19  # 状态维数，这里点云对应19个方向
    action_dim = 2  # 动作维数，考虑只需要映射rho=sigma和dire
    max_action = float(1)  # 限制动作最大值
    min_action = -max_action  # 限制动作最小值

    expl_noise = 0.25  # 加入噪声提高鲁棒性

    Max_episode = 2000000  # 最大轮数
    save_interval = 100  # 保存模型间隔

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)  # 为CPU设置随机数种子
        np.random.seed(random_seed)  # 固定np随机数种子

    kwargs = {
        "env_with_Dead": env_with_Dead,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "gamma": 0.99,  # reward discount
        "net_width": 200,
        "a_lr": 1e-4,  # 学习率
        "c_lr": 1e-4,  # 学习率
        "Q_batchsize": 256,  # 样本数量
    }
    model = TD3(**kwargs)
    load_ep = 1600
    model.load(load_ep)
    #replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(1e6))

    all_ep_r = []

    for episode in range(Max_episode):
        # env.reset()  # 重置gazebo环境,得到初始状态s，s为数组:[x,y],表示最近障碍物的点相对机器人的坐标
        s_stop = [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5]
        #rospy.loginfo("----------第{}轮循环，总共{}轮----------".format(episode, Max_episode))
        episode_reward_sum = 0  # 初始化该循环对应的episode的总奖励

        steps = 0
        expl_noise *= 0.999

        '''Interact & trian'''
        while True:  # 开始一个episode，该episode结束的条件：机器人到达终点 或 机器人撞上障碍物
            steps += 1  # 步数

            s, P_x, P_y, P_z = env.get_state()

            a = model.select_action(np.array(s))  # action
            rho, sigma, dire = env.analyse_action(a)
            rospy.loginfo("step:{},rho:{},sigma:{},dire:{}".format(steps, rho, sigma, dire))
            # 导引向量场方法
            d_des = env.vetor(rho, sigma, dire, P_x, P_y, P_z, s, s_stop)
            s_prime, P_x, P_y, P_z = env.get_state()

            r = env.setReward(P_z, d_des)

            s = s_prime
            episode_reward_sum += r  # 逐步加上一个episode内每个step的reward

            if env.done or steps > 500:  # 判断结束
                env.done = False  # 重置成未完成，以待下一次开始
                rospy.loginfo("第{}轮经过了{}步,running_reward_sum:{:.2f}".format(episode, steps, episode_reward_sum))
                break  # 结束