# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 14:57:21 2018
minimum transmit power: 5dBm/ maximum: 38dBm
bandwidth 10MHz
AWGN power -114dBm
path loss 120.9+37.6log10(d) (dB) d: transmitting distance (km)
using interferers' set and therefore reducing the computation complexity
multiple users / single BS
downlink
localized reward function
@author: mengxiaomao
"""

import scipy
import numpy as np
from scipy import special

dtype = np.float32


class Env_cellular():
    def __init__(self, fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num):
        self.fd = fd  # Frecuencia Doppler maxima
        self.Ts = Ts  # Tiempo de intervalo entre instantes adyacentes
        self.n_x = n_x  # BS in axis x
        self.n_y = n_y  # BS in axis y
        self.L = L  # outer cluster in consideration
        self.C = C  # Ic : Number of Interferers considered on the state
        self.maxM = maxM  # user number in one BS
        self.min_dis = min_dis  # Minimum distance in km
        self.max_dis = max_dis  # Maximum distance in km
        self.max_p = max_p  # Maximum Power in dBm
        self.p_n = p_n  # Maximum Power in dBm
        self.power_num = power_num  # Number of actions

        self.c = 3 * self.L * (self.L + 1) + 1  # adjascent BS
        self.K = self.maxM * self.c  # maximum adjascent users, including itself
        #        self.state_num = 2*self.C + 1    #  2*C + 1
        self.state_num = 3 * self.C + 2  # C + 1                            #  * * * * * * * * * * * * *Why the plus 2?
        self.N = self.n_x * self.n_y  # Number of BS
        self.M = self.N * self.maxM  # maximum number of users
        self.W = np.ones((self.M), dtype=dtype)  # Bandwidth
        self.sigma2 = 1e-3 * pow(10., self.p_n / 10.)  # Power noise in Watts
        self.maxP = 1e-3 * pow(10., self.max_p / 10.)  # maxP in Watts
        self.p_array, self.p_list = self.generate_environment()  # Positional indexes of UEs



    def get_power_set(self, min_p):
        power_set = np.hstack(
            [np.zeros((1), dtype=dtype), 1e-3 * pow(10., np.linspace(min_p, self.max_p, self.power_num - 1) / 10.)])
        return power_set

    def set_Ns(self, Ns):
        self.Ns = int(Ns)

    def generate_H_set(self):
        '''
        Jakes model
        '''

        H_set = np.zeros([self.M, self.K, self.Ns], dtype=dtype)
        pho = np.float32(scipy.special.k0(2 * np.pi * self.fd * self.Ts))

        H_set[:, :, 0] = np.kron(
            np.sqrt(0.5 * (np.random.randn(self.M, self.c) ** 2 + np.random.randn(self.M, self.c) ** 2)),
            np.ones((1, self.maxM), dtype=np.int32))
        for i in range(1, self.Ns):
            H_set[:, :, i] = H_set[:, :, i - 1] * pho + np.sqrt(
                (1. - pho ** 2) * 0.5 * (np.random.randn(self.M, self.K) ** 2 + np.random.randn(self.M, self.K) ** 2))

        path_loss = self.generate_path_loss()
        H2_set = np.square(H_set) * np.tile(np.expand_dims(path_loss, axis=2), [1, 1, self.Ns])

        return H2_set


    def generate_environment(self):
        path_matrix = self.M * np.ones((self.n_y + 2 * self.L, self.n_x + 2 * self.L, self.maxM), dtype=np.int32)

        for i in range(self.L, self.n_y + self.L):  # Loop for generatin adjacent UEs index
            for j in range(self.L, self.n_x + self.L):
                for l in range(self.maxM):
                    path_matrix[i, j, l] = ((i - self.L) * self.n_x + (
                                j - self.L)) * self.maxM + l  # matrix of positions by index
        p_array = np.zeros((self.M, self.K), dtype=np.int32)  # adyacent K UEs by UE
        for n in range(self.N):
            i = n // self.n_x  # ''//'' floor division
            j = n % self.n_x
            Jx = np.zeros((0), dtype=np.int32)
            Jy = np.zeros((0), dtype=np.int32)
            for u in range(i - self.L, i + self.L + 1):
                v = 2 * self.L + 1 - np.abs(u - i)
                jx = j - (v - i % 2) // 2 + np.linspace(0, v - 1, num=v, dtype=np.int32) + self.L
                jy = np.ones((v), dtype=np.int32) * u + self.L
                Jx = np.hstack((Jx, jx))
                Jy = np.hstack((Jy, jy))
            for l in range(self.maxM):
                for k in range(self.c):
                    for u in range(self.maxM):
                        p_array[n * self.maxM + l, k * self.maxM + u] = path_matrix[Jy[k], Jx[k], u]
        p_main = p_array[:, (self.c - 1) // 2 * self.maxM:(self.c + 1) // 2 * self.maxM]
        for n in range(self.N):
            for l in range(self.maxM):
                temp = p_main[n * self.maxM + l, l]
                p_main[n * self.maxM + l, l] = p_main[n * self.maxM + l, 0]
                p_main[n * self.maxM + l, 0] = temp
        p_inter = np.hstack([p_array[:, :(self.c - 1) // 2 * self.maxM], p_array[:, (self.c + 1) // 2 * self.maxM:]])
        p_array = np.hstack([p_main, p_inter])
        p_list = list()
        for m in range(self.M):
            p_list_temp = list()
            for k in range(self.K):
                p_list_temp.append([p_array[m, k]])
            p_list.append(p_list_temp)

        return p_array, p_list

    def generate_path_loss(self):
        p_tx = np.zeros((self.n_y, self.n_x))
        p_ty = np.zeros((self.n_y, self.n_x))
        p_rx = np.zeros((self.n_y, self.n_x, self.maxM))
        p_ry = np.zeros((self.n_y, self.n_x, self.maxM))
        dis_rx = np.random.uniform(self.min_dis, self.max_dis, size=(self.n_y, self.n_x, self.maxM))
        phi_rx = np.random.uniform(-np.pi, np.pi, size=(self.n_y, self.n_x, self.maxM))
        for i in range(self.n_y):
            for j in range(self.n_x):
                p_tx[i, j] = 2 * self.max_dis * j + (i % 2) * self.max_dis
                p_ty[i, j] = np.sqrt(3.) * self.max_dis * i
                for k in range(self.maxM):
                    p_rx[i, j, k] = p_tx[i, j] + dis_rx[i, j, k] * np.cos(phi_rx[i, j, k])
                    p_ry[i, j, k] = p_ty[i, j] + dis_rx[i, j, k] * np.sin(phi_rx[i, j, k])
        dis = 1e10 * np.ones((self.p_array.shape[0], self.K), dtype=dtype)  # This is a dummy value for nonexistent BSs
        lognormal = np.random.lognormal(size=(self.p_array.shape[0], self.K), sigma=8)

        for k in range(self.p_array.shape[0]):
            for i in range(self.c):
                for j in range(self.maxM):
                    if self.p_array[k, i * self.maxM + j] < self.M:
                        bs = self.p_array[k, i * self.maxM + j] // self.maxM
                        dx2 = np.square((p_rx[k // self.maxM // self.n_x][k // self.maxM % self.n_x][k % self.maxM] -
                                         p_tx[bs // self.n_x][bs % self.n_x]))
                        dy2 = np.square((p_ry[k // self.maxM // self.n_x][k // self.maxM % self.n_x][k % self.maxM] -
                                         p_ty[bs // self.n_x][bs % self.n_x]))
                        distance = np.sqrt(dx2 + dy2)
                        dis[k, i * self.maxM + j] = distance

        path_loss = lognormal * pow(10., -(128.1 + 37.6 * np.log10(dis)) / 10.)

        return path_loss

    def calculate_rate(self, P):

        H2 = self.H2_set[:, :, self.count]  # This give us the H2 of each UEs interferer on interval self.count
        p_extend = np.concatenate([P, np.zeros((1), dtype=dtype)], axis=0)  # vvvv p_extend vvvv
        p_matrix = p_extend[self.p_array]
        path_main = H2[:, 0] * p_matrix[:, 0]  # The first column contain the gains and the power (1-to-maxMxBSs)
        path_inter = np.sum(H2[:, 1:] * p_matrix[:, 1:], axis=1)  # Vector of interferences (1-to-maxMxBSs)
        maxC = 1000. # This Represent the capped sinr(30) = 10**(30/10) = 1000.
        sinr = np.minimum(path_main / (path_inter + self.sigma2), maxC)        # capped sinr calculation (1-to-maxMxBSs)

        index = np.where(sinr <= 10 ** ((-3.8279) / 10))[0]  # SINR THRESHOLD
        rate = self.W * np.log2(1. + sinr)  # rates (1-to-maxMxBSs)
        rate[index] = 0

        rate_extend = np.concatenate([rate, np.zeros((1), dtype=dtype)], axis=0)  # Adding 0 rate value for not

        reward_rate = np.zeros(self.M)
        for i in np.arange(0,self.M):
            X = self.p_array[i,1:]
            Neighbors = np.extract(X != self.M, X)
            index_aux = np.where(self.p_array[Neighbors,:] == i)[1]
            inter_substract = H2[Neighbors,index_aux] * p_matrix[Neighbors,index_aux]
            sinr_k = np.minimum((path_main[Neighbors] / ((path_inter[Neighbors] - inter_substract) + self.sigma2)),maxC)
            index_k = np.where(sinr_k <= 10 ** ((-3.8279) / 10))[0]  # SINR THRESHOLD
            C_k = np.log2(1. + sinr_k)  # rates (1-to-maxMxBSs)
            C_k[index_k] = 0


            pi_aux = (C_k-rate[Neighbors])
            reward_rate[i] = rate[i] - np.sum(pi_aux)

        rate_matrix = rate_extend[self.p_array]  # consider all UEs rates on state
        '''
        Calculate reward, sum-rate
        '''
        sum_rate = np.mean(rate)  # average rate of all (UEsxBs)

        return p_matrix, rate_matrix, reward_rate, sum_rate

    def generate_next_state(self, H2, p_matrix, rate_matrix):
        sinr_norm_inv = H2[:, 1:] / np.tile(H2[:, 0:1], [1, self.K - 1])
        sinr_norm_inv = np.log2(1. + sinr_norm_inv)  # log representation
        indices1 = np.tile(
            np.expand_dims(np.linspace(0, p_matrix.shape[0] - 1, num=p_matrix.shape[0], dtype=np.int32), axis=1),
            [1, self.C])
        indices2 = np.argsort(sinr_norm_inv, axis=1)[:, -self.C:]
        sinr_norm_inv = sinr_norm_inv[indices1, indices2]
        p_last = np.hstack([p_matrix[:, 0:1], p_matrix[indices1, indices2 + 1]])
        rate_last = np.hstack([rate_matrix[:, 0:1], rate_matrix[indices1, indices2 + 1]])
        s_actor_next = np.hstack([sinr_norm_inv, p_last, rate_last])

        '''
        Generate state for critic
        '''
        s_critic_next = H2
        return s_actor_next, s_critic_next

    def counter_step(self):
        self.count += 1

    def reset(self, seed=None):
        np.random.seed(seed)

        self.count = 0  # Intervals Counter initialization
        self.H2_set = self.generate_H_set()  # Reset UEs Location and generate correlated gains on Ns intervals
        P = np.zeros([self.M], dtype=dtype)  # Power is set to 0s for the initial interval of size equals UEs

        p_matrix, rate_matrix, _, _ = self.calculate_rate(P)  # Zero Matrix initialization, since P = zeros
        H2 = self.H2_set[:, :, self.count]  # Channel gain at zero interval (maxM*BSxInterferers*MaxM)
        s_actor, s_critic = self.generate_next_state(H2, p_matrix, rate_matrix)


        return s_actor, s_critic, self.H2_set


    def testing_reset(self, seed=None):
        np.random.seed(seed)

        self.count = 0  # Intervals Counter initialization
        self.H2_set = self.generate_H_set()  # Reset UEs Location and generate correlated gains on Ns intervals

        np.random.seed(42)
        rand_idx = np.random.permutation(self.H2_set.shape[2])
        self.H2_set = shuffled_a = self.H2_set[:, :,rand_idx]

        P = np.zeros([self.M], dtype=dtype)  # Power is set to 0s for the initial interval of size equals UEs

        p_matrix, rate_matrix, _, _ = self.calculate_rate(P)  # Zero Matrix initialization, since P = zeros
        H2 = self.H2_set[:, :, self.count]  # Channel gain at zero interval (maxM*BSxInterferers*MaxM)
        s_actor, s_critic = self.generate_next_state(H2, p_matrix, rate_matrix)

        return s_actor, s_critic, self.H2_set

    def reset_bench(self, seed=None):
        np.random.seed(seed)

        self.count = 0  # Intervals Counter initialization
        self.H2_set, self.H2_set_2 = self.generate_H_set_2()  # Reset UEs Location and generate correlated gains on Ns intervals

        return self.H2_set, self.H2_set_2


    def reset__(self, seed=None):
        np.random.seed(seed)
        self.count = 0
        self.H2_set = self.generate_H_set()
        H2 = self.H2_set[:, :, self.count]

        return H2


    def step(self, P):
        p_matrix, rate_matrix, reward_rate, sum_rate = self.calculate_rate(P)
        self.count = self.count + 1
        H2_next = self.H2_set[:, :, self.count]
        s_actor_next, s_critic_next = self.generate_next_state(H2_next, p_matrix, rate_matrix)
        return s_actor_next, s_critic_next, reward_rate, sum_rate

    def calculate_sumrate_fp(self, P, H2):
        p_extend = np.concatenate([P, np.zeros((1), dtype=dtype)], axis=0)
        p_matrix = p_extend[self.p_array]
        path_main = H2[:, 0] * p_matrix[:, 0]
        path_inter = np.sum(H2[:, 1:] * p_matrix[:, 1:], axis=1)

        maxC = 1000.
        sinr = np.minimum(path_main / (path_inter + self.sigma2), maxC)    #capped sinr
        index = np.where(sinr <= 10 ** ((-3.8279) / 10))[0]  # SINR THRESHOLD

        rate = self.W * np.log2(1. + sinr)  # rates (1-to-maxMxBSs)
        rate[index] = 0

        sum_rate = np.mean(rate)
        return sum_rate


    def calculate_sumrate(self, P):
        #maxC = 1000.

        H2 = self.H2_set[:, :, self.count]
        p_extend = np.concatenate([P, np.zeros((1), dtype=dtype)], axis=0)
        p_matrix = p_extend[self.p_array]
        path_main = H2[:, 0] * p_matrix[:, 0]
        path_inter = np.sum(H2[:, 1:] * p_matrix[:, 1:], axis=1)

        maxC = 1000.
        sinr = np.minimum(path_main / (path_inter + self.sigma2), maxC)    #capped sinr

        index = np.where(sinr <= 10 ** ((-3.8279) / 10))[0]  # SINR THRESHOLD

        rate = self.W * np.log2(1. + sinr)  # rates (1-to-maxMxBSs)
        rate[index] = 0

        sum_rate = np.mean(rate)


        return sum_rate

    def step__(self, P):
        reward_rate = list()

        for p in P:
            reward_rate.append(self.calculate_sumrate(p))
        self.count = self.count + 1
        H2_next = self.H2_set[:, :, self.count]

        return H2_next, reward_rate

