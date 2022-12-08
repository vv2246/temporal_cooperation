#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:42:02 2022

@author: vvasiliau
"""



from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
# plt.rcParams.update({'font.size': 20})

def derivative( x, t,  L, amplitude = 0.1 , timescale = 1, phase = np.array([0,np.pi]), K = 0.01, B= 0.2):
    # print(L, x)
    period = np.pi* 2 * timescale  
    dpdt =  - K * np.matmul(L, x) + B  * (amplitude/ period)* np.cos(t/period + phase )
    # print(dpdt[0])
    return dpdt

def get_p_coop(p_coop,timescale, phase,amplitude, t_obs, A, K, B):
    L = -A + np.diag(A.sum(0)) 
    sol  = odeint(derivative, p_coop, t_obs, args = (L, amplitude, timescale, phase, K, B))
    # plt.plot(sol)
    sol = np.clip(sol,0,1)
    return sol


def analytic_solution(x0, A, B, K,t,  amplitude = 0.1 , timescale = 1, phase = np.array([0,np.pi])):
    
    L = np.zeros_like(A)
    np.fill_diagonal(L, A.sum(0))
    L = L -A
    eig = np.linalg.eig(L)
    eigvals = eig[0]
    eigvec = eig[1]
    a0 =np.array(x0.T * eigvec)
    a_t = a0.T * np.exp(-B * eigvals[:,None] * t[None,:])
    
    x_t =np.array( eigvec * a_t)
    
    period = np.pi* 2 * timescale  
    print(K)
    sol  = x_t .T + K* (amplitude/ period)* np.sin((t/period)[:,None]  + phase[None,:] )

    return  np.clip(sol,0,1)

T = 10000 #number of timesteps
step = 1    #to plot every step'th step 
t_obs = np.linspace(0,80*np.pi,T)
# delta3=  0.3
# p0 = np.array([delta1,delta2] )
G = nx.Graph()
edges = [(0,1)]#, (1,2),(0,2)]
for (i,j) in edges:
    G.add_edge(i,j)
A = np.array(nx.adjacency_matrix(G).todense())
L = -A + np.diag(A.sum(0)) 
timescale = 1
# K = 0.004
B= 0.9#1 - K
K = np.array([10,10])

phase= np.array([0,0])


p0 = np.array([1,0] )
sol = analytic_solution(p0, np.matrix(A), B,K, t_obs)

# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(projection='3d')

K1 = 0.1
delta1 = 0.6
# sol  = get_p_coop(p0,timescale,phase,amplitude = 0.1,t_obs= t_obs,A=    A, K = K, B = B)
plt.plot(sol)
# for K1 in np.linspace(0,1,20):
#     K2 = 1-K1
#     for delta1 in np.linspace(0,1,20):
#         delta2 = 1-delta1
#         K = np.array([K1,K2])
#         p0 = np.array([delta1,delta2] )
        
#         sol  = get_p_coop(p0,timescale,phase,amplitude = 0.1,t_obs= t_obs,A=    A, K = K, B = B)

#         # plt.plot(sol)
#         # print(np.mean(sol[300:,:]) - (K1 * delta2 + K2 * delta1 ) / (K1 + K2))
#         ax.scatter(K1, delta1, np.mean(sol[300:,:]) ,c = 'b', marker='o')#,alpha = delta1)
# ax.set_zlabel('$\\langle p_i\\rangle_i$')
# ax.set_xlabel('$K_1$')
# ax.set_ylabel('$\\delta_1$')

# plt.tight_layout()
# plt.savefig("avg_p_varied_delta_i_and_Ki.pdf")
# plt.show()

# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(projection='3d')

# for K1 in np.linspace(0,1,20):
#     for K2 in np.linspace(0,1,20):
#         delta1 = 1
#         delta2 = 0
#         K = np.array([K1,K2])
#         p0 = np.array([delta1,delta2] )
        
#         sol  = get_p_coop(p0,timescale,phase,amplitude = 0.1,t_obs= t_obs,A=    A, K = K, B = B)

#         # plt.plot(sol)
#         # print(np.mean(sol[300:,:]) - (K1 * delta2 + K2 * delta1 ) / (K1 + K2))
#         ax.scatter(K1, K2, np.mean(sol[300:,:]) ,c = 'b', marker='o',alpha = delta1)
# ax.set_zlabel('$\\langle p_i\\rangle_i$')
# ax.set_xlabel('$K_1$')
# ax.set_ylabel('$K_2$')
# plt.tight_layout()
# plt.savefig("avg_p_varied_K_i.pdf")



# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(projection='3d')

# for delta1 in np.linspace(0,1,20):
#     for delta2 in np.linspace(0,1,20):
#         K1 = 1
#         K2 = 1
#         K = np.array([K1,K2])
#         p0 = np.array([delta1,delta2] )
        
#         sol  = get_p_coop(p0,timescale,phase,amplitude = 0.1,t_obs= t_obs,A=    A, K = K, B = B)

#         # plt.plot(sol)
#         # print(np.mean(sol[300:,:]) - (K1 * delta2 + K2 * delta1 ) / (K1 + K2))
#         ax.scatter(delta1, delta2, np.mean(sol[300:,:]) ,c = 'b', marker='o',alpha = K1)
# ax.set_zlabel('$\\langle p_i\\rangle_i$')
# ax.set_xlabel('$\\delta_1$')
# ax.set_ylabel('$\\delta_2$')
# plt.tight_layout()
# plt.savefig("avg_p_varied_delta_i.pdf")

# # K1 * delta2 + K1 * delta3 + K2 * delta1 + K2 * delta3+K3 * delta2 + K3 * delta1