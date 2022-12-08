#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 11:20:44 2022

@author: vvasiliau

Prisoner's dilemma
"""


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.integrate import odeint


def derivative( x, t,  L, amplitude = 0.1 , timescale = 1, phase = np.array([0,np.pi]), K = 0.01, B= 0.2):
    period = np.pi* 2 * timescale  
    dpdt =  - K * np.matmul(L, x) + B  * (amplitude/ period)* np.cos(t/period + phase )
    return dpdt

def get_p_coop(p_coop,timescale, phase,amplitude, t_obs, A, K, B):
    L = -A + np.diag(A.sum(0)) 
    sol  = odeint(derivative, p_coop, t_obs, args = (L, amplitude, timescale, phase, K, B))
    sol = np.clip(sol,0,1)
    return sol
    

class Agent():
    """
    idx - index of an agent in the list of all agents
    p_coop - probabillity of cooperating. If None, draws it from a uniform distribution U(0,1)
    T - number of timesteps. Currently, the time of observation is hardcoded to range between 0 and 10pi
    modulate - if True, modulation of p_coop is performed using parameters
        - timescale
        - phase
        - amplitude
    payoff_mat - payoff matrix
    """
    def __init__(self, idx, p_coops, T,t_obs ,G):
        self.history = []
        self.payoff = []
        self.T = T
        self.p_coop = p_coops[:,idx]
        self.cumm_payoff = []
        self.idx = idx
        self.payoff_mat = np.array([[3,0],[5,1]])
        self.dt = np.diff(t_obs)[0]
        self.nbrs_idx = list(G.neighbors(self.idx))
        self.nnbrs = len(self.nbrs_idx)
        self.N = G.number_of_nodes()
        
    def decision(self , t):
        '''
        Makes a decision to cooperate with probability self.p_coop[t]
        '''
        r = np.random.rand(self.nnbrs)
        res = np.ones(self.nnbrs)
        res[self.p_coop[t] < r] = 0
        res_all = np.zeros(self.N)
        res_all[:] =np.nan
        for i in range(self.nnbrs):
            res_all[self.nbrs_idx[i]] = res[i]
        self.history.append(res_all)
        return res
    
    def update(self, tstart, tfinish):
        
        self.p_coop[tstart: tfinish] = self.p_coop_const 
        if self.modulate:
            self.p_coop[tstart: tfinish]+= self.amplitude*np.sin(self.period*(self.t_obs[tstart: tfinish] +self.phase ))
        self.p_coop = np.clip(self.p_coop,0,1)



    def evolution(self , t, tau, agents):
        '''
        Evolution. At each tau step, copy the winning agent's strategy 
        with some probability
        '''
        if t % tau != 0:
            print("no update this time")
        else:
            nbrs = {i:agents[i].payoff[-1] for i in range(len(agents))}
            a = max(nbrs, key=nbrs.get)
            if  agents[a].payoff[-1]>  self.payoff[-1]:
                p_a = agents[a].p_coop[t]
                r = np.random.rand()
                p_up = (1+np.exp((self.p_coop_const - p_a))/self.K)**(-1)
                ### 1 means cooperating
                # print(p_up)
                if p_up >= r:
                    if self.p_coop_const == p_a:
                        # print("would update, but probabilities are equivalent")
                        pass
                    else:
                        # print("update from ", round(self.p_coop_const,2) ," as other agent got payoff of " , agents[a].payoff[-1], " whereas this one got " , self.payoff[-1],end=". ")
                        self.p_coop_const = p_a
                        self.update(t,self.T)
    
    def calc_payoff(self , agents ):
        '''
        Calculates payoff at a previous timestep
        based on the decisions of neighbours of a node in the graph 
        if iit is a 2-player game, there is one edge and each node is another's neighbour
        '''
        current_pay = 0
        for nbr in self.nbrs_idx:
            agent = agents[nbr]
            d1,d2 =  self.history[-1][nbr], agent.history[-1][self.idx]
            ### 1 means cooperating therefore need to swop around indices for a matrix
            if d1 == 1: idx1 = 0 
            else: idx1 = 1 
            if d2 == 1: idx2 = 0 
            else : idx2 = 1
            pay = self.payoff_mat[idx1,idx2]
            current_pay += pay
        self.payoff.append(current_pay)
        self.cumm_payoff.append(current_pay + sum(self.payoff))
            
    def print_stats(self, ax = None,plot=False,col= "blue"):
        if plot:
            ax.plot(self.payoff, label= "p={:.3f}".format(self.p_coop_const),color=col)
        print("Agent with strategy of cooperating with p= {:.3f} accumulated {:.3f} payoff".format(self.p_coop_const, sum(self.payoff)))
    
   

def calc_mutual_coop(G, agents, t):
    n_coop_games=0
    for i,j in G.edges():
        n_coop_games +=agents[i].history[t][j] * agents[j].history[t][i] 
    return n_coop_games
        
        
plt.rcParams.update({'font.size': 30})

if __name__ =="__main__":
    T = 400 #number of timesteps
    step = 1    #to plot every step'th step 
    t_obs = np.linspace(0,80*np.pi,T)
    N=10
    p0 = np.random.rand(N)
        
    G = nx.erdos_renyi_graph(N,1)
    # G.add_nodes_from(range(N))
    # for (i,j) in edges:
        # G.add_edge(i,j)
    A = np.array(nx.adjacency_matrix(G).todense())
    L = -A + np.diag(A.sum(0)) 
    timescale = 1
    K = 0.001
    for K in np.linspace(0,0.1,10):
        res = []
        for niter in range(100):
            B= 1# - K*20
            
            phase= np.random.rand(N)*2*np.pi
            sol  = get_p_coop(p0,timescale= np.random.rand(N) ,phase = phase ,amplitude = np.random.rand(N)*0.1,t_obs= t_obs,A=    A, K = K, B = B)
                        
            # ===============
            # PD for N agents
            # ===============
            agents = [Agent(idx=i,p_coops=sol,T= T,G = G,t_obs=t_obs)   for i in range(N)]
            
            
            ######## 
            #  Game   
            ########
            coop_game_array = []
            for _ in range(0,T,step):
                for i in range(N):
                    agents[i].decision(_)
                for i in range(N):
                    agents[i].calc_payoff(agents) 
                coop_game_array.append(calc_mutual_coop(G, agents, _))
            res .append( np.mean(coop_game_array[50:]))
        plt.scatter(K, np.mean(res), color = "black")
    
    
    # ######## 
    # #  Plot figure 2   
    # ########
    # fig,ax = plt.subplots(ncols  = 2, figsize= (40,16), gridspec_kw={'width_ratios': [1, 2]})
    # time =np.linspace(0,10*np.pi,T)
    # ax[1].plot(time,agents[0].p_coop,color=colors[0])
    # ax[1].plot(time,agents[1].p_coop,color=colors[1])
    # colors2 =   ["r", "orange", "purple", "green"]
    
    # if phase == 2 and p_coops[0] ==0.5:
        
    #     idx_mid = np.where(abs(agents[0].p_coop - agents[1].p_coop) <0.03)[0]
    #     idx_mid= idx_mid[np.where(~(np.diff(idx_mid[:-1])<3))] #get rid of multples
    #     idx_max0 = np.where(((max(agents[0].p_coop ) - agents[0].p_coop) < 0.01)) #where agent0 prob maxes out
    #     idx_max1 = np.where((max(agents[1].p_coop ) - agents[1].p_coop) < 0.01)#where agent1 prob maxes out
        
    #     ax[0].errorbar( 3, np.mean(np.array(agents[0].payoff)[idx_mid] + np.array(agents[1].payoff)[idx_mid]),
    #         np.std(np.array(agents[0].payoff)[idx_mid] + np.array(agents[1].payoff)[idx_mid]),capsize=10,c=colors2[0],markersize=50,marker='s',markerfacecolor= "none")
    
    #     ax[0].set_xticks(np.linspace(1,3,3))
    #     ax[1].scatter(time[idx_mid],agents[0].p_coop[idx_mid], edgecolors=colors2[0],s=300, facecolors="none",marker="^")
    #     ax[1].scatter(time[idx_mid],agents[1].p_coop[idx_mid], edgecolors=colors2[0],s=300, facecolors="none",marker="^")
    
    
    
    # elif phase == 0 and p_coops[0] ==0.5:
    #     idx_mid = np.where(abs(agents[0].p_coop - agents[1].p_coop) <0.01)[0]
    #     idx_max0 = np.where(((max(agents[0].p_coop ) - agents[0].p_coop) < 0.01)) #where agent0 prob maxes out
    #     idx_max1 = np.where(abs(min(agents[1].p_coop ) - agents[1].p_coop) < 0.01)#where agent1 prob maxes out
        
    #     ax[0].errorbar( 3, np.mean(np.array(agents[0].payoff)[idx_mid] + np.array(agents[1].payoff)[idx_mid]),
    #         np.std(np.array(agents[0].payoff)[idx_mid] + np.array(agents[1].payoff)[idx_mid]),capsize=10,c=colors2[0],markersize=50,marker='s',markerfacecolor= "none")
    
    #     ax[0].set_xticks(np.linspace(1,3,3))
    #     ax[1].scatter(time[idx_mid],agents[0].p_coop[idx_mid], edgecolors=colors2[0],s=300, facecolors="none",marker="^")
    #     ax[1].scatter(time[idx_mid],agents[1].p_coop[idx_mid], edgecolors=colors2[0],s=300, facecolors="none",marker="^")
    
    
    # elif phase == 0 and p_coops[0] ==0 :# or p_coops[1]==0):
    #     print("iim here1" )
    #     idx_max0 = np.where(((max(agents[0].p_coop ) - agents[0].p_coop) < 0.01)) #where agent0 prob maxes out
    #     idx_max1 = np.where((abs(min(agents[1].p_coop ) - agents[1].p_coop)) < 0.01)
    #     ax[0].set_xlim(0,2.5)#ticks(np.linspace(1,2,2))
    
    # elif phase == 2 and p_coops[0] ==0 :# or p_coops[1]==0):
    #     print("iim here")
    #     idx_max0 = np.where(((max(agents[0].p_coop ) - agents[0].p_coop) < 0.01)) #where agent0 prob maxes out
    #     idx_max1 = np.where(((max(agents[1].p_coop ) - agents[1].p_coop) < 0.01))
    #     m1 = int(max(np.diff(idx_max1)[0]))
    #     idx_max1 =idx_max1[0][int(m1/2):][::m1]
    #     ax[0].set_xlim(0,2.5)#ticks(np.linspace(1,2,2))
    
    # ax[1].scatter(time[idx_max0],agents[1].p_coop[idx_max0], edgecolors=colors2[1],s=300, facecolors="none",marker="o")
    # ax[1].scatter(time[idx_max0],agents[0].p_coop[idx_max0], edgecolors=colors2[1],s=300, facecolors="none", marker="o")
    # ax[1].scatter(time[idx_max1],agents[1].p_coop[idx_max1], edgecolors=colors2[2],s=300, facecolors="none",marker="s")
    # ax[1].scatter(time[idx_max1],agents[0].p_coop[idx_max1], edgecolors=colors2[2],s=300, facecolors="none", marker="s")
    
    
    # ax[0].errorbar( 1, np.mean(np.array(agents[0].payoff)[idx_max0] + np.array(agents[1].payoff)[idx_max0]),
    #     np.std(np.array(agents[0].payoff)[idx_max0] + np.array(agents[1].payoff)[idx_max0]),capsize=10,c=colors2[1],markersize=50,marker='^',markerfacecolor= "none")
    
    # ax[0].errorbar( 2, np.mean(np.array(agents[0].payoff)[idx_max1] + np.array(agents[1].payoff)[idx_max1]),
    #     np.std(np.array(agents[0].payoff)[idx_max1] + np.array(agents[1].payoff)[idx_max1]),capsize=10,c=colors2[2],markersize=50,marker='o',markerfacecolor= "none")
    
    # ax[0].set_ylabel("average payoff")
    # ax[1].set_ylabel("probability of cooperating")
    # ax[1].set_xlabel("t")
    
    
    # ax[0].set_xticklabels([])
    
    # # fig.savefig(f"./figures/PD_avgpayoff_N_{N}_p1_{p_coops[0]}_p2_{p_coops[1]}_phase_{phase}_T_{T}_step_{step}.pdf")
    # plt.show()
    
    
    ######## 
    #  PD on a network    
    # ########
        
    # T = 100
    # step = 1
    # n=10
    # p_coops = np.random.rand(n)
    # phase = 0
    
    # G = nx.erdos_renyi_graph(n,0.5) #creates a random graph
    # t_obs = np.linspace(0,20*np.pi,T)
    # agents = [Agent(idx=i,p_coop= p_coops[i],T= T, G= G, modulate= True,timescale= np.random.random(),phase= 2*np.pi*np.random.random(), t_obs= t_obs)  for i in range(n)]
    
    
    
    # for _ in range(0,T,step):
    #     for i in range(n):
    #         agents[i].decision(_)
    #     for i in range(n):
    #         agents[i].calc_payoff(agents) 
    
    
    
    # fig,ax = plt.subplots(ncols  = 3, figsize= (60,16))
    # ax[0].plot(sum(np.array(agents[i].payoff) for i in range(n)))
    # ax[1].plot(sum(agents[i].p_coop for i in range(n)))
    # # ax[1].plot(agents[0].p_coop+agents[1].p_coop,label="summed")
    # ax[1].scatter(range(0,T,step),agents[0].p_coop[::step])
    # ax[1].scatter(range(0,T,step),agents[1].p_coop[::step])
    # # ax[1].plot(agents[1].p_coop)
    # ax[0].legend()
    # ax[0].set_ylabel("payoff at t")
    # ax[1].set_ylabel("probability of cooperating")
    # ax[0].set_xlabel("t")
    # ax[1].set_xlabel("t")
    # # ax[2].scatter(range(0,T,step),agents[0].history)
    # # ax[2].scatter(range(0,T,step),agents[1].history)
    # ax[2].scatter( np.array(agents[0].payoff) + np.array(agents[1].payoff), (agents[0].p_coop+agents[1].p_coop)[::step])
    # ax[2].set_xlabel("Agent 1 cooperates + Agent 2 cooperates")
    # ax[2].set_ylabel("sum of payoff")
    
    # plt.show()
    
    
