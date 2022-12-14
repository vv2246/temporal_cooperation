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
import random
from scipy.integrate import odeint


def derivative( x, t,  L, amplitude  , timescale , phase , K):#, B= 0.2):
    period = np.pi* 2 * timescale  
    dpdt =  - K * np.matmul(L, x) + (amplitude/period)* np.cos(t/period + phase ) 
    return dpdt

def get_p_coop(p_coop,timescale, phase,amplitude, t_obs, A, K):#, B):
    L = -A + np.diag(A.sum(0)) 
    # print(phase)
    sol  = odeint(derivative, p_coop, t_obs, args = (L, amplitude, timescale, phase, K))#, B))
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
    def __init__(self, idx, p_coops, T,t_obs ,G, K_f):
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
        self.p0 = np.ones(len(t_obs)) * self.p_coop[0]
        self.K_f = K_f
        
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
    
    def update(self,new_p_coops, tstart):
        self.p_coop[tstart: ] = new_p_coops[:,self.idx]



    def evolution(self , t, tau, agents):
        '''
        Evolution. At each tau step, copy the winning agent's strategy 
        with some probability
        '''
        # if t % tau != 0:
        #     print("no update this time")
        # else:
        nbrs = {i:agents[i].payoff[-1] for i in self.nbrs_idx}
        a = random.choice(list(nbrs.keys()))#max(nbrs, key=nbrs.get)
        # if  agents[a].payoff[-1]>  self.payoff[-1]:
        p_a = agents[a].p_coop[t]
        r = np.random.rand()
        p_up = (1+np.exp((self.p_coop[t] - p_a))/self.K_f)**(-1)
        if p_up >= r:
            # print(f"at {t} update from ", round(self.p_coop[t],2) ,"to" ,p_a, " as other agent got payoff of " , agents[a].payoff[-1], " whereas this one got " , self.payoff[-1],end="\n ")
            
            self.p_coop[t] = p_a
            
    def calc_payoff(self , agents ):
        '''
        Calculates payoff at a previous timestep
        based on the decisions of neighbours of a node in the graph 
        if iit is a 2-player game, there is one edge and each node is another's neighbour
        '''
        debug =0
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
            if debug:
                print("node:", self.idx, "self:" ,d1, " nbr: ", d2, "pay to self: ", pay)
            current_pay += pay
        self.payoff.append(current_pay/self.nnbrs)
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
        
        
        
def test():
    edges = [(0,1),(1,2)]
    G = nx.Graph()
    G.add_edges_from(edges)
    T = 100
    step =1 
    N = G.number_of_nodes()
    t_obs = np.linspace(0,80*np.pi,T)
    p0 = np.array([0,0.5,1])
    A = np.array(nx.adjacency_matrix(G).todense())
    timescale = 1
    K = 0.01
    B= 1# - K*20
    amplitude = 0.1
    phase= np.zeros(N)
    sol  = get_p_coop(p0,timescale= timescale ,phase = phase ,amplitude = amplitude,t_obs= t_obs,A=    A, K = K, B = B)
    # plt.plot(sol)
    agents = [Agent(idx=i,p_coops=sol,T= T,G = G,t_obs=t_obs, K_f=0.1)   for i in range(N)]
    for i in range(N):
        plt.plot(agents[i].p_coop)
    coop_game_array = []
    tau = 50
    # print("test to check payoffs are OK")
    for _ in range(1,T,step):
        for i in range(N):
            agents[i].decision(_)
            # print( agents[i].p_coop[_],end=",")
        # print("\n")
        for i in range(N):
            agents[i].calc_payoff(agents) 
        if _ % tau == 0:
            for i in range(N):
                agents[i].evolution(_, 2, agents)
            pt = np.array([agents[i].p_coop[_] for i in range(N)])
            new_p_coops = get_p_coop(pt, timescale, phase, amplitude, t_obs[_:], A, K, B)
            
            for i in range(N):
                agents[i].update(new_p_coops ,_)
    for i in range(N):
        plt.plot(agents[i].p_coop)
            
        # break
    
        # coop_game_array.append(calc_mutual_coop(G, agents, _))
    # print( np.mean(coop_game_array[:2000]), np.mean(coop_game_array[2000:]))
   
    


 
def run_n_player_game(K, p0, inphase = True, evolution = False, 
                      amplitude = 0.1, T= 400, tau = 10, gdensity=1, 
                      T_mult=80, K_f= 0.1, graph_type= "ER",return_graph =False):
     
    step = 1    #to plot every step'th step 
    t_obs = np.linspace(0,T_mult*np.pi,T)
    N=p0.shape[0]
    connected = False
    if graph_type == "ER":
        while connected == False:
            G = nx.erdos_renyi_graph(N,gdensity)
            connected = nx.is_connected(G)
    elif graph_type =="BA":
        G = nx.barabasi_albert_graph(N, 3)
    elif graph_type == "lattice":
        G = nx.convert_node_labels_to_integers(nx.grid_graph(dim=(4, 5, 5)))
        
    # elif graph_type == ""
    A = np.array(nx.adjacency_matrix(G).todense())
    timescale = 1
    if inphase:
        phase= np.zeros(N)
    else:
        if N==2:
            phase = np.array([0,np.pi/timescale])
        else:
            phase= np.random.rand(N)*np.pi/timescale
    sol  = get_p_coop(p0,timescale,phase, amplitude = amplitude, t_obs= t_obs, A= A, K = K)
    # plt.plot(sol)
    # plt.show()
    # if simulate:
    agents = [Agent(idx=i,p_coops=sol,T= T,G = G,t_obs=t_obs, K_f=K_f)   for i in range(N)]
    ######## 
    #  Game   
    ########
    for _ in range(0,T,step):
        for a in agents:
            a.decision(_)
        for a in agents:
            a.calc_payoff( agents)
        if evolution:
            if _ % tau == 0:
                for i in range(N):
                    # print("bla")
                    agents[i].evolution(_, 2, agents)
            pt = np.array([agents[i].p_coop[_] for i in range(N)])
            new_p_coops = get_p_coop(pt, timescale, phase, amplitude, t_obs[_:], A, K)
                    
            for i in range(N):
                agents[i].update(new_p_coops ,_)
    if return_graph == False:
        return agents
    else:
        return agents, G

        
plt.rcParams.update({'font.size': 30})

if __name__ =="__main__":
    
    # test()
    Klist = np.linspace(0,0.04,5)
    graph_type = "ER"
    fig,ax = plt.subplots(figsize= (10,8))
    col = {"ER": "deepskyblue", "BA": "tomato", "lattice": "limegreen" }
    shape = {"ER": "^", "BA": "o", "lattice": "*" }
    
    for graph_type in ["ER", "BA", "lattice"]:
        N = 100
        res_K = []
        res_K_std = []
        evolution = False
        for K in Klist:
            res = []
            print(K)
            for niter in range(10):
                print(niter,end=",")
                # if K_random:
                
                #    agents, G= run_n_player_game(K=K * np.random.rand(N), p0 = np.linspace(0,1,1000), inphase = True,evolution = evolution,
                #                           amplitude=0.0, T =200,T_mult = 40, tau=10,gdensity = 6/1000, return_graph= True, graph_type=graph_type)
                # else:
                    
                agents, G= run_n_player_game(K=K , p0 = np.linspace(0,1,N), inphase = True,evolution = evolution,
                                          amplitude=0.1, T =200,T_mult = 40, tau=1,gdensity = 6/N, return_graph= True, graph_type=graph_type)
               
               
                f_C= 0
                for ti in range(100,200):
                    f_C += calc_mutual_coop(G, agents, ti)#sum([np.nansum(agents[i].history[ti]) for i in range(N)])sum([np.nansum(agents[i].history[ti]) for i in range(N)]) #calc_mutual_coop(G, agents, ti)#sum([np.nansum(agents[i].history[ti]) for i in range(N)]) #
                res.append(f_C/G.number_of_edges()/100/2)
         
            # for i in range(N):
            #     plt.plot(agents[i].p_coop)
            # plt.show()
            res_K.append(np.mean(res))
            res_K_std.append(np.std(res))
        # pass
        # import pandas as pd
        # df = pd.DataFrame([Klist, res_K,res_K_std]).T
        # df.columns = ["K","Mean","Error"]
        # df.to_csv(f"K_random_{K_random}_vs_f_C_graph_{graph_type}_evo_{evolution}.csv")
    
        ax.errorbar(Klist, res_K, yerr= res_K_std, fmt= shape[graph_type],capsize = 5,label= graph_type, color = col[graph_type],ms=20)
    ax.legend(loc=4)
    ax.set_ylabel("$f_C$")
    ax.set_xlabel("$K$")
    ax.set_ylim(0.0,.2)
    plt.tight_layout()
    plt.savefig("network_f_C_vs_K_no_phase.pdf")
    # T = 400 #number of timesteps
    # step = 1    #to plot every step'th step 
    # t_obs = np.linspace(0,80*np.pi,T)
    # N=10
    # p0 = np.random.rand(N)
        
    # G = nx.erdos_renyi_graph(N,0.5)
    # A = np.array(nx.adjacency_matrix(G).todense())
    # timescale = 1
    # K = 0.001
    # B = 1
    # phase= np.zeros(N)
    # amplitude = 0.1
    # sol  = get_p_coop(p0,timescale= timescale ,phase = phase ,amplitude = amplitude,t_obs= t_obs,A=    A, K = K, B = B)
    # agents = [Agent(idx=i,p_coops=sol,T= T,G = G,t_obs=t_obs, K_f=0.1)   for i in range(N)]
    
    # tau = 500
    # for _ in range(1,T,step):
    #     for i in range(N):
    #         agents[i].decision(_)
    #     for i in range(N):
    #         agents[i].calc_payoff(agents) 
    #     if _ % tau == 0:
    #         for i in range(N):
    #             agents[i].evolution(_, 2, agents)
    #         pt = np.array([agents[i].p_coop[_] for i in range(N)])
    #         new_p_coops = get_p_coop(pt, timescale, phase, amplitude, t_obs[_:], A, K, B)
            
    #         for i in range(N):
    #             agents[i].update(new_p_coops ,_)
    
    # for K in np.linspace(0,0.1,10):
    #     res = []
    #     for niter in range(100):
    #         B= 1# - K*20
            
    #         phase= np.random.rand(N)*2*np.pi
    #         sol  = get_p_coop(p0,timescale= np.random.rand(N) ,phase = phase ,amplitude = np.random.rand(N)*0.1,t_obs= t_obs,A=    A, K = K, B = B)
                        
    #         # ===============
    #         # PD for N agents
    #         # ===============
    #         agents = [Agent(idx=i,p_coops=sol,T= T,G = G,t_obs=t_obs)   for i in range(N)]
            
            
    #         ######## 
    #         #  Game   
    #         ########
    #         coop_game_array = []
    #         for _ in range(0,T,step):
    #             for i in range(N):
    #                 agents[i].decision(_)
    #             for i in range(N):
    #                 agents[i].calc_payoff(agents) 
    #             coop_game_array.append(calc_mutual_coop(G, agents, _))
    #         res .append( np.mean(coop_game_array[50:]))
    #     plt.scatter(K, np.mean(res), color = "black")
    
    
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
    
    
