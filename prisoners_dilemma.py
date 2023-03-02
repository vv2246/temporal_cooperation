#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 11:20:44 2022

@author: vvasiliau

Prisoner's dilemma

module load python/3.7.4

python
import os
import numpy as np 
fig_no = 1
Klist = np.linspace(0,0.05,6)
for K in Klist: 
    for graph_type in ["ER","BA","SW"]:
        os.system(f'bsub -W "4:00" -n 4 -R "rusage[mem=10000]" "python prisoners_dilemma.py {K} {graph_type} {fig_no}"')
 
    
bsub -W "24:00" -n 4 -R "rusage[mem=10000]" python prisoners_dilemma.py  0 ER 2
bsub -W "24:00" -n 4 -R "rusage[mem=10000]" python prisoners_dilemma.py  0 BA 2
bsub -W "24:00" -n 4 -R "rusage[mem=10000]" python prisoners_dilemma.py  0 SW 2
 
"""


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from scipy.integrate import odeint
import sys
import pandas as pd

def derivative( x, t,  L, amplitude  , timescale , phase , K):
    period =  timescale
    dpdt =  - K * np.matmul(L, x) + (2*np.pi * amplitude/period)* np.cos(2*np.pi * (t + phase)/period ) 
    return dpdt

def get_p_coop(p_coop,timescale, phase,amplitude, t_obs, A, K):
    L = -A + np.diag(A.sum(0)) 
    sol  = odeint(derivative, p_coop, t_obs, args = (L, amplitude, timescale, phase, K))
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
    def __init__(self, idx, p_coops, T,t_obs ,G, K_f, alpha = 1):
        self.history = []
        self.payoff = []
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
        self.alpha = alpha
        
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
        nbrs = {i:agents[i].payoff[-1] for i in self.nbrs_idx}
        a = random.choice(list(nbrs.keys()))
        p_a = agents[a].p_coop[t]
        r = np.random.rand()
        p_up = (1+np.exp((self.p_coop[t] - p_a))/self.K_f)**(-1)
        if p_up >= r:
            # self.p_coop[t] = p_a
            self.p_coop[t] += self.alpha*(p_a - self.p_coop[t])
            
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
    sol  = get_p_coop(p0,timescale= timescale ,phase = phase ,amplitude = amplitude,t_obs= t_obs,A=  A, K = K, B = B)
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
            



def run_n_player_game(K, p0, inphase = True, evolution = False, 
                      amplitude = 0.1, T= 40, tau = 10, gdensity=1, 
                       K_f= 0.1, graph_type= "ER",return_graph =False, G=None,phase_scale=1,
                      timescale = 1, alpha = 1):
     
    t_obs = np.linspace(0,T,T*10 +1)
    N=p0.shape[0]
    connected = False
    if G == None:
        if graph_type == "ER":
            while connected == False:
                G = nx.erdos_renyi_graph(N,gdensity)
                connected = nx.is_connected(G)
        elif graph_type =="BA":
            G = nx.barabasi_albert_graph(N, 3)
        elif graph_type == "SW":
            G = nx.watts_strogatz_graph(N, 6, 0.05 )
    A = np.array(nx.adjacency_matrix(G).todense())
    if inphase:
        phase= np.zeros(N)
    else:
        if N==2:
            phase = np.array([0,timescale/2])*phase_scale
        else:
            if phase_scale == 0:
                phase= np.zeros(N)
                
            elif phase_scale ==1:
                phase= np.random.rand(N)*timescale
            else:
                phase = np.random.beta(1/phase_scale, 1/phase_scale,N) * timescale
            
    sol  = get_p_coop(p0,timescale,phase, amplitude = amplitude, t_obs= t_obs, A= A, K = K)
    agents = [Agent(idx=i,p_coops=sol,T= T,G = G,t_obs=t_obs, K_f=K_f, alpha =alpha)   for i in range(N)]
    
    ######## 
    #  Game   
    ########
    for _ in range(0,len(t_obs)):
        for a in agents:
            a.decision(_)
        for a in agents:
            a.calc_payoff( agents)
        if evolution:
            if _ % tau == 0 and _ !=0:
                # print(_)
                for i in range(N):
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
    
    ###
    # Effect of neural syncrhony
    ###
    N = 100
    K, graph_type, fig_no  = 3, "ER", 2# sys.argv[1:]
    # print("bla")
    # print(K)
    K = float(K)
    fig_no = int(fig_no)
    if fig_no ==1:
        res = []
        for i in range(100):
            print(i,end = ",")
            agents, G= run_n_player_game(K=K , p0 = np.random.rand(N),
                                     inphase = False,evolution = False,
                                     amplitude=0., T =20, tau=1,gdensity = 6/N, 
                                     return_graph= True, graph_type=graph_type,phase_scale=0.1,timescale = 5)
            
            payoff0 = [agents[i].payoff[0] for i in range(N)]
            payofflast = [agents[i].payoff[-1] for i in range(N)]
            res.append([K,np.mean(payoff0),np.std(payoff0),np.mean(payofflast),np.std(payofflast),graph_type])
        df = pd.DataFrame(res)
        df.columns=["K", "payoff_mean0","payoff_std0","payoff_meanlast","payoff_stdlast","graph"]
        df.to_csv("figures/fig5/payoff_first_last_1.csv", header = False, mode= "a+")
        
        # plotting
        fig,ax = plt.subplots(ncols = 3,figsize= (16,6), sharey = True)
        graphs = ["ER","BA", "SW"]
        for i in [0,1,2]:
            graph_type = graphs[i]
            df = pd.read_csv("figures/fig5/payoff_first_last_1.csv", header= None)#, nan=)
            df.columns=["i","K", "payoff_mean0","payoff_std0","payoff_meanlast","payoff_stdlast","graph"]
            df = df [ df.graph ==graph_type]
            dfmean = df.groupby("K").mean()
            dferr = df.groupby("K").std()
            ax[i].errorbar(dfmean.index, dfmean.payoff_std0, yerr= dferr.payoff_std0,capsize = 5, markersize = 15,marker='o', label = "t=0", color= "r")
            ax[i].errorbar(dfmean.index, dfmean.payoff_stdlast,yerr= dferr.payoff_stdlast,capsize = 5, markersize = 15,marker='^',label = "t=200", color = "b")
            if graph_type =="SW":
                graph_type = "WS"
            ax[i].set_title(graph_type)
            ax[i].set_xlabel("$K$")
            ax[i].set_ylim(0.7,1.1)
        ax[0].set_ylabel("$\\sigma_i[\\pi_i^t]$")
        plt.tight_layout()
        fig.savefig("figures/fig5/network_result_variance_vs_K.pdf")
    
    
    if fig_no==2:
        phase_Scale = [0,0.5,1]
        res = []
        niter = 100
        evo= True
        for s in phase_Scale : 
            for K in np.linspace(0,0.05,6):
                timescale = 0
                for i in range(niter):
                    if timescale == 0:
                        tscl = np.ones(N)
                    else:
                        tscl = np.random.beta(timescale**(-1),timescale**(-1),N)+0.5
                    agents, G = run_n_player_game(K=K , p0 = np.random.rand(N),#int(0,2,N), 
                                             inphase = False,evolution = evo,
                                             amplitude=0.1, T =20, tau=1, gdensity = 6/N, 
                                             return_graph= True, graph_type=graph_type, phase_scale = s, timescale=tscl*5)
                    E = G.number_of_edges()
                    for t in range(agents[0].p_coop.shape[0]):
                        res.append([s,timescale,K, t, calc_mutual_coop(G, agents, t)/E] )
                        
           
        res1 = []
        for i in range(niter):
            agents, G = run_n_player_game(K=0 , p0 = np.random.rand(N),
                                     inphase = False,evolution = evo,
                                     amplitude=0, T =20, tau=1, gdensity = 6/N, 
                                     return_graph= True, graph_type=graph_type)
            E = G.number_of_edges()
            for t in range(agents[0].p_coop.shape[0]):
                res1.append( [t,calc_mutual_coop(G, agents, t)/E])
           
        df = pd.DataFrame(res) 
        df.columns = ["phase_scale","timescale", "K","t", "fC"]
        df.to_csv(f"figures/fig6/phase_result_network_{graph_type}_2_{evo}.csv")     
        df = pd.DataFrame(res1)     
        df.columns = [ "t", "fC"]
        df.to_csv(f"figures/fig6/phase_result_network_{graph_type}_null_2_{evo}.csv")
        import matplotlib.colors as colors
        import seaborn as sns
        for graph_type in ["ER"]:
            fig,ax = plt.subplots(figsize=(9,7))
            df = pd.read_csv(f"phase_result_network_{graph_type}_2_{evo}.csv")
            df = df[df.t >=150]
            df = df.groupby(["phase_scale","K"]).mean()
            dfnull = pd.read_csv(f"phase_result_network_{graph_type}_null_2_{evo}.csv")
            df["fC"] = df["fC"]/dfnull.fC.mean()
            df = df[["fC"]]
            df = df.reset_index()
            table = df.pivot("phase_scale","K","fC")
            divnorm = colors.TwoSlopeNorm(vmin=0.6,  vcenter= 1,vmax=1.05)
            ax = sns.heatmap(table,norm =divnorm,cmap='BrBG', cbar_kws={"ticks":[0.6,1,1.05]})
            ax.invert_yaxis()
            ax.set_title("$\\frac{f_C(a=0.1,K)}{f_C(a=0,K=0)}$")
            ax.set_ylabel("$\\sigma_i^2[\\theta_i]$")
            ax.set_xlabel("$K$")
            ax.set_yticklabels([0, 1.2, 2.1],rotation=0)
            ax.set_xticklabels([0, 0.01, 0.02,0.03, 0.04,0.05],rotation=90)
            plt.tight_layout()
            fig.savefig(f"figures/fig6/phase_result_network_{graph_type}_2_{evo}.pdf")
            plt.show()
               



