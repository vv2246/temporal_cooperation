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
    def __init__(self, idx, p_coop, T, modulate= True, timescale = 100,phase=0,amplitude = 0.35 ):
        self.history = []
        self.payoff = []
        if p_coop == None:
            self.p_coop_const= np.random.rand()
        else:
            self.p_coop_const = p_coop
        if modulate:
            period = np.pi* 2 * timescale  
            phase = phase#np.random.rand()*T#np.pi/2  * np.random.rand()
            self.p_coop = self.p_coop_const + amplitude*np.sin(period*(np.linspace(0,10*np.pi,T) +phase ))
            self.p_coop = np.clip(self.p_coop,0,1)
        else: 
            self.p_coop = self.p_coop_const *  np.ones(T)  
        self.cumm_payoff = []
        self.idx = idx
        self.payoff_mat = np.array([[3,0],[5,1]])
        
    def decision(self , t):
        '''
        Makes a decision to cooperate with probability self.p_coop[t]
        '''
        r = np.random.rand()
        ### 1 means cooperating
        if self.p_coop[t] > r:
            res = 1
        else: res = 0
        self.history.append(res)
            
        return res
    
    def calc_payoff(self, graph , agents ):
        '''
        Calculates payoff at a previous timestep
        based on the decisions of neighbours of a node in the graph 
        if iit is a 2-player game, there is one edge and each node is another's neighbour
        '''
        current_pay = 0
        for nbr in graph.neighbors(self.idx):
            agent = agents[nbr]
            d1,d2 =  self.history[-1], agent.history[-1]
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
    
   
plt.rcParams.update({'font.size': 60})#, "font.family":"helvetica"})

 
# for p_coops in [[1,0],[0.5,0.5]]:
#     for phase in [0,2]:
T = 500     #number of timesteps
step = 1    #to plot every step'th step 
p_coops = [0.5,0.5]    #probabilities to ccooperate
phase = 0  #phase shift. If one agen'ts pahse =0, and anothers = 2, they are out-of-phase, becauuse the period is 4
agents = [Agent(idx=0,p_coop= p_coops[0],T= T, modulate= True,timescale= 0.25,phase= 0) ,
          Agent(idx =1,p_coop= p_coops[1],T= T, modulate=True,timescale = 0.25,phase =phase, amplitude=0.2)]

colors = {0: "teal",  1 : "orange"} #colors associated with agent 0 and agent  1

######## 
#  Graph   
########
G = nx.Graph()
N=len(agents)
edges = [(0,1)]
G.add_nodes_from(range(len(agents)))
for (i,j) in edges:
    G.add_edge(i,j)
    
######## 
#  Game   
########
for _ in range(0,T,step):
    action1, action2 = agents[0].decision(_),agents[1].decision(_) 
    for a in agents:
        a.calc_payoff(G, agents)
    
######## 
#  Plot figure 1   
########
fig,ax = plt.subplots(ncols  = 3, figsize= (47,13))
# agents[0].print_stats(ax[0],col=colors[0])
# agents[1].print_stats(ax[0],col=colors[0])
ax[0].plot(np.linspace(0,10*np.pi,T),agents[0].payoff,color = colors[0],label= "p={:.1f}".format(agents[0].p_coop_const))
ax[0].plot(np.linspace(0,10*np.pi,T),agents[1].payoff,color = colors[1], label= "p={:.1f}".format(agents[1].p_coop_const))
ax[1].plot(np.linspace(0,10*np.pi,T),agents[0].p_coop, alpha=0.1, color= colors[0])
ax[1].plot(np.linspace(0,10*np.pi,T),agents[1].p_coop,alpha =0.1, color= colors[1])
# ax[1].plot(agents[0].p_coop+agents[1].p_coop,label="summed")
ax[1].scatter(np.linspace(0,10*np.pi,T),agents[0].p_coop[::step],edgecolors = colors[0],marker="^", facecolors="none",s=250)
ax[1].scatter(np.linspace(0,10*np.pi,T),agents[1].p_coop[::step],edgecolors=colors[1],marker="o",facecolors="none",s=250)
# ax[1].plot(agents[1].p_coop)
# ax[0].legend(loc =1)
ax[0].set_ylabel("payoff")
ax[1].set_ylabel("probability of cooperating")
ax[0].set_xlabel("t")
ax[1].set_xlabel("t")
ax[2].scatter(np.linspace(0,10*np.pi,T), np.array(agents[0].history)*0.6667 + np.array(agents[1].history)*1.3333, edgecolors="k",s= 250,marker="o",facecolors="none")#, (agents[0].p_coop+agents[1].p_coop)[::step])

ax[0].set_yticks(range(0,6,1))
ax[1].set_ylim(-0.1,1.1)
ax2 = ax[2].twinx()
ax[2].set_yticks(np.linspace(0,2,4))
ax[2].set_yticklabels(["D", "C","D","C"])
ax[2].tick_params(colors=colors[0], which='both' , axis = "y") 
ax[2].set_ylabel("strategy")
ax2.set_yticks(np.linspace(0.,2,4))
ax2.set_yticklabels(["D", "D","C","C"])
ax2.tick_params(colors=colors[1], which='both' , axis = "y") 
ax2.set_ylim(ax[2].get_ylim())
ax[2].set_xlabel("t")
plt.tight_layout()
# fig.savefig(f"./figures/PD_N_{N}_p1_{p_coops[0]}_p2_{p_coops[1]}_phase_{phase}_T_{T}_step_{step}.pdf")
plt.show()


######## 
#  Plot figure 2   
########
fig,ax = plt.subplots(ncols  = 2, figsize= (40,16), gridspec_kw={'width_ratios': [1, 2]})
time =np.linspace(0,10*np.pi,T)
ax[1].plot(time,agents[0].p_coop,color=colors[0])
ax[1].plot(time,agents[1].p_coop,color=colors[1])
colors2 =   ["r", "orange", "purple", "green"]

if phase == 2 and p_coops[0] ==0.5:
    
    idx_mid = np.where(abs(agents[0].p_coop - agents[1].p_coop) <0.03)[0]
    idx_mid= idx_mid[np.where(~(np.diff(idx_mid[:-1])<3))] #get rid of multples
    idx_max0 = np.where(((max(agents[0].p_coop ) - agents[0].p_coop) < 0.01)) #where agent0 prob maxes out
    idx_max1 = np.where((max(agents[1].p_coop ) - agents[1].p_coop) < 0.01)#where agent1 prob maxes out
    
    ax[0].errorbar( 3, np.mean(np.array(agents[0].payoff)[idx_mid] + np.array(agents[1].payoff)[idx_mid]),
        np.std(np.array(agents[0].payoff)[idx_mid] + np.array(agents[1].payoff)[idx_mid]),capsize=10,c=colors2[0],markersize=50,marker='s',markerfacecolor= "none")

    ax[0].set_xticks(np.linspace(1,3,3))
    ax[1].scatter(time[idx_mid],agents[0].p_coop[idx_mid], edgecolors=colors2[0],s=300, facecolors="none",marker="^")
    ax[1].scatter(time[idx_mid],agents[1].p_coop[idx_mid], edgecolors=colors2[0],s=300, facecolors="none",marker="^")



elif phase == 0 and p_coops[0] ==0.5:
    idx_mid = np.where(abs(agents[0].p_coop - agents[1].p_coop) <0.01)[0]
    idx_max0 = np.where(((max(agents[0].p_coop ) - agents[0].p_coop) < 0.01)) #where agent0 prob maxes out
    idx_max1 = np.where(abs(min(agents[1].p_coop ) - agents[1].p_coop) < 0.01)#where agent1 prob maxes out
    
    ax[0].errorbar( 3, np.mean(np.array(agents[0].payoff)[idx_mid] + np.array(agents[1].payoff)[idx_mid]),
        np.std(np.array(agents[0].payoff)[idx_mid] + np.array(agents[1].payoff)[idx_mid]),capsize=10,c=colors2[0],markersize=50,marker='s',markerfacecolor= "none")

    ax[0].set_xticks(np.linspace(1,3,3))
    ax[1].scatter(time[idx_mid],agents[0].p_coop[idx_mid], edgecolors=colors2[0],s=300, facecolors="none",marker="^")
    ax[1].scatter(time[idx_mid],agents[1].p_coop[idx_mid], edgecolors=colors2[0],s=300, facecolors="none",marker="^")


elif phase == 0 and p_coops[0] ==0 :# or p_coops[1]==0):
    print("iim here1" )
    idx_max0 = np.where(((max(agents[0].p_coop ) - agents[0].p_coop) < 0.01)) #where agent0 prob maxes out
    idx_max1 = np.where((abs(min(agents[1].p_coop ) - agents[1].p_coop)) < 0.01)
    ax[0].set_xlim(0,2.5)#ticks(np.linspace(1,2,2))

elif phase == 2 and p_coops[0] ==0 :# or p_coops[1]==0):
    print("iim here")
    idx_max0 = np.where(((max(agents[0].p_coop ) - agents[0].p_coop) < 0.01)) #where agent0 prob maxes out
    idx_max1 = np.where(((max(agents[1].p_coop ) - agents[1].p_coop) < 0.01))
    m1 = int(max(np.diff(idx_max1)[0]))
    idx_max1 =idx_max1[0][int(m1/2):][::m1]
    ax[0].set_xlim(0,2.5)#ticks(np.linspace(1,2,2))

ax[1].scatter(time[idx_max0],agents[1].p_coop[idx_max0], edgecolors=colors2[1],s=300, facecolors="none",marker="o")
ax[1].scatter(time[idx_max0],agents[0].p_coop[idx_max0], edgecolors=colors2[1],s=300, facecolors="none", marker="o")
ax[1].scatter(time[idx_max1],agents[1].p_coop[idx_max1], edgecolors=colors2[2],s=300, facecolors="none",marker="s")
ax[1].scatter(time[idx_max1],agents[0].p_coop[idx_max1], edgecolors=colors2[2],s=300, facecolors="none", marker="s")


ax[0].errorbar( 1, np.mean(np.array(agents[0].payoff)[idx_max0] + np.array(agents[1].payoff)[idx_max0]),
    np.std(np.array(agents[0].payoff)[idx_max0] + np.array(agents[1].payoff)[idx_max0]),capsize=10,c=colors2[1],markersize=50,marker='^',markerfacecolor= "none")

ax[0].errorbar( 2, np.mean(np.array(agents[0].payoff)[idx_max1] + np.array(agents[1].payoff)[idx_max1]),
    np.std(np.array(agents[0].payoff)[idx_max1] + np.array(agents[1].payoff)[idx_max1]),capsize=10,c=colors2[2],markersize=50,marker='o',markerfacecolor= "none")

ax[0].set_ylabel("average payoff")
ax[1].set_ylabel("probability of cooperating")
ax[1].set_xlabel("t")


ax[0].set_xticklabels([])

fig.savefig(f"./figures/PD_avgpayoff_N_{N}_p1_{p_coops[0]}_p2_{p_coops[1]}_phase_{phase}_T_{T}_step_{step}.pdf")
plt.show()


######## 
#  PD on a network    
########
    
T = 100
step = 1
n=10
p_coops = np.random.rand(n)
phase = 0
agents = [Agent(idx=i,p_coop= p_coops[i],T= T, modulate= True,timescale= np.random.random(),phase= 2*np.pi*np.random.random())  for i in range(n)]


G = nx.erdos_renyi_graph(n,0.5) #creates a random graph


for _ in range(0,T,step):
    for i in range(n):
        agents[i].decision(_) 
    for i in range(n):
        agents[i].calc_payoff(G, agents) 


for i in range(n):
    agents[i].print_stats(ax[0]) 


fig,ax = plt.subplots(ncols  = 3, figsize= (60,16))
ax[0].plot(sum(np.array(agents[i].payoff) for i in range(n)))
ax[1].plot(sum(agents[i].p_coop for i in range(n)))
# ax[1].plot(agents[0].p_coop+agents[1].p_coop,label="summed")
ax[1].scatter(range(0,T,step),agents[0].p_coop[::step])
ax[1].scatter(range(0,T,step),agents[1].p_coop[::step])
# ax[1].plot(agents[1].p_coop)
ax[0].legend()
ax[0].set_ylabel("payoff at t")
ax[1].set_ylabel("probability of cooperating")
ax[0].set_xlabel("t")
ax[1].set_xlabel("t")
# ax[2].scatter(range(0,T,step),agents[0].history)
# ax[2].scatter(range(0,T,step),agents[1].history)
ax[2].scatter( np.array(agents[0].payoff) + np.array(agents[1].payoff), (agents[0].p_coop+agents[1].p_coop)[::step])
ax[2].set_xlabel("Agent 1 cooperates + Agent 2 cooperates")
ax[2].set_ylabel("sum of payoff")

plt.show()


