#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 12:40:08 2022

@author: vvasiliau
"""

from prisoners_dilemma import *
from tqdm import tqdm

if __name__=="__main__":
    
    T = 1000 #number of timesteps
    step = 1    #to plot every step'th step 
    p_coops = [0.5,0.54]    #probabilities to ccooperate
    t_obs = np.linspace(0,20*np.pi,T)
    evolve= False
    
    nsim = 1000
    res = []
    total_payoff=[]
    for niter in tqdm(range(nsim)):
        
        phase = 2  #phase shift. If one agen'ts pahse =0, and anothers = 2, they are out-of-phase, becauuse the period is 4
        agents = [Agent(idx=0,p_coop= p_coops[0],T= T, modulate= True,timescale= 0.25,phase= 0,t_obs=t_obs,amplitude=0.5) ,
                  Agent(idx =1,p_coop= p_coops[1],T= T, modulate=True,timescale = 0.01,phase =phase,t_obs=t_obs,amplitude=0.5)]#, amplitude=0.2)]
        # agents = [Agent(idx=0,p_coop= p_coops[0],T= T, modulate= True,timescale= 0.25,phase= 0,t_obs=t_obs) ,
        #           Agent(idx =1,p_coop= p_coops[1],T= T, modulate=True,timescale = 0.25,phase =phase,t_obs=t_obs)]#, amplitude=0.2)]
        
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
            # print(_)
            action1, action2 = agents[0].decision(_),agents[1].decision(_) 
            for a in agents:
                a.calc_payoff(G, agents)
            if evolve:
                for a in agents:
                    a.evolution(_,1, agents)
        total_payoff.append(sum(agents[0].payoff)  + sum(agents[1].payoff))
        
        width=10
        data = np.array(agents[0].history)
        result = data[:(data.size // width) * width].reshape(-1, width).mean(axis=1)
        data = np.array(agents[1].history)
        result2= data[:(data.size // width) * width].reshape(-1, width).mean(axis=1)
        
        res.append(result+result2)
        
    # plt.rcParams.update({'font.size': 30})
    # fig,ax = plt.subplots(figsize=(10,8))
    # ax.plot(t_obs,agents[1].p_coop+agents[0].p_coop,label="$p_1^{(t)}+p_2^{(t)}$",color=colors[0])
    # ax.plot(t_obs[::width], np.array(res).mean(0),label="mutual strategy",color=colors[1])
    # ax.legend(loc=1)
    # ax.set_xlabel("$t$")
    # plt.tight_layout()
    # plt.savefig("p_coop_resulted_coop.pdf")
    # fig.show()

    # ######## 
    # #  Does variability have an effect?
    # ########
    # fig,ax = plt.subplots(ncols  = 3, figsize= (47,13))
    # # agents[0].print_stats(ax[0],col=colors[0])
    # # agents[1].print_stats(ax[0],col=colors[0])
    # ax[0].plot(t_obs,agents[0].payoff,color = colors[0],label= "p={:.1f}".format(agents[0].p_coop_const))
    # ax[0].plot(t_obs,agents[1].payoff,color = colors[1], label= "p={:.1f}".format(agents[1].p_coop_const))
    # ax[1].plot(t_obs,agents[0].p_coop, alpha=0.1, color= colors[0])
    # ax[1].plot(t_obs,agents[1].p_coop,alpha =0.1, color= colors[1])
    # # ax[1].plot(agents[0].p_coop+agents[1].p_coop,label="summed")
    # ax[1].scatter(t_obs,agents[0].p_coop[::step],edgecolors = colors[0],marker="^", facecolors="none",s=250)
    # ax[1].scatter(t_obs,agents[1].p_coop[::step],edgecolors=colors[1],marker="o",facecolors="none",s=250)
    # # ax[1].plot(agents[1].p_coop)
    # # ax[0].legend(loc =1)
    # ax[0].set_ylabel("payoff")
    # ax[1].set_ylabel("probability of cooperating")
    # ax[0].set_xlabel("t")
    # ax[1].set_xlabel("t")
    # ax[2].scatter(t_obs, np.array(agents[0].history)*0.6667 + np.array(agents[1].history)*1.3333, edgecolors="k",s= 250,marker="o",facecolors="none")#, (agents[0].p_coop+agents[1].p_coop)[::step])
    
    # ax[0].set_yticks(range(0,6,1))
    # ax[1].set_ylim(-0.1,1.1)
    # ax2 = ax[2].twinx()
    # ax[2].set_yticks(np.linspace(0,2,4))
    # ax[2].set_yticklabels(["D", "C","D","C"])
    # ax[2].tick_params(colors=colors[0], which='both' , axis = "y") 
    # ax[2].set_ylabel("strategy")
    # ax2.set_yticks(np.linspace(0.,2,4))
    # ax2.set_yticklabels(["D", "D","C","C"])
    # ax2.tick_params(colors=colors[1], which='both' , axis = "y") 
    # ax2.set_ylim(ax[2].get_ylim())
    # ax[2].set_xlabel("t")
    # plt.tight_layout()
    # # fig.savefig(f"./figures/PD_N_{N}_p1_{p_coops[0]}_p2_{p_coops[1]}_phase_{phase}_T_{T}_step_{step}.pdf")
    # plt.show()
    