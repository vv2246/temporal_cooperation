#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 16:38:55 2022

@author: vvasiliau
"""

from prisoners_dilemma import *
from scipy.integrate import odeint
import numpy as np

 
# def run_2_player_game(K, p1, inphase = True,evolution = False):
     
#     T = 400 #number of timesteps
#     step = 1    #to plot every step'th step 
#     t_obs = np.linspace(0,80*np.pi,T)
#     p0 = np.array([0.0,p1] )
        
#     G = nx.Graph()
#     N=2
#     edges = [(0,1)]
#     G.add_nodes_from(range(N))
#     for (i,j) in edges:
#         G.add_edge(i,j)
#     A = np.array(nx.adjacency_matrix(G).todense())
#     L = -A + np.diag(A.sum(0)) 
#     timescale = 1
#     # K = 0.004
#     B= 1# - K*20
#     if inphase:
        
#         phase= np.array([0,0])
#     else:
#         phase= np.array([0,np.pi])
#     sol  = get_p_coop(p0,timescale,phase,amplitude = 0.5,t_obs= t_obs,A=    A, K = K, B = B)
                
#     # ===============
#     # PD for a pair of agents
#     # ===============
#     agents = [Agent(idx=0,p_coops=sol,T= T,G = G,t_obs=t_obs, K_f = 0) ,
#               Agent(idx =1,p_coops= sol,T= T, G = G,t_obs=t_obs, K_f = 0)]#, amplitude=0.2)]
    
    
#     ######## 
#     #  Game   
#     ########
#     for _ in range(0,T,step):
#         action1, action2 = agents[0].decision(_),agents[1].decision(_) 
#         for a in agents:
#             a.calc_payoff( agents)
#         break
#     return agents





def run_K_B(K,B,):
    T = 1000 #number of timesteps
    step = 1    #to plot every step'th step 
    t_obs = np.linspace(0,80*np.pi,T)
    p0 = np.array([0.0,1] )
        
    G = nx.Graph()
    N=2
    edges = [(0,1)]
    G.add_nodes_from(range(N))
    for (i,j) in edges:
        G.add_edge(i,j)
    A = np.array(nx.adjacency_matrix(G).todense())
    L = -A + np.diag(A.sum(0)) 
    timescale = 1
    
    phase= np.array([0,0])
    sol  = get_p_coop(p0,timescale,phase,amplitude = 0.2,t_obs= t_obs,A=    A, K = K, B = B)
    return sol

    
    
    
    
if __name__ =="__main__":
    
        
    
    plt.rcParams.update({'font.size': 50})
    T=400
    step = 1    #to plot every step'th step 
    # t_obs = np.linspace(0,80*np.pi,T)
    colors = {0: "teal",  1 : "orange"} #colors associated with agent 0 and agent  1
    

    fig_no = "4"
    
    ##########
    #  Plot figure 2a
    # ########
    if fig_no =="2a":
        plt.rcParams.update({'font.size': 70})
        t_obs = np.linspace(0,80*np.pi,T)
        inphase=True
        p0=np.array([0,1])
        K = np.array([0.01, 0.01])
        agents= run_n_player_game(K=K, p0 = p0, inphase = inphase,evolution = False,amplitude=0.2, T =400, tau=None,gdensity = 1)
        fig,ax = plt.subplots(ncols  = 3, figsize= (38,13))
        ax[0].plot(t_obs,agents[0].payoff,color = colors[0])#,label= "p={:.1f}".format(p0[0]))
        ax[0].plot(t_obs,agents[1].payoff,color = colors[1])#, label= "p={:.1f}".format(p0[1]))
        ax[1].plot(t_obs,agents[0].p_coop, alpha=0.1, color= colors[0])
        ax[1].plot(t_obs,agents[1].p_coop,alpha =0.1, color= colors[1])
        ax[1].scatter(t_obs,agents[0].p_coop[::step],edgecolors = colors[0],marker="^", facecolors="none",s=250)
        ax[1].scatter(t_obs,agents[1].p_coop[::step],edgecolors=colors[1],marker="o",facecolors="none",s=250)
        # ax[0].set_ylabel("payoff")
        # ax[1].set_ylabel("probability of cooperating")
        ax[0].set_xlabel("t")
        ax[1].set_xlabel("t")
        ax[2].scatter(t_obs, np.nansum(np.array(agents[0].history),1)*0.6667 + np.nansum(np.array(agents[1].history),1)*1.3333, edgecolors="k",s= 250,marker="o",facecolors="none")#, (agents[0].p_coop+agents[1].p_coop)[::step])
        
        ax[0].set_yticks(range(0,6,1))
        ax[1].set_ylim(-0.1,1.1)
        ax2 = ax[2].twinx()
        ax[2].set_yticks(np.linspace(0,2,4))
        ax[2].set_yticklabels(["D", "C","D","C"])
        ax[2].tick_params(colors=colors[0], which='both' , axis = "y") 
        # ax[2].set_ylabel("strategy")
        ax[0].set_title("payoff")
        ax[1].set_title("$p(t)$")
        ax[2].set_title("strategy")
        ax2.set_yticks(np.linspace(0.,2,4))
        ax2.set_yticklabels(["D", "D","C","C"])
        ax2.tick_params(colors=colors[1], which='both' , axis = "y") 
        ax2.set_ylim(ax[2].get_ylim())
        ax[2].set_xlabel("t")
        plt.tight_layout()
        fig.savefig(f"./figures/PD_N_2_p1_{p0[0]}_p2_{p0[1]}_inphase_{inphase}_T_{T}_K0_{K[0]}_K1_{K[1]}.pdf")
        plt.show()
        
    
    
    ######## 
    #  Plot figure 2b
    ########
    if fig_no =="2b":
        
        fig,ax = plt.subplots(figsize=(10,13))
        for Kval in np.linspace(0,0.015,10):
            res = []
            K = np.array([Kval, Kval])
            for i in range(100):
                agents= run_n_player_game(K=K, p0 = p0, inphase = True,evolution = False,amplitude=0.2, T =400, tau=None,gdensity = 1)
                h1, h2 = np.array(agents[0].history)[:-100,1], np.array(agents[1].history)[:-100,0]
                res.append(sum(h1*h2)/len(h1))
            
            res2 = []
            for i in range(100):
                agents= run_n_player_game(K=K, p0 = p0, inphase = False,evolution = False,amplitude=0.2, T =400, tau=None,gdensity = 1)
                h1, h2 = np.array(agents[0].history)[:-100,1], np.array(agents[1].history)[:-100,0]
                res2.append(sum(h1*h2)/len(h1))
        #     print(np.mean(np.array([agents[0].p_coop[:-100],agents[1].p_coop[:-100]])),end= ",")
            res1 = []
            for i in range(100):
                agents= run_n_player_game(K=np.array([0, 0]), p0 = p0, inphase = True,evolution = False,amplitude=0.2, T =400, tau=None,gdensity = 1)
                h1, h2 = np.array(agents[0].history)[:-100,1], np.array(agents[1].history)[:-100,0]
                res1.append(sum(h1*h2)/len(h1))
        #     print(np.mean(np.array([agents[0].p_coop[:-100],agents[1].p_coop[:-100]])),end= "\n")
            
            ax.errorbar(x= Kval,y=  np.mean(res), yerr= np.std(res),color = "k",capsize= 5,markersize = 15,marker='o')
            ax.errorbar(x= Kval,y=  np.mean(res1), yerr= np.std(res1),color = "r",capsize= 5,markersize = 15,marker='^')
            ax.errorbar(x= Kval,y=  np.mean(res2), yerr= np.std(res2),color = "g",capsize= 5,markersize = 15,marker='*')
    
        ax.errorbar(x= Kval,y=  np.mean(res), yerr= np.std(res),color = "k",capsize= 5,markersize = 15,marker='o',label = "$|\\theta_1−\\theta_2 |=0$")
        ax.errorbar(x= Kval,y=  np.mean(res2), yerr= np.std(res2),color = "g",capsize= 5,markersize = 15,marker='*', label = "$|\\theta_1−\\theta_2 |=𝑇/2$")
        ax.errorbar(x= Kval,y=  np.mean(res1), yerr= np.std(res1),color = "r",capsize= 5,markersize = 15,marker='^', label = "$K=0$")
        ax.set_ylabel("$f_C(K)$")
        ax.set_xlabel("$K$")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
              fancybox=True, shadow=False, ncol=1)
        plt.tight_layout()
        plt.savefig("figures/PD_2_player_fraction_coop_vs_K.pdf")
        plt.show()
        
    
    ######## 
    #  Plot figure 3
    ########
    
    
    if fig_no =="3":
        fig,ax = plt.subplots(figsize=(10,12))
        # for p1 in np.linspace(0,1,10):
        p0=np.array([0,1])
        p1 = 1
        K_min = 0
        K_max =0.015
        K_1 = K_max/2
        a = 0.2
        niter =10
        for K_2 in np.linspace(0,0.015,10):
            K = np.array([K_1,K_2])
            res = []
            resp = []
            for i in range(niter):
                agents= run_n_player_game(K=K, p0 = p0, inphase = False,evolution = False,amplitude=a, T =400, tau=None,gdensity = 1)
                h1, h2 = np.array(agents[0].history)[:-100,1], np.array(agents[1].history)[:-100,0]
                res.append(sum(h1*h2)/len(h1))
                resp.append(((agents[0].p_coop +agents[1].p_coop)/2)[:-100].mean())
            res1 = []
            resp1 = []
            for i in range(niter):
                agents= run_n_player_game(K=0, p0 = p0, inphase = False,evolution = False,amplitude=a, T =400, tau=None,gdensity = 1)
                h1, h2 = np.array(agents[0].history)[:-100,1], np.array(agents[1].history)[:-100,0]
                res1.append(sum(h1*h2)/len(h1))
                resp1.append(((agents[0].p_coop +agents[1].p_coop)/2)[:-100].mean())
            
            ax.errorbar(x= K_2/K_1,y=  np.mean(res), yerr= np.std(res),alpha = p1,capsize= 10,markersize = 20,marker='o', color='salmon')
            ax.errorbar(x= K_2/K_1,y=  np.mean(res1), yerr= np.std(res1), alpha = p1,capsize= 10,markersize = 20,marker='o',mfc='w', color='salmon')
            ax.errorbar(x= K_2/K_1,y=  np.mean(resp), yerr= np.std(resp), alpha = p1,capsize= 10,markersize = 20,marker='^',color='cornflowerblue')
            ax.errorbar(x= K_2/K_1,y=  np.mean(resp1), yerr= np.std(resp1), alpha = p1,capsize= 10,markersize = 20,marker='^',mfc='w', color='cornflowerblue')


        a = 0.2
        
        for K_2 in np.linspace(0,0.015,10):
            K = np.array([K_1,K_2])
            res = []
            resp = []
            for i in range(niter):
                agents= run_n_player_game(K=K, p0 = p0, inphase = False,evolution = False,amplitude=a, T =400, tau=None,gdensity = 1)
                h1, h2 = np.array(agents[0].history)[:-100,1], np.array(agents[1].history)[:-100,0]
                res.append(sum(h1*h2)/len(h1))
                resp.append(((agents[0].p_coop +agents[1].p_coop)/2)[:-100].mean())
            res1 = []
            resp1 = []
            for i in range(niter):
                agents= run_n_player_game(K=0, p0 = p0, inphase = False,evolution = False,amplitude=a, T =400, tau=None,gdensity = 1)
                h1, h2 = np.array(agents[0].history)[:-100,1], np.array(agents[1].history)[:-100,0]
                res1.append(sum(h1*h2)/len(h1))
                resp1.append(((agents[0].p_coop +agents[1].p_coop)/2)[:-100].mean())
            
            ax.errorbar(x= K_2/K_1,y=  np.mean(res), yerr= np.std(res),alpha = p1,capsize= 10,markersize = 20,marker='o', color='salmon')
            ax.errorbar(x= K_2/K_1,y=  np.mean(res1), yerr= np.std(res1), alpha = p1,capsize= 10,markersize = 20,marker='o',mfc='w', color='salmon')
            ax.errorbar(x= K_2/K_1,y=  np.mean(resp), yerr= np.std(resp), alpha = p1,capsize= 10,markersize = 20,marker='^',color='cornflowerblue')
            ax.errorbar(x= K_2/K_1,y=  np.mean(resp1), yerr= np.std(resp1), alpha = p1,capsize= 10,markersize = 20,marker='^',mfc='w', color='cornflowerblue')


        # for K_2 in np.linspace(0,0.015,10):
        #     K = np.array([K_1,K_2])
        #     res = []
        #     resp = []
        #     print(K)
        #     for i in range(niter):
        #         agents= run_n_player_game(K=K, p0 = p0, inphase = True,evolution = False,amplitude=a, T =400, tau=None,gdensity = 1)
            
        #         h1, h2 = np.array(agents[0].history)[:-100,1], np.array(agents[1].history)[:-100,0]
        #         res.append(sum(h1*h2)/len(h1))
        #         resp.append(((agents[0].p_coop +agents[1].p_coop)/2)[:-100].mean())
        #     print(np.mean(np.array([agents[0].p_coop[:-100],agents[1].p_coop[:-100]])),end= ",")
        #     res1 = []
        #     resp1 = []
        #     for i in range(niter):
        #         agents= run_n_player_game(K=0, p0 = p0, inphase = True,evolution = False,amplitude=a, T =400, tau=None,gdensity = 1)
            
        #         h1, h2 = np.array(agents[0].history)[:-100,1], np.array(agents[1].history)[:-100,0]
        #         res1.append(sum(h1*h2)/len(h1))
        #         resp1.append(((agents[0].p_coop +agents[1].p_coop)/2)[:-100].mean())
        #     print(np.mean(np.array([agents[0].p_coop[:-100],agents[1].p_coop[:-100]])),end= "\n")
            
        #     if K_2!=0.015:
        #         ax.errorbar(x= K_2/K_1,y=  np.mean(res), yerr= np.std(res),alpha = p1,capsize= 10,markersize = 20,marker='o', color='r')
        #         ax.errorbar(x= K_2/K_1,y=  np.mean(res1), yerr= np.std(res1), alpha = p1,capsize= 10,markersize = 20,marker='o',mfc='w', color='r')
        #         ax.errorbar(x= K_2/K_1,y=  np.mean(resp), yerr= np.std(resp), alpha = p1,capsize= 10,markersize = 20,marker='^',color='b')
        #         ax.errorbar(x= K_2/K_1,y=  np.mean(resp1), yerr= np.std(resp1), alpha = p1,capsize= 10,markersize = 20,marker='^',mfc='w', color='b')
    
        # ax.errorbar(x= K_2/K_1,y=  np.mean(res), yerr= np.std(res),alpha = p1,capsize= 10,markersize = 20,marker='o', color='r',label = "$f_C,K_1=K_{\\mathrm{max}}/2$")
        # ax.errorbar(x= K_2/K_1,y=  np.mean(res1), yerr= np.std(res1), alpha = p1,capsize= 10,markersize = 20,marker='o',mfc='w', color='r',label = "$f_C,K_1,K_2=0$")
        # ax.errorbar(x= K_2/K_1,y=  np.mean(resp), yerr= np.std(resp), alpha = p1,capsize= 10,markersize = 20,marker='^',color='b',label="$\\langle p\\rangle, K_1,K_2\\neq 0$")
        # ax.errorbar(x= K_2/K_1,y=  np.mean(resp1), yerr= np.std(resp1), alpha = p1,capsize= 10,markersize = 20,marker='^',mfc='w', color='b',label="$\\langle p\\rangle, K_1,K_2= 0$")
        
        
        # ax.errorbar(x= K_2/K_1,y=  np.mean(resp), yerr= np.std(resp), alpha = p1,capsize= 10,markersize = 20,marker='^',color='b',label="$\\langle p\\rangle, K_1,K_2\\neq 0$")
        # ax.errorbar(x= K_2/K_1,y=  np.mean(resp1), yerr= np.std(resp1), alpha = p1,capsize= 10,markersize = 20,marker='^',mfc='w', color='b',label="$\\langle p\\rangle, K_1,K_2= 0$")
        # ax.set_ylabel("$f_C(K)$")
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
        #       fancybox=True, shadow=True, ncol=2)
    
        ax.set_xlabel("$K_2/K_1$")
        # plt.legend(loc= 1, fontsize= 30)
        ax.set_ylim(0,0.8)
        plt.tight_layout()
        # plt.savefig(f"figures/PD_2_player_fraction_coop_vs_K_ratio_amplitude_{a}.pdf")
        plt.show()
        
        
    
    ######## 
    #  Plot figure 4
    ########
        
        
    if fig_no =="4":
        res = []
        # res1 = []
        # K = np.array([0.1,0.1])
        K0 = np.array([0.,0.])
        p0 = np.array([0,1])
        for Kval in range(0,11,1):
            Kval = Kval/100
            K = np.array([Kval,Kval])
            for tau in range(1,11,1):
                fc = 0
                for niter in range(100):
                    
                    agents= run_n_player_game(K=K, p0 = p0, inphase = True,evolution = True,amplitude=0.2, T =400, tau=10,gdensity = 1)
                    h1, h2 = np.array(agents[0].history)[:-100,1], np.array(agents[1].history)[:-100,0]
                    fc +=sum(h1*h2)/len(h1)
                fc = fc/100 
                
                fc1 = 0
                for niter in range(100):
                    agents= run_n_player_game(K=K0, p0 = p0, inphase = True,evolution = True,amplitude=0.2, T =400, tau=10,gdensity = 1)
                    h1, h2 = np.array(agents[0].history)[:-100,1], np.array(agents[1].history)[:-100,0]
                    fc1 +=sum(h1*h2)/len(h1)
                fc1 = fc1/100
                res.append((Kval, tau, fc,fc1))
                
                    
        import pandas as pd
        import seaborn as sns
    # 
        fig,ax = plt.subplots(figsize=(12,10))
        # plt.rcParams.update({'font.size': 30})
        df = pd.DataFrame(res, columns = ["K","tau","fc","fc0"])
        df["res"] = df.fc/df.fc0
        table = df.pivot("K","tau","res")
        yticks = np.linspace(df.K.min(),df.K.max(),8)
        ax = sns.heatmap(table,vmin=0, vmax=4.3)#,cmap='BrBG')#,yticklabels=list(yticks))
        # ax.set_yticks(yticks)
        ax.invert_yaxis()
        ax.set_title("$f_C(K)/f_C(K=0)$")
        ax.set_xlabel("$\\tau$")
        ax.set_yticks(np.array(range(len(list(set(df.K)))))+0.5)
        ax.set_yticklabels(np.array(range(0,11,1))/10,rotation=0)
        ax.set_xticks(np.array(range(len(list(set(df.tau)))))+0.5)
        ax.set_xticklabels(np.array(list(set(df.tau))),rotation=0)
        plt.tight_layout()
        plt.savefig("2_player_p_vs_K_tau.pdf")
        
        # plt.tight_layout()
    ######## 
    #  Plot figure 2c
    ########
    # fig,ax = plt.subplots(figsize=(10,8))
    # res = []
    # res2 =[]
    # for Kval in np.linspace(0,0.5,10):
        
    #     K = np.array([Kval, Kval])
    #     res.append( run_n_player_game(K, p0 , inphase = True,evolution = False,amplitude = 0.1, T= 400, tau = 10, gdensity=1, simulate= False).mean())
    #     res2.append( run_n_player_game(K, p0 , inphase = True,evolution = False,amplitude = 0.1, T= 400, tau = 10, gdensity=1, simulate= False).mean())
            
    #     for B in  np.linspace(0,1,10):
    #         sol = np.array([run_K_B(K,B).mean() for i in range(10)])
    #         res.append((round(K,1),round(B,1), round(sol.mean(),3)))
            
            
    # import pandas as pd
    # import seaborn as sns
    # df = pd.DataFrame(res, columns = ["K","B","res"])
    # table = df.pivot("K","B","res")
    # ax = sns.heatmap(table)
    # ax.invert_yaxis()
    # ax.set_title("$\\langle p_i(t)\\rangle_{it}$")
    # plt.tight_layout()
    # plt.savefig("2_player_p_vs_K_B.pdf")
    
    