#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 16:38:55 2022

bsub -W "24:00" -n 4 -R "rusage[mem=10000]" python two_player_synchrony.py
module load openjdk/17.0.0_35
module load python/3.7.4

python 
import os
# for K_f in [0.01, 0.1,1 ]:
K_f = 0.1
for Kval in range(0,11,1):
    Kval = Kval/100
    for tau in range(1,11,1):
        os.system(f'bsub -W "4:00" -n 4 -R "rusage[mem=10000]" "python two_player_synchrony.py {K_f} {Kval} {tau}"')

    
@author: vvasiliau
"""

from prisoners_dilemma import *
from scipy.integrate import odeint
import numpy as np
import sys
from matplotlib.legend_handler import HandlerTuple
import pandas as pd
    
    
if __name__ =="__main__":
    
    plt.rcParams.update({'font.size': 50})
    T=40
    colors = {0: "teal",  1 : "orange"} #colors associated with agent 0 and agent  1  

    fig_no = "2b"
    
    p0=np.array([0,1])
    ##########
    #  Plot figure 2a
    # ########
    if fig_no =="2a":
        plt.rcParams.update({'font.size': 70})
        t_obs = np.linspace(0,T,T*10+1 )
        inphase=False
        K = np.array([0., 0.0])
        agents= run_n_player_game(K=K, p0 = p0, inphase = inphase,evolution = False,amplitude=0.2, T =T,  tau=None,gdensity = 1,timescale = 5)
        fig,ax = plt.subplots(ncols  = 3, figsize= (38,13))
        ax[0].plot(t_obs,agents[0].payoff,color = colors[0])#,label= "p={:.1f}".format(p0[0]))
        ax[0].plot(t_obs,agents[1].payoff,color = colors[1])#, label= "p={:.1f}".format(p0[1]))
        ax[1].plot(t_obs,agents[0].p_coop, alpha=0.1, color= colors[0])
        ax[1].plot(t_obs,agents[1].p_coop,alpha =0.1, color= colors[1])
        ax[1].scatter(t_obs,agents[0].p_coop[::],edgecolors = colors[0],marker="^", facecolors="none",s=250)
        ax[1].scatter(t_obs,agents[1].p_coop[::],edgecolors=colors[1],marker="o",facecolors="none",s=250)
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
        fig.savefig(f"./figures/fig1/PD_N_2_p1_{p0[0]}_p2_{p0[1]}_inphase_{inphase}_T_{T}_K0_{K[0]}_K1_{K[1]}.pdf")
        plt.show()
        
    
    
    ######## 
    #  Plot figure 2
    ########
    if fig_no =="2b":
        
        fig,ax = plt.subplots(figsize=(12,10))
        for Kval in np.linspace(0,0.1,10):
            res = []
            K = np.array([Kval, Kval])
            for i in range(100):
                agents= run_n_player_game(K=K, p0 = p0, inphase = True,evolution = False,amplitude=0.2, T =T, tau=None,gdensity = 1,timescale = 5)
                h1, h2 = np.array(agents[0].history)[:-100,1], np.array(agents[1].history)[:-100,0]
                res.append(sum(h1*h2)/len(h1))
            
            res2 = []
            for i in range(100):
                agents= run_n_player_game(K=K, p0 = p0, inphase = False,evolution = False,amplitude=0.2, T =T, tau=None,gdensity = 1, timescale = 5)
                h1, h2 = np.array(agents[0].history)[:-100,1], np.array(agents[1].history)[:-100,0]
                res2.append(sum(h1*h2)/len(h1))
                        
            ax.errorbar(x= Kval,y=  np.mean(res), yerr= np.std(res),color = "red",capsize= 5,markersize = 20,marker='o')
            ax.errorbar(x= Kval,y=  np.mean(res2), yerr= np.std(res2),color = "blue",capsize= 5,markersize = 20,marker='^')
    
        ax.errorbar(x= Kval,y=  np.mean(res), yerr= np.std(res),color = "red",capsize= 5,markersize = 20,marker='o',label = "$0$")
        ax.errorbar(x= Kval,y=  np.mean(res2), yerr= np.std(res2),color = "blue",capsize= 5,markersize = 20,marker='^', label = "$T/2$")
        ax.set_ylabel("$f_C(K)$")
        ax.set_xlabel("$K$")
        ax.legend(loc=4)
        ax.get_legend().set_title("$|\\theta_1−\\theta_2|$")
        plt.tight_layout()
        plt.savefig("figures/fig2/PD_2_player_fraction_coop_vs_K.pdf")
        plt.show()
        
    
    ######## 
    #  Plot figure 3
    ########
    
    
    if fig_no =="3":
        fig,ax = plt.subplots(figsize=(10,12))
        p0=np.array([0,1])
        p1 = 1
        K_min = 0
        K_max =0.1
        K_1 = K_max/2
        a = 0.5
        niter =100
        for K_2 in np.linspace(0,K_max,10):
            K = np.array([K_1,K_2])
            res = []
            resp = []
            for i in range(niter):
                agents= run_n_player_game(K=K, p0 = p0, inphase = False,evolution = False,amplitude=a,  T =T,  tau=None,gdensity = 1,timescale=5)
                h1, h2 = np.array(agents[0].history)[:-100,1], np.array(agents[1].history)[:-100,0]
                res.append(sum(h1*h2)/len(h1))
                resp.append(((agents[0].p_coop +agents[1].p_coop)/2)[:-100].mean())
             
            f1= ax.errorbar(x= K_2/K_1,y=  np.mean(res), yerr= np.std(res),alpha = p1,capsize= 10,markersize = 20,marker='o', color='salmon')
            f2= ax.errorbar(x= K_2/K_1,y=  np.mean(resp), yerr= np.std(resp), alpha = p1,capsize= 10,markersize = 20,marker='^',color='cornflowerblue')
            

        
        for K_2 in np.linspace(0,K_max,10):
            K = np.array([K_1,K_2])
            res = []
            resp = []
            for i in range(niter):
                agents= run_n_player_game(K=K, p0 = p0, inphase = True,evolution = False,amplitude=a,  T =T, tau=None,gdensity = 1, timescale =5)
                h1, h2 = np.array(agents[0].history)[:-100,1], np.array(agents[1].history)[:-100,0]
                res.append(sum(h1*h2)/len(h1))
                resp.append(((agents[0].p_coop +agents[1].p_coop)/2)[:-100].mean())
            
            f3 = ax.errorbar(x= K_2/K_1,y=  np.mean(res), yerr= np.std(res),alpha = p1,capsize= 10,markersize = 20,marker='o', color='red')
            f4 = ax.errorbar(x= K_2/K_1,y=  np.mean(resp), yerr= np.std(resp), alpha = p1,capsize= 10,markersize = 20,marker='^',color='blue')
                
        ax.set_xlabel("$K_2/K_1$")
        ax.set_ylim(0,0.8)
        plt.tight_layout()
        plt.savefig(f"figures/fig3/PD_2_player_fraction_coop_vs_K_ratio_amplitude_{a}.pdf")
        plt.show()
        
        
    
    ######## 
    #  Plot figure 4
    ########
        
        
    if fig_no =="4":
        res = []
        T = 20
        K0 = np.array([0.,0.])
        K_f, Kval, tau =sys.argv[1:]
        K_f, Kval, tau= float(K_f), float(Kval), int(tau)
        print(Kval)
        K = np.array([Kval,Kval])
        fc = 0
        fc1 = 0
        for niter in range(2000):
            p0 = np.random.rand(2)#np.array([0,1])
            agents= run_n_player_game(K=K, p0 = p0, inphase = True,evolution = True,amplitude=0., T=T, tau=tau,gdensity = 1,K_f=K_f,timescale =5)
            h1, h2 = np.array(agents[0].history)[:-50,1], np.array(agents[1].history)[:-50,0]
            fc +=sum(h1*h2)/len(h1)
            agents= run_n_player_game(K=K0, p0 = p0, inphase = True,evolution = True,amplitude=0.,tau=tau,gdensity = 1, T =T, K_f=K_f,timescale =5)
            h1, h2 = np.array(agents[0].history)[:-50,1], np.array(agents[1].history)[:-50,0]
            fc1 +=sum(h1*h2)/len(h1)
        fc1 = fc1/2000
        fc = fc/2000
        res.append((Kval, tau, fc,fc1))
                    
        df = pd.DataFrame(res, columns = ["K","tau","fc","fc0"])
        df["res"] = df.fc/df.fc0
        df.to_csv(f"figures/fig4/2_player_p_vs_K_tau_Kf_{K_f}_2.csv", header = None,mode = "a+")
        
        # plotting
        import seaborn as sns
        import matplotlib.colors as colors
        # for K_f in [0.1]:#[0.01,0.1,1.0]:
        df = pd.read_csv(f"./figures/fig4/2_player_p_vs_K_tau_Kf_{K_f}_2.csv",header = None)
        df.columns =  ["i", "K","tau","fc","fc0","res"]
        fig,ax = plt.subplots(figsize=(12,10))
        table = df.pivot("K","tau","res")
        print(min(df.res),max(df.res))
        divnorm = colors.TwoSlopeNorm(vmin=0.8, vcenter=1, vmax=1.1)
        ax = sns.heatmap(table,norm =divnorm,cmap='BrBG')#norm=divnorm
        # ax.set_yticks(yticks) 
        ax.invert_yaxis()
        ax.set_title("$f_C(K)/f_C(K=0)$")
        ax.set_xlabel("$\\tau$")
        ax.set_yticks(np.array(range(len(list(set(df.K)))))+0.5)
        ax.set_yticklabels(np.array(range(0,11,1))/100,rotation=0)
        ax.set_xticks(np.array(range(len(list(set(df.tau)))))+0.5)
        ax.set_xticklabels(np.array(list(set(df.tau))),rotation=0)
        plt.tight_layout()
        plt.savefig(f"figures/fig4/2_player_p_vs_K_tau_Kf_{K_f}.pdf")
        plt.show()
        
        
        