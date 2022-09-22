#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 14:42:53 2022

@author: vvasiliau
"""
import networkx as nx
G = nx.MultiDiGraph()
G.add_edge("network_structure", "cooperation" )
G.add_edge("network_structure", "cooperation" )
G.add_edge("network_structure", "cooperation" )
G.add_edge("network_structure", "cooperation" )
G["network_structure"]["cooperation"][0]["name"] = "proximity"
G["network_structure"]["cooperation"][1]["name"] = "dynamic"
G["network_structure"]["cooperation"][2]["name"] = "higherorder"
G["network_structure"]["cooperation"][3]["name"] = "centrality"

G.add_edge("higherorder_games", "nondominant_strategists")
G.add_edge( "nondominant_strategists", "cooperation")
G.add_edge("higherorder_games", "strategic_diversity")
G.add_edge("strategic_diversity", "cooperation")


G.add_edge("norms_enforcement", "cooperation")
G.add_edge("willingness_punish", "cooperation")
G.add_edge("oxytocin","norms_enforcement")
G.add_edge("oxytocin","willingness_punish")
G.add_edge("neural_activity","oxytocin")


G.add_edge("circadian_clock","endocryne_system")
G.add_edge("endocryne_system", "neural_activity")
 

G.add_edge("circadian_clock","network_structure")

G.add_edge("exclusion","mentalizing_network")
G.add_edge("mentalizing_network","network_structure")
G["mentalizing_network"]["network_structure"][0]["name"] = "egocentric"

G.add_edge("neural_activity","network_structure")

# G["neural_activity"]["network_structure"][0]["name"] = "centrality"

G.add_edge("cognitive_abilities","network_structure")
G["cognitive_abilities"]["network_structure"][0]["name"]  = "memory"

nx.set_node_attributes(G,{a:a for a in list(G.nodes())},"name")
nx.write_graphml(G,"DAG.graphml")

# ("network_structure", "cooperation" ): "proximity"
# ("network_structure", "cooperation" ): "dynamic"

# ("higherorder_games", "nondominant_strategists" ,"cooperation"): 1
# ("network_structure", "cooperation" ): "higherorder" #the more triplets there are, the higher the cooperation 
# ("higherorder_games", "strategic_diversity","cooperation"  ): 1

# ("oxytocin", "norms_enforcement","cooperation"): 1 + ("network_structure", "cooperation"): "centrality"
# ("oxytocin", "willingness_punish","cooperation"): 1   + ("network_structure", "cooperation"): "centrality"

# ("circadian_clock", "network_structure") : 1


# ("exclusion", "mentalizing_network", "network_structure") : "egocentric"
# ("network_structure", "mentalizing_network", "exclusion") : "egocentric"

# ("neural_activity","centrality") : 1
# ("centrality","neural_activity") : "assortativity" #in other words - diversity



# 
# ("cognitive_abilities","network_structure") : "memory"