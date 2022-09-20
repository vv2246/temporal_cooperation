rm(list=ls())
library(dagitty)
library(ggdag)
library(ggplot2)
library(dplyr)


myDAG <- dagitty( 'dag {
neural_activity->network_position->network
network->network_position->neural_activity

circadian_clock->endocryne_system
endocrine_system->neural_activity->network_structure->network
exclusion->neural_activity->mentalizing_system->network_structure->network
circadian_clock->network_connectivity->network

neural_activity->oxytocin->punishing->cooperation

higherorder_games->strategic_nondominance->cooperation
higherorder_games->strategic_diversity->cooperation

network->network_temporality->cooperation 
inequality->cooperation->network_topology->network

network->network_structure->cooperation
cooperation->network_structure->network
                  }' )
plot( myDAG )
