#!/bin/bash
# variable params
mux_num_addr_bits=4
stree_max_depth=5
stree_theta_build=250

# static / calced params
declare -A xcs_pop_sizes=( [2]=400 [3]=800 [4]=2400 [5]=9600 )
xcs_pop_size="${xcs_pop_sizes[$mux_num_addr_bits]}"
xcs_beta=0.2
xcs_alpha=0.1
xcs_epsilon_nought=0.1
xcs_nu=5
xcs_theta_ga=25
xcs_chi=0.8
xcs_tau=0.5
xcs_upsilon=0.5
xcs_mu=0.05
xcs_theta_del=25
xcs_delta=0.1
xcs_theta_sub=25
declare -A xcs_p_hashes=( [2]=0.33 [3]=0.33 [4]=0.33 [5]=0.5 )
xcs_p_hash="${xcs_p_hashes[$mux_num_addr_bits]}"
xcs_pred_i=0.01
xcs_epsilon_i=0.01
xcs_fitness_i=0.01
xcs_p_exp=0.5
xcs_theta_fm=25

monitor_steps_per_tick=100
declare -A monitor_num_tickss=( [2]=500 [3]=500 [4]=1000 [5]=50000 )
monitor_num_ticks="${monitor_num_tickss[$mux_num_addr_bits]}"

for xcs_seed in {420..420}; do
   sbatch xcs_fm_mux.sh \
        "$mux_num_addr_bits" \
        "$xcs_seed" \
        "$xcs_pop_size" \
        "$xcs_beta" \
        "$xcs_alpha" \
        "$xcs_epsilon_nought" \
        "$xcs_nu" \
        "$xcs_theta_ga" \
        "$xcs_chi" \
        "$xcs_tau" \
        "$xcs_upsilon" \
        "$xcs_mu" \
        "$xcs_theta_del" \
        "$xcs_delta" \
        "$xcs_theta_sub" \
        "$xcs_p_hash" \
        "$xcs_pred_i" \
        "$xcs_epsilon_i" \
        "$xcs_fitness_i" \
        "$xcs_p_exp" \
        "$xcs_theta_fm" \
        "$stree_max_depth" \
        "$stree_theta_build" \
        "$monitor_steps_per_tick" \
        "$monitor_num_ticks"
done
