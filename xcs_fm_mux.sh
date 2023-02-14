#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1

source ~/virtualenvs/xcssl/bin/activate
#python3 -m cProfile -o "${SLURM_JOB_ID}.prof" xcs_fm_mux.py \
python3 xcs_fm_mux.py \
    --experiment-name="$SLURM_JOB_ID" \
    --mux-num-addr-bits="$1" \
    --xcs-seed="$2" \
    --xcs-pop-size="$3" \
    --xcs-beta="$4" \
    --xcs-alpha="$5" \
    --xcs-epsilon-nought="$6" \
    --xcs-nu="$7" \
    --xcs-theta-ga="$8" \
    --xcs-chi="$9" \
    --xcs-tau="${10}" \
    --xcs-upsilon="${11}" \
    --xcs-mu="${12}" \
    --xcs-theta-del="${13}" \
    --xcs-delta="${14}" \
    --xcs-theta-sub="${15}" \
    --xcs-p-hash="${16}" \
    --xcs-pred-i="${17}" \
    --xcs-epsilon-i="${18}" \
    --xcs-fitness-i="${19}" \
    --xcs-p-exp="${20}" \
    --xcs-theta-fm="${21}" \
    --stree-max-depth="${22}" \
    --stree-theta-build="${23}" \
    --xcs-do-ga-subsumption \
    --monitor-steps-per-tick="${24}" \
    --monitor-num-ticks="${25}"
mv "slurm-${SLURM_JOB_ID}.out" "${SLURM_JOB_ID}/"
#mv "${SLURM_JOB_ID}.prof" "${SLURM_JOB_ID}/"
