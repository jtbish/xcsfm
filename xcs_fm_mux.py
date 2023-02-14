#!/usr/bin/python3
import argparse
import copy
import glob
import logging
import os
import pickle
import shutil
import subprocess
import time
from pathlib import Path

import __main__
import numpy as np
from xcssl.encoding import TernaryEncoding
from xcssl.mux import make_mux_stream_env
from xcssl.xcs import make_xcs

_NUM_CPUS = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
_MUX_SEED = 0
_USE_FM = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--mux-num-addr-bits", type=int, required=True)
    parser.add_argument("--xcs-seed", type=int, required=True)
    parser.add_argument("--xcs-pop-size", type=int, required=True)
    parser.add_argument("--xcs-beta", type=float, required=True)
    parser.add_argument("--xcs-alpha", type=float, required=True)
    parser.add_argument("--xcs-epsilon-nought", type=float, required=True)
    parser.add_argument("--xcs-nu", type=float, required=True)
    parser.add_argument("--xcs-theta-ga", type=int, required=True)
    parser.add_argument("--xcs-chi", type=float, required=True)
    parser.add_argument("--xcs-tau", type=float, required=True)
    parser.add_argument("--xcs-upsilon", type=float, required=True)
    parser.add_argument("--xcs-mu", type=float, required=True)
    parser.add_argument("--xcs-theta-del", type=int, required=True)
    parser.add_argument("--xcs-delta", type=float, required=True)
    parser.add_argument("--xcs-theta-sub", type=int, required=True)
    parser.add_argument("--xcs-p-hash", type=float, required=True)
    parser.add_argument("--xcs-pred-i", type=float, required=True)
    parser.add_argument("--xcs-epsilon-i", type=float, required=True)
    parser.add_argument("--xcs-fitness-i", type=float, required=True)
    parser.add_argument("--xcs-p-exp", type=float, required=True)
    parser.add_argument("--xcs-theta-fm", type=int, required=True)
    parser.add_argument("--stree-max-depth", type=int, required=True)
    parser.add_argument("--stree-theta-build", type=int, required=True)
    parser.add_argument("--xcs-do-ga-subsumption", action="store_true")
    parser.add_argument("--xcs-do-as-subsumption", action="store_true")
    parser.add_argument("--monitor-steps-per-tick", type=int, required=True)
    parser.add_argument("--monitor-num-ticks", type=int, required=True)
    return parser.parse_args()


def main(args):
    save_path = _setup_save_path(args.experiment_name)
    _setup_logging(save_path)
    logging.info(str(args))

    env = make_mux_stream_env(num_addr_bits=args.mux_num_addr_bits,
                              seed=_MUX_SEED)

    xcs_hyperparams = {
        "seed": args.xcs_seed,
        "N": args.xcs_pop_size,
        "beta": args.xcs_beta,
        "alpha": args.xcs_alpha,
        "epsilon_nought": args.xcs_epsilon_nought,
        "nu": args.xcs_nu,
        "theta_ga": args.xcs_theta_ga,
        "chi": args.xcs_chi,
        "tau": args.xcs_tau,
        "upsilon": args.xcs_upsilon,
        "mu": args.xcs_mu,
        "theta_del": args.xcs_theta_del,
        "delta": args.xcs_delta,
        "theta_sub": args.xcs_theta_sub,
        "p_hash": args.xcs_p_hash,
        "pred_I": args.xcs_pred_i,
        "epsilon_I": args.xcs_epsilon_i,
        "fitness_I": args.xcs_fitness_i,
        "p_exp": args.xcs_p_exp,
        "theta_fm": args.xcs_theta_fm,
        "stree_max_depth": args.stree_max_depth,
        "stree_theta_build": args.stree_theta_build,
        "do_ga_subsumption": args.xcs_do_ga_subsumption,
        "do_as_subsumption": args.xcs_do_as_subsumption,
    }
    logging.info(xcs_hyperparams)

    encoding = TernaryEncoding(obs_space=env.obs_space)
    xcs = make_xcs(env, encoding, xcs_hyperparams, use_fm=_USE_FM)

    spt = args.monitor_steps_per_tick
    nt = args.monitor_num_ticks
    assert spt >= 1
    assert nt >= 1
    logging.info(f"Training for {spt} * {nt} = {spt*nt} steps")

    num_steps_done = 0
    for _ in range(nt):
        xcs.train_for_steps(spt)
        num_steps_done += spt

        # pop stats
        pop = xcs.pop
        logging.info(f"\nAfter {num_steps_done} steps")
        logging.info(f"Num macros: {pop.num_macros}")
        logging.info(f"Num micros: {pop.num_micros}")
        ratio = pop.num_micros / pop.num_macros
        logging.info(f"Micro:macro ratio: {ratio:.4f}")

        errors = [clfr.error for clfr in pop]
        min_error = min(errors)
        avg_error = sum([clfr.error * clfr.numerosity
                         for clfr in pop]) / pop.num_micros
        median_error = np.median(errors)
        max_error = max(errors)
        logging.info(f"Min error: {min_error}")
        logging.info(f"Mean error: {avg_error}")
        logging.info(f"Median error: {median_error}")
        logging.info(f"Max error: {max_error}")

        numerosities = [clfr.numerosity for clfr in pop]
        logging.info(f"Min numerosity: {min(numerosities)}")
        logging.info(f"Mean numerosity: {np.mean(numerosities)}")
        logging.info(f"Median numerosity: {np.median(numerosities)}")
        logging.info(f"Max numerosity: {max(numerosities)}")

        generalities = [clfr.condition.generality for clfr in pop]
        logging.info(f"Min generality: {min(generalities)}")
        logging.info(f"Mean generality: {np.mean(generalities)}")
        logging.info(f"Median generality: {np.median(generalities)}")
        logging.info(f"Max generality: {max(generalities)}")

        logging.info(f"Pop ops history: {pop.ops_history}")

        try:
            perf = xcs.calc_exploit_perf()
        except ValueError:
            logging.info("Exploit perf: not enough data")
        else:
            logging.info(f"Exploit perf: {perf}")

    #_save_xcs_pop(save_path, xcs.pop, num_steps_done)
    _save_main_py_script(save_path)


def _setup_save_path(experiment_name):
    save_path = Path(args.experiment_name)
    save_path.mkdir(exist_ok=False)
    return save_path


def _setup_logging(save_path):
    logging.basicConfig(filename=save_path / "experiment.log",
                        format="%(levelname)s: %(message)s",
                        level=logging.DEBUG)


def _save_xcs_pop(save_path, pop, num_steps_done):
    with open(f"{save_path}/xcs_pop_{num_steps_done}_steps.pkl", "wb") as fp:
        pickle.dump(pop, fp)


def _save_main_py_script(save_path):
    main_file_path = Path(__main__.__file__)
    shutil.copy(main_file_path, save_path)


if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    main(args)
    end_time = time.time()
    elpased = end_time - start_time
    logging.info(f"Runtime: {elpased:.3f}s with {_NUM_CPUS} cpus")
