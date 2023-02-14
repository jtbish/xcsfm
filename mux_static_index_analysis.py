import copy
import math
import pickle
import sys
import warnings

import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from xcssl.encoding import TernaryEncoding
from xcssl.lsh import HammingLSH
from xcssl.mux import make_mux_epoch_env, make_mux_stream_env
from xcssl.population import FastMatchingPopulation

_NUM_ADDR_BITSS = (2, 3, 4)

_ENV_SEED = 0

# full dataset for 2, 3 addr bits. sample for 4
_NUM_ENV_SAMPLESS = {2: 64, 3: 2048, 4: 50000}

_DUMMY_ACTION = 0

_P_LO_MULT = 0.1
_P_HI_MULT = 0.9

_NUM_TRIALS = 50


def main():

    df = pd.DataFrame(columns=[
        "mux_num_addr_bits", "vpop_num_phenotypes", "lsh_num_projs",
        "trial_num", "sp_num_phenotypes", "sp_num_cells", "sp_cell_size_mean",
        "sp_cell_size_med", "sp_subsumer_genr_mean", "sp_subsumer_genr_med",
        "sp_num_match_ops_mean", "sp_num_match_ops_med"
    ])

    for num_addr_bits in _NUM_ADDR_BITSS:

        total_num_bits = (num_addr_bits + 2**num_addr_bits)

        env = make_mux_epoch_env(num_addr_bits, seed=_ENV_SEED)

        with open(f"{total_num_bits}mux_repr/xcs_pop.pkl", "rb") as fp:
            vanilla_pop = pickle.load(fp)

        vanilla_pop_num_phenotypes = vanilla_pop.num_macros
        num_env_samples = _NUM_ENV_SAMPLESS[num_addr_bits]

        #        vanilla_matching_mat = np.full(shape=(num_env_samples,
        #                                              vanilla_pop_num_phenotypes),
        #                                       fill_value=np.nan)
        #        env_copy = copy.deepcopy(env)
        #
        #        env_copy.init_epoch()
        #
        #        for idx in range(num_env_samples):
        #
        #            obs = env_copy.curr_obs
        #            (trace, _) = vanilla_pop.gen_matching_trace(obs)
        #            vanilla_matching_mat[idx, :] = trace
        #
        #            env_copy.step(_DUMMY_ACTION)
        #
        #        assert not np.any(np.isnan(vanilla_matching_mat))

        encoding = TernaryEncoding(env.obs_space)
        d = encoding.calc_num_phenotype_vec_dims()

        p_lo = math.ceil(_P_LO_MULT * d)
        p_hi = math.floor(_P_HI_MULT * d)

        ps = list(range(p_lo, (p_hi + 1), 1))

        seed = 0

        for p in ps:
            for t in range(_NUM_TRIALS):

                lsh = HammingLSH(num_dims=d, num_projs=p, seed=seed)

                fm_pop = FastMatchingPopulation(
                    vanilla_pop=copy.deepcopy(vanilla_pop),
                    encoding=encoding,
                    lsh=lsh)

                env_copy = copy.deepcopy(env)
                env_copy.init_epoch()

                #                fm_matching_mat = np.full(shape=(num_env_samples,
                #                                                 vanilla_pop_num_phenotypes),
                #                                          fill_value=np.nan)
                num_matching_ops_done_trans_sample = []

                for idx in range(num_env_samples):

                    obs = env_copy.curr_obs
                    num_matching_ops_done = fm_pop.gen_partial_matching_trace(
                        obs)

                    #                    fm_matching_mat[idx, :] = trace

                    num_matching_ops_done_trans_sample.append(
                        num_matching_ops_done)

                    env_copy.step(_DUMMY_ACTION)


#                assert not np.any(np.isnan(fm_matching_mat))
#                assert np.all(vanilla_matching_mat == fm_matching_mat)

                sp = fm_pop._index

                sp_num_phenotypes = len(sp._phenotype_count_map)
                sp_num_cells = len(sp._lsh_key_cell_map)

                sp_cell_sizes = [
                    cell.size for cell in sp._lsh_key_cell_map.values()
                ]
                sp_cell_size_mean = np.mean(sp_cell_sizes)
                sp_cell_size_med = np.median(sp_cell_sizes)

                sp_subsumer_genrs = [
                    encoding.calc_phenotype_generality(cell.subsumer_phenotype)
                    for cell in sp._lsh_key_cell_map.values()
                ]
                sp_subsumer_genr_mean = np.mean(sp_subsumer_genrs)
                sp_subsumer_genr_med = np.median(sp_subsumer_genrs)

                num_matching_ops_mean = \
                    np.mean(num_matching_ops_done_trans_sample)
                num_matching_ops_med = \
                    np.median(num_matching_ops_done_trans_sample)

                df_row = {
                    "mux_num_addr_bits": num_addr_bits,
                    "vpop_num_phenotypes": vanilla_pop_num_phenotypes,
                    "lsh_num_projs": p,
                    "trial_num": t,
                    "sp_num_phenotypes": sp_num_phenotypes,
                    "sp_num_cells": sp_num_cells,
                    "sp_cell_size_mean": sp_cell_size_mean,
                    "sp_cell_size_med": sp_cell_size_med,
                    "sp_subsumer_genr_mean": sp_subsumer_genr_mean,
                    "sp_subsumer_genr_med": sp_subsumer_genr_med,
                    "sp_num_match_ops_mean": num_matching_ops_mean,
                    "sp_num_match_ops_med": num_matching_ops_med
                }
                print(df_row)
                sys.stdout.flush()
                df = df.append(df_row, ignore_index=True)

                seed += 1

        print("\n")

    with open("mux_static_index_analysis_df.pkl", "wb") as fp:
        pickle.dump(df, fp)

if __name__ == "__main__":
    main()
