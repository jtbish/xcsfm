import copy
import pickle
import sys
import warnings

import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from xcssl.encoding import TernaryEncoding
from xcssl.mux import make_mux_epoch_env, make_mux_stream_env
from xcssl.population import FastMatchingPopulation
from xcssl.rasterizer import IntegerObsSpaceRasterizer

_NUM_ADDR_BITSS = (4, 3, 2)

_ENV_SEED = 0

# full dataset for 2, 3 addr bits. sample for 4
_NUM_ENV_SAMPLESS = {2: 64, 3: 2048, 4: 50000}

_DUMMY_ACTION = 0

_NUM_TRIALS = 50

_DO_MATCH_MAT_COMP = False


def main():

    df = pd.DataFrame(columns=[
        "mux_num_addr_bits", "vpop_num_phenotypes", "idx_num_grid_dims",
        "trial_num", "idx_num_phenotypes", "idx_num_grid_refs",
        "idx_num_grid_cells", "idx_num_cells_per_phenotype_mean",
        "idx_cell_size_mean"
    ])

    for num_addr_bits in _NUM_ADDR_BITSS:

        total_num_bits = (num_addr_bits + 2**num_addr_bits)

        if num_addr_bits < 4:
            env = make_mux_epoch_env(num_addr_bits, seed=_ENV_SEED)
        else:
            env = make_mux_stream_env(num_addr_bits, seed=_ENV_SEED)

        with open(f"{total_num_bits}mux_repr/xcs_pop.pkl", "rb") as fp:
            vanilla_pop = pickle.load(fp)

        vanilla_pop_num_phenotypes = vanilla_pop.num_macros
        num_env_samples = _NUM_ENV_SAMPLESS[num_addr_bits]

        if _DO_MATCH_MAT_COMP:

            vanilla_matching_mat = np.full(shape=(num_env_samples,
                                                  vanilla_pop_num_phenotypes),
                                           fill_value=np.nan)
            env_copy = copy.deepcopy(env)
            try:
                env_copy.init_epoch()
            except AttributeError:
                pass

            for idx in range(num_env_samples):

                obs = env_copy.curr_obs
                (trace, _) = vanilla_pop.gen_matching_trace(obs)
                vanilla_matching_mat[idx, :] = trace

                env_copy.step(_DUMMY_ACTION)

            assert not np.any(np.isnan(vanilla_matching_mat))

        obs_space = env.obs_space
        encoding = TernaryEncoding(obs_space)
        d = len(obs_space)

        ks = list(reversed(range(1, (d + 1), 1)))

        seed = 0

        for k in ks:
            # only need to do single trial for k == d since only 1 option for k
            # dims
            num_trials = (1 if k == d else _NUM_TRIALS)
            for t in range(num_trials):

                rasterizer = IntegerObsSpaceRasterizer(obs_space,
                                                       num_grid_dims=k,
                                                       seed=seed)

                fm_pop = FastMatchingPopulation(
                    vanilla_pop=copy.deepcopy(vanilla_pop),
                    encoding=encoding,
                    rasterizer=rasterizer)

                env_copy = copy.deepcopy(env)
                try:
                    env_copy.init_epoch()
                except AttributeError:
                    pass

                num_matching_ops_done_trans_sample = []

                if _DO_MATCH_MAT_COMP:

                    fm_matching_mat = np.full(
                        shape=(num_env_samples, vanilla_pop_num_phenotypes),
                        fill_value=np.nan)

                    for idx in range(num_env_samples):

                        obs = env_copy.curr_obs
                        (trace, num_matching_ops_done
                         ) = fm_pop.gen_matching_trace(obs)

                        fm_matching_mat[idx, :] = trace

                        num_matching_ops_done_trans_sample.append(
                            num_matching_ops_done)

                        env_copy.step(_DUMMY_ACTION)

                    assert not np.any(np.isnan(fm_matching_mat))
                    assert np.all(vanilla_matching_mat == fm_matching_mat)

                else:
                    pass


#                    for idx in range(num_env_samples):
#
#                        obs = env_copy.curr_obs
#                        num_matching_ops_done = \
#                            fm_pop.gen_partial_matching_trace(obs)
#
#                        num_matching_ops_done_trans_sample.append(
#                            num_matching_ops_done)
#
#                        env_copy.step(_DUMMY_ACTION)

                idx = fm_pop._index

                idx_num_phenotypes = len(idx._phenotype_count_map)
                idx_num_grid_refs = sum(
                    len(cell) for cell in idx._grid_cell_phenotypes_map)
                idx_num_grid_cells = len(idx._grid_cell_phenotypes_map)

                idx_num_cells_per_phenotype_mean = np.mean([
                    len(cells)
                    for cells in idx._phenotype_grid_cells_map.values()
                ])
                idx_cell_size_mean = np.mean(
                    [len(cell) for cell in idx._grid_cell_phenotypes_map])
                #                idx_num_match_ops_mean = np.mean(
                #                    num_matching_ops_done_trans_sample)

                df_row = {
                    "mux_num_addr_bits": num_addr_bits,
                    "vpop_num_phenotypes": vanilla_pop_num_phenotypes,
                    "idx_num_grid_dims": k,
                    "trial_num": t,
                    "idx_num_phenotypes": idx_num_phenotypes,
                    "idx_num_grid_refs": idx_num_grid_refs,
                    "idx_num_grid_cells": idx_num_grid_cells,
                    "idx_num_cells_per_phenotype_mean":
                    idx_num_cells_per_phenotype_mean,
                    "idx_cell_size_mean": idx_cell_size_mean
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
