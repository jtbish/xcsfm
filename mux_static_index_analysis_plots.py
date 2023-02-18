import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_NROWS = 2
_NCOLS = 1

_NUM_ADDR_BITSS = (2, 3, 4)

_NUM_OBS_SPACE_DIMSS = {nab: nab + (2**nab) for nab in _NUM_ADDR_BITSS}

_NUM_TRIALS = 50

_COLORS = {2: "tab:green", 3: "tab:blue", 4: "tab:red"}


def main():
    with open("mux_static_index_analysis_df.pkl", "rb") as fp:
        df = pickle.load(fp)

    # first, add computed cols for plotting to df

    # independent var.
    f = np.vectorize(lambda ingd, mnab: ingd / _NUM_OBS_SPACE_DIMSS[mnab])
    df["idx_num_grid_dims_normd"] = f(df.idx_num_grid_dims,
                                      df.mux_num_addr_bits)

    # dependent vars.
    df["idx_num_grid_refs_normd"] = df["idx_num_grid_refs"] / \
        df["vpop_num_phenotypes"]
    df["idx_cell_size_mean_normd"] = df["vpop_num_phenotypes"] / \
        df["idx_cell_size_mean"]

    (fig, axs) = plt.subplots(nrows=_NROWS, ncols=_NCOLS, sharex=True)

    _plot_idx_num_grid_refs_normd(axs[0], df)
    _plot_idx_cell_size_mean_normd(axs[1], df)

    legend_handles = [
        Line2D([0], [0], lw=1.5, color=color) for color in _COLORS.values()
    ]
    legend_labels = [f"{nosd}-MUX" for nosd in _NUM_OBS_SPACE_DIMSS.values()]
    fig.legend(legend_handles,
               legend_labels,
               loc="center left",
               bbox_to_anchor=(1, 0.5))
    fig.tight_layout()

    plt.savefig("mux_static_index_analysis_plots.pdf", bbox_inches="tight")


def _plot_idx_num_grid_refs_normd(ax, df):
    _plot_dependent_var_on_ax(ax,
                              df,
                              dependent_var_col_name="idx_num_grid_refs_normd")

    ax.set_xticks([0.1 * i for i in range(1, 10 + 1)])
    ax.set_yscale("log")
    ax.set_ylim(bottom=10**0, top=(10**5 / 2))
    ax.set_ylabel("Memory usage")
    ax.grid()


def _plot_idx_cell_size_mean_normd(ax, df):
    _plot_dependent_var_on_ax(
        ax, df, dependent_var_col_name="idx_cell_size_mean_normd")

    ax.set_xticks([0.1 * i for i in range(1, 10 + 1)])
    ax.set_xlabel("k (normd)")
    ax.set_yscale("log")
    ax.set_ylim(bottom=10**0, top=(10**5 / 2))
    ax.set_ylabel("Speedup")
    ax.grid()


def _plot_dependent_var_on_ax(ax, df, dependent_var_col_name):
    for nab in _NUM_ADDR_BITSS:
        sub_df = df.loc[df["mux_num_addr_bits"] == nab]

        xs = list(sub_df.idx_num_grid_dims_normd.unique())
        ys = []
        for x in xs:
            # take a mean over all rows for the dependent var.
            sub_sub_df = sub_df.loc[sub_df["idx_num_grid_dims_normd"] == x]

            if x == 1.0:
                assert len(sub_sub_df) == 1
            else:
                assert len(sub_sub_df) == _NUM_TRIALS

            ys.append(np.mean(sub_sub_df[dependent_var_col_name]))

        ax.plot(xs, ys, color=_COLORS[nab])


if __name__ == "__main__":
    main()
