import pickle

import matplotlib.pyplot as plt
import numpy as np

_NROWS = 2
_NCOLS = 2

_NUM_ADDR_BITSS = (2, 3, 4)

_NUM_PHENOTYPE_VEC_DIMSS = {
    nab: (2 * (nab + 2**nab))
    for nab in _NUM_ADDR_BITSS
}
_SUBSUMER_MAX_GENRS = {nab: (nab + 2**nab) for nab in _NUM_ADDR_BITSS}

_NUM_TRIALS = 50

_COLORS = {2: "tab:green", 3: "tab:blue", 4: "tab:red"}


def main():
    with open("mux_static_index_analysis_df.pkl", "rb") as fp:
        df = pickle.load(fp)

    # first, add computed cols for plotting to df

    # independent var.
    f = np.vectorize(lambda lnp, mnab: lnp / _NUM_PHENOTYPE_VEC_DIMSS[mnab])
    df["lsh_num_projs_normd"] = f(df.lsh_num_projs, df.mux_num_addr_bits)

    # dependent vars.
    df["sp_num_cells_normd"] = df["sp_num_cells"] / df["sp_num_phenotypes"]

    f = np.vectorize(lambda sgm, mnab: sgm / _SUBSUMER_MAX_GENRS[mnab])
    df["sp_subsumer_genr_mean_normd"] = f(df.sp_subsumer_genr_mean,
                                          df.mux_num_addr_bits)

    df["sp_num_match_ops_mean_normd"] = (df["sp_num_match_ops_mean"] /
                                         df["vpop_num_phenotypes"])

    (fig, axs) = plt.subplots(nrows=_NROWS, ncols=_NCOLS, sharex=True)

    _plot_num_cells_normd(axs[0][0], df)
    _plot_subsumer_genr_normd(axs[0][1], df)
    _plot_cell_size_logd(axs[1][0], df)
    _plot_num_match_ops_normd(axs[1][1], df)

    fig.tight_layout()

    plt.savefig("mux_static_index_analysis_plots.pdf", bbox_inches="tight")


def _plot_num_cells_normd(ax, df):
    _plot_dependent_var_on_ax(ax,
                              df,
                              dependent_var_col_name="sp_num_cells_normd")
    ax.set_ylabel("Num cells (normd)")


def _plot_cell_size_logd(ax, df):
    _plot_dependent_var_on_ax(ax,
                              df,
                              dependent_var_col_name="sp_cell_size_mean")
    ax.set_xlabel("LSH num projs (normd)")
    ax.set_ylabel("Cell size")
    ax.set_yscale("log")


def _plot_subsumer_genr_normd(ax, df):
    _plot_dependent_var_on_ax(
        ax, df, dependent_var_col_name="sp_subsumer_genr_mean_normd")
    ax.set_ylabel("Subsumer genr (normd)")


def _plot_num_match_ops_normd(ax, df):
    _plot_dependent_var_on_ax(
        ax, df, dependent_var_col_name="sp_num_match_ops_mean_normd")
    ax.set_xlabel("LSH num projs (normd)")
    ax.set_ylabel("Num match ops (normd)")


def _plot_dependent_var_on_ax(ax, df, dependent_var_col_name):
    for nab in _NUM_ADDR_BITSS:
        sub_df = df.loc[df["mux_num_addr_bits"] == nab]

        xs = list(sub_df.lsh_num_projs_normd.unique())
        ys = []
        for x in xs:
            # take a mean over all rows for the dependent var.
            sub_sub_df = sub_df.loc[sub_df["lsh_num_projs_normd"] == x]
            assert len(sub_sub_df) == _NUM_TRIALS
            ys.append(np.mean(sub_sub_df[dependent_var_col_name]))

        ax.plot(xs, ys, color=_COLORS[nab])


if __name__ == "__main__":
    main()
