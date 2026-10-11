import pandas as pd

from results import visualise_results_chatterjee as plots


def test_functional_power_is_not_pooled_across_snrs():
    data = pd.DataFrame({
        "rho": [0.1, 0.1, 0.5, 0.5],
        "n": [100] * 4,
        "method": ["DC"] * 4,
        "functional_form": ["linear"] * 4,
        "Rejection": [False, False, True, True],
    })

    result = plots.aggregate_sharded_functional_results(data).set_index("rho")

    assert result["Rejection_mean"].to_dict() == {0.1: 0.0, 0.5: 1.0}


def test_functional_plots_cover_each_snr_and_predictor(monkeypatch, tmp_path):
    data = pd.DataFrame({
        "rho": [0.1, 0.2, 0.3, 0.4, 0.5] * 2,
        "n": [100] * 5 + [300] * 5,
        "method": ["DC"] * 10,
        "functional_form": ["linear"] * 10,
        "Rejection": [False] * 5 + [True] * 5,
    })
    calls = []

    def record_plot(fig, output_dir, filename):
        assert output_dir == tmp_path
        assert len(fig.axes) == 1
        assert all(len(ax.get_xticks()) == 7 for ax in fig.axes)
        assert all(len(ax.lines) >= 1 for ax in fig.axes)
        assert fig.axes[0].lines[0].get_ydata().tolist() == [1.0]
        calls.append(filename)
        plots.plt.close(fig)

    monkeypatch.setattr(plots, "save_figure", record_plot)
    plots.plot_sharded_functional_figures(data, data, tmp_path)

    assert set(calls) == {
        f"{prefix}_n300_k1_snr_{snr:g}"
        for prefix in ("28_functional_forms_gaussian_predictors",
                       "29_functional_forms_uniform_rdpg_predictors")
        for snr in [0.1, 0.2, 0.3, 0.4, 0.5]
    }
