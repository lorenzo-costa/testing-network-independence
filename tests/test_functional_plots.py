import pandas as pd
import pytest

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


@pytest.mark.parametrize("k, count, latent", [(1, 7, "true"), (4, 11, "true"), (1, 7, "estimated")])
def test_functional_plots_cover_each_snr_and_predictor(monkeypatch, tmp_path, k, count, latent):
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
        assert len(fig.axes) in (1, 4)
        assert all(len(ax.get_xticks()) == count for ax in fig.axes)
        assert all(len(ax.lines) >= 1 for ax in fig.axes)
        if len(fig.axes) == 1:
            assert fig.axes[0].lines[0].get_ydata().tolist() == [1.0]
            assert fig.axes[0].get_title() == f"{latent.capitalize()} latent positions"
        else:
            assert [ax.lines[0].get_ydata().tolist() for ax in fig.axes] == [
                [0.0], [0.0], [1.0], [1.0],
            ]
            assert f"{latent.capitalize()} latent positions" in fig._suptitle.get_text()
        calls.append(filename)
        plots.plt.close(fig)

    monkeypatch.setattr(plots, "save_figure", record_plot)
    plots.plot_sharded_functional_figures(data, data, tmp_path, k=k, latent=latent)

    assert set(calls) == {
        f"{prefix}_n300_k{k}_{latent}_latent_snr_{snr:g}"
        for prefix in ("28_functional_forms_gaussian_predictors",
                       "29_functional_forms_uniform_rdpg_predictors")
        for snr in [0.1, 0.2, 0.3, 0.4, 0.5]
    } | {
        f"28_functional_alternatives_k{k}_{latent}_latent_snr_{snr:g}"
        for snr in [0.1, 0.2, 0.3, 0.4, 0.5]
    }
