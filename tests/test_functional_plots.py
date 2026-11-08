import pandas as pd
import pytest

from results import visualise_results_chatterjee as plots


def test_estimated_functional_results_replace_sine_batch(monkeypatch, tmp_path):
    original = pd.DataFrame({
        "functional_form": ["linear", "sine", "sine"],
        "dgp_name": ["BernoulliNetwork"] * 3,
        "k": [1] * 3,
        "predictor_distribution": ["gaussian", "gaussian", "uniform_rdpg"],
        "source": ["original"] * 3,
    })
    replacement = pd.DataFrame({
        "functional_form": ["sine", "sine"],
        "dgp_name": ["BernoulliNetwork"] * 2,
        "k": [1] * 2,
        "predictor_distribution": ["gaussian", "uniform_rdpg"],
        "source": ["replacement"] * 2,
    })

    def fake_read(_results_dir, filenames):
        if filenames == plots.FUNCTIONAL_ESTIMATED_RESULT_FILES:
            return original
        if filenames == plots.FUNCTIONAL_ESTIMATED_SINE_RESULT_FILES:
            return replacement
        raise AssertionError(f"Unexpected result batch: {filenames}")

    monkeypatch.setattr(plots, "read_result_batch", fake_read)
    monkeypatch.setattr(plots, "process_results", lambda data: data)

    gaussian, uniform = plots.prepare_sharded_functional_results(
        tmp_path, k=1, latent="estimated"
    )

    assert gaussian["source"].tolist() == ["original", "replacement"]
    assert uniform["source"].tolist() == ["replacement"]


def test_functional_power_is_not_pooled_across_snrs():
    data = pd.DataFrame({
        "rho": [0.1, 0.1, 0.5, 0.5],
        "n": [100] * 4,
        "method": ["RVTest_permutation"] * 4,
        "functional_form": ["linear"] * 4,
        "Rejection": [False, False, True, True],
    })

    result = plots.aggregate_sharded_functional_results(data).set_index("rho")

    assert result["Rejection_mean"].to_dict() == {0.1: 0.0, 0.5: 1.0}


@pytest.mark.parametrize("latent, expected", [("true", True), ("estimated", False)])
@pytest.mark.parametrize("encode", [bool, str])
def test_k4_results_separate_latent_positions(monkeypatch, tmp_path, latent, expected, encode):
    data = pd.DataFrame([
        {
            "args": {"use_true_latent": encode(use_true)},
            "dgp_name": "BernoulliNetwork",
            "k": 4,
            "predictor_distribution": predictor,
            "Rejection": use_true,
        }
        for predictor in ("gaussian", "uniform_rdpg")
        for use_true in (True, False)
    ])
    monkeypatch.setattr(plots, "read_result_batch", lambda *_: data)
    monkeypatch.setattr(plots, "process_results", lambda data: data)

    gaussian, uniform = plots.prepare_sharded_functional_results(
        tmp_path, k=4, latent=latent
    )

    assert gaussian["Rejection"].tolist() == [expected]
    assert uniform["Rejection"].tolist() == [expected]


@pytest.mark.parametrize("k, count, latent", [(1, 7, "true"), (4, 11, "true"), (1, 8, "estimated"), (4, 11, "estimated")])
def test_functional_plots_cover_each_snr_and_predictor(monkeypatch, tmp_path, k, count, latent):
    data = pd.DataFrame({
        "rho": [0.1, 0.2, 0.3, 0.4, 0.5] * 2,
        "n": [100] * 5 + [300] * 5,
        "method": ["RVTest_permutation"] * 10,
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
