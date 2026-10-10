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
        "n": [100] * 10,
        "method": ["DC"] * 10,
        "functional_form": ["linear"] * 10,
        "Rejection": [False] * 5 + [True] * 5,
    })
    calls = []

    def record_plot(aggregated, output_dir, *, predictor_name, filename, snr):
        assert set(aggregated["rho"]) == {snr}
        assert output_dir == tmp_path
        assert filename.endswith(f"_snr_{snr:g}")
        calls.append((predictor_name, snr, filename))

    monkeypatch.setattr(plots, "plot_sharded_functional_snr", record_plot)
    plots.plot_sharded_functional_figures(data, data, tmp_path)

    assert {(predictor, snr) for predictor, snr, _ in calls} == {
        (predictor, snr)
        for predictor in ["Gaussian", "uniform RDPG"]
        for snr in [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    assert len({filename for _, _, filename in calls}) == 10
