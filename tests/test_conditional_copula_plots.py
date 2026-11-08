import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from src.helper_functions.plot_functions import visualise_conditional_copula_latent


def _sample(offset):
    classes = np.array([[0], [0], [1], [1]])
    latent_z = np.array([[0.0], [1.0], [2.0], [3.0]]) + offset
    latent_y = np.array([[-2.0], [-1.0], [4.0], [5.0]]) + offset
    return latent_z, latent_y, classes


def test_visualise_conditional_copula_latent_separates_classes(monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    figure = visualise_conditional_copula_latent(
        [_sample(0), _sample(10)],
        setup_titles=["Gaussian", "Pareto"],
        layout="separate",
        kdplot=False,
    )

    assert len(figure.axes) == 4
    assert [axis.get_title() for axis in figure.axes[:2]] == ["X = 0", "X = 1"]
    assert figure.axes[0].get_ylabel() == "Gaussian\nLatent Z"
    assert figure.axes[2].get_ylabel() == "Pareto\nLatent Z"
    assert figure.axes[0].collections[0].get_offsets()[:, 0].tolist() == [-2.0, -1.0]
    assert figure.axes[1].collections[0].get_offsets()[:, 0].tolist() == [4.0, 5.0]


def test_visualise_conditional_copula_latent_overlays_classes(monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    figure = visualise_conditional_copula_latent(
        [_sample(0), _sample(10)],
        setup_titles=["Gaussian", "Pareto"],
        layout="overlay",
        kdplot=False,
    )

    assert len(figure.axes) == 2
    assert [axis.get_title() for axis in figure.axes] == ["Gaussian", "Pareto"]
    assert [collection.get_label() for collection in figure.axes[0].collections] == [
        "X = 0",
        "X = 1",
    ]
    assert len(figure.axes[0].get_legend().get_texts()) == 2


@pytest.mark.parametrize(
    "samples, message",
    [
        ([], "at least one"),
        ([(np.ones((2, 1)), np.ones((2, 1)), np.array([0, 2]))], "C=2"),
    ],
)
def test_visualise_conditional_copula_latent_validates_samples(samples, message):
    with pytest.raises(ValueError, match=message):
        visualise_conditional_copula_latent(samples, kdplot=False)


def test_visualise_conditional_copula_latent_saves_both_formats(tmp_path):
    output = tmp_path / "latent_comparison"

    visualise_conditional_copula_latent(
        [_sample(0)], layout="overlay", kdplot=False, save_path=output
    )

    assert output.with_suffix(".png").is_file()
    assert output.with_suffix(".pdf").is_file()
