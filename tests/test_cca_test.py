import numpy as np
import pytest

from src.methods import CanonicalCorrelationTest
from src.test_functions import first_cca_component


def test_first_cca_component_matches_known_one_dimensional_correlation():
    x = np.array([0.0, 1.0, 3.0, 4.0])[:, None]
    z = np.array([1.0, 4.0, 2.0, 7.0])[:, None]
    assert first_cca_component(x, z) == pytest.approx(
        abs(np.corrcoef(x[:, 0], z[:, 0])[0, 1])
    )
    assert first_cca_component(x, np.ones_like(z)) == 0.0


def test_cca_permutation_matches_observed_coefficient_and_is_reproducible():
    data_rng = np.random.default_rng(25)
    data = {
        "Z": data_rng.normal(size=(20, 2)),
        "Y": data_rng.normal(size=(20, 2)),
    }
    results = []
    for _ in range(2):
        method = CanonicalCorrelationTest(
            use_true_latent=True,
            npermutations=9,
            rng=np.random.default_rng(4),
        )
        method.fit(data)
        results.append(method)

    first, second = results
    assert first.get_name() == "CCA_PermutationTest_covariate"
    assert first.test_stat_estimate == pytest.approx(
        first_cca_component(data["Z"], data["Y"])
    )
    assert first.null_distribution == pytest.approx(second.null_distribution)
    assert first.pvalue == second.pvalue
    assert 0.0 <= first.pvalue <= 1.0
    assert first.reject_null == (first.pvalue < first.alpha)
