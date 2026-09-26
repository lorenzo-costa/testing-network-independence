"""Shared default series styles; study recipes may override them."""

METHOD_LABELS = {
    "RVTest_permutation": "RV (permutation)",
    "RVTest_asymptotic_independence": "RV (asymptotic — independence)",
    "RVTest_asymptotic_zero_covariance": ("RV (asymptotic — zero covariance)"),
    "CCA": "CCA",
    "DC": "MGC",
    "MRQAP": "MRQAP (adjacency)",
}

COLORS = {
    "RVTest_permutation": "#E69F00",
    "RVTest_asymptotic_independence": "#E69F00",
    "RVTest_asymptotic_zero_covariance": "#D55E00",
    "CCA": "#0072B2",
    "DC": "#009E73",
    "MRQAP": "#CC79A7",
}

MARKERS = {
    "RVTest_permutation": "o",
    "RVTest_asymptotic_independence": "D",
    "RVTest_asymptotic_zero_covariance": "X",
    "CCA": "s",
    "DC": "^",
    "MRQAP": "v",
}

LINESTYLES = {
    "RVTest_permutation": "-",
    "RVTest_asymptotic_independence": "--",
    "RVTest_asymptotic_zero_covariance": ":",
    "CCA": "-",
    "DC": "-",
    "MRQAP": "-",
}
