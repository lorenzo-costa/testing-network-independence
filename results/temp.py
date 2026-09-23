import pandas as pd
import numpy as np
from visualise_linear_model import prepare_linear_model_results

ASYMPTOTIC_RESULT_FILES: tuple[str, ...] = (
    "linear_model_asymptotic_results_61636044_shard-000-of-003.csv",
    "linear_model_asymptotic_results_61636044_shard-001-of-003.csv",
    "linear_model_asymptotic_results_61636044_shard-002-of-003.csv",
)

dt = prepare_linear_model_results('', ASYMPTOTIC_RESULT_FILES, include_methods=("RVTest_asymptotic",))
