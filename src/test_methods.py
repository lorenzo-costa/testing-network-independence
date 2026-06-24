import numpy as np

from .test_functions.rv_cca_coefficients import rv_coefficient_adjusted
from .test_functions.rv_cca_coefficients import first_cca_component
from .test_functions.ac_coefficient import multivariate_ac_coefficient_permutation
from .helper_functions.dgp._base_class import BaseMethod, BasePermutationTest
from .helper_functions.imhof import imhof

import sys
import os
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))





