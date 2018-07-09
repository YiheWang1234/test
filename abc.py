import hyperopt

import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from hyperopt import fmin, tpe, hp


def objective(lambda1, lambda2):
    return lambda1 - 3*lambda2


space = [hp.loguniform('lambda1', np.log(.1), np.log(24.3)), hp.loguniform('lambda2', np.log(.1), np.log(24.3))]


def objective2(args):
    return objective(*args)


fmin(fn=objective2, space=space, algo=tpe.suggest, max_evals=10)