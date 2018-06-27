import GPyOpt

import numpy as np
from statsmodels.tsa.arima_model import ARIMA

n = 100
data = [1, 2]
np.random.seed(123)
gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}
for i in range(n):
    data.append(data[-1]*0.8 - data[-2]*0.3 + np.random.normal(0, 0.1, 1)[0])


def arima(p, d, q):
    print(p,d,q)
    try:
        model = ARIMA(data[:-10], order=(p, d, q))
        result = model.fit()
        #result.predict(start=n+1, end=n+10)
        return -result.aic
    except ValueError:
        return 100000

