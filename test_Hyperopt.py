import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from hyperopt import fmin, tpe, hp

n = 200
data = [0, 1]
np.random.seed(123)
err = np.random.normal(0, 1, n+2)
gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}
for i in range(n):
    data.append(data[-1]*0.6 - data[-2]*0.3 + 0.7*err[i] - 0.2*err[i+1])


def arima(p,d,q):
    #p, d, q = list(params)

    p = int(p)
    d = int(d)
    q = int(q)

    #print(p, d, q)
    try:
        model = ARIMA(data[:-10], order=(p, d, q))
        result = model.fit()
        #result.predict(start=n+1, end=n+10)
        #print(result.aic)
        return result.aic
    except ValueError:
        #print("not use")
        return np.inf


space1 = (hp.randint("p", 4), hp.randint("d", 4), hp.randint("q", 4))
space1 = [("p", hp.randint("p", 4)), ("d", hp.randint("d", 4)), ("q", hp.randint("q", 4))]
space1 = hp.uniform('p', -10, 10)

fmin(fn=arima, space=space1, algo=tpe.suggest, max_evals=100)

#python3 not working, implementated in python2
from hyperopt import fmin, tpe, hp
best = fmin(fn=lambda x: x ** 2,
    space=hp.uniform('x', -10, 10),
    algo=tpe.suggest,
    max_evals=100)
print(best)

