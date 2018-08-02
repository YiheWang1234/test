import numpy as np
import pandas as pd
import datarobot as dr

data = np.load('output.npz')

z_train = data['z_t']
z_val = data['z']

n_train, pz = z_train.shape
n_val, _ = z_val.shape

x_train = data['X_train'][0:n_train, :]
x_train_out = data['X_train_out'][0:n_train]

x_val = data['X_val'][0:n_val, :]
x_val_out = data['X_val_out'][0:n_val]

sigma_train = data['sig_t']
sigma_val = data['sigma_val']

data_train_x = np.concatenate((x_train_out.reshape((n_train, 1)),
                               x_train), axis=1)

data_train_z = np.concatenate((x_train_out.reshape((n_train, 1)),
                               z_train), axis=1)

data_test_x = np.concatenate((x_val_out.reshape((n_val, 1)),
                               x_val), axis=1)

data_test_z = np.concatenate((x_val_out.reshape((n_val, 1)),
                               z_val), axis=1)

dtr_x = pd.DataFrame(data_train_x)
dtr_z = pd.DataFrame(data_train_z)
dte_x = pd.DataFrame(data_test_x)
dte_z = pd.DataFrame(data_test_z)

dtr_x.to_excel('dtr_x.xlsx', index=False)
dtr_z.to_excel('dtr_z.xlsx', index=False)
dte_x.to_excel('dte_x.xlsx', index=False)
dte_z.to_excel('dte_z.xlsx', index=False)

# ========= #
# Datarobot #
# ========= #

API_TOKEN = '-aP9mLf539Zy_1FLr2FzZkY8ZeoI59uA'
END_POINT = 'https://app.datarobot.com/api/v2'

# Intantiate DataRobot Client
dr.Client(token= API_TOKEN, endpoint=END_POINT)
def get_projects_by_name(name):
    return list(filter(lambda x: name in x.project_name, dr.Project.list()))

# ======= #
# Model x #
# ======= #

# create project
TRAIN_SET = '/Users/alex/Desktop/roar/test/dtr_x.xlsx'
TEST_SET = '/Users/alex/Desktop/roar/test/dte_x.xlsx'

project_autopilot = dr.Project.create(TRAIN_SET, project_name='AE_x')
project_autopilot.set_target(target='0', mode=dr.AUTOPILOT_MODE.QUICK, worker_count=4)
models = project_autopilot.get_models()

blue_prints = project_autopilot.get_blueprints()
blue_prints

# prediction
projects = get_projects_by_name('AE_x')
project_autopilot = projects[0]
dataset = project_autopilot.upload_dataset(TEST_SET)
models = project_autopilot.get_models()
predict_job = models[0].request_predictions(dataset.id)
predictions = predict_job.get_result_when_complete()

MSE_X = np.sum((np.array(predictions.iloc[:, 0]) - x_val_out)**2)/n_val
MSE_X
# 0.11836186431663784

# ======= #
# Model z #
# ======= #

# create project

TRAIN_SET = '/Users/alex/Desktop/roar/test/dtr_z.xlsx'
TEST_SET = '/Users/alex/Desktop/roar/test/dte_z.xlsx'

project_autopilot = dr.Project.create(TRAIN_SET, project_name='AE_z')
project_autopilot.set_target(target='0', mode=dr.AUTOPILOT_MODE.QUICK, worker_count=4)
models = project_autopilot.get_models()

# prediction
projects = get_projects_by_name('AE_z')
project_autopilot = projects[0]
dataset = project_autopilot.upload_dataset(TEST_SET)
models = project_autopilot.get_models()
predict_job = models[0].request_predictions(dataset.id)
predictions = predict_job.get_result_when_complete()

MSE_Z = np.sum((np.array(predictions.iloc[:, 0]) - x_val_out)**2)/n_val
MSE_Z
# 0.5925748682059658

# ========== #
# Sigma mean #
# ========== #

AVE_sigma = np.sum(sigma_val[:, -1]**2)/n_val
AVE_sigma
# 3.7791295369466145

from arch import arch_model
varvec = []
for i in range(n_val):
    am = arch_model(data_test_x[i, :], vol='Garch', p=1, o=0, q=1, dist='Normal')
    res = am.fit(update_freq=10)
    forecast = res.forecast()
    varvec.append(float(forecast.variance.iloc[-1]))

AVE_garch_sigma = np.mean(varvec)
# 4.0762582624823782

