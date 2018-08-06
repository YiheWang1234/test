import numpy as np
import datarobot as dr
from AE_ts_model import open_data
import pandas as pd

"""Load the data"""
X_train, X_val, y_train, y_val = open_data('./UCR_TS_Archive_2015')
X_train, X_train_out = X_train[:, :-1], X_train[:, -1]
X_val, X_val_out = X_val[:, :-1], X_val[:, -1]

N = X_train.shape[0]
Nval = X_val.shape[0]
D = X_train.shape[1]

ave = 10
Z_train = np.zeros((N, D - ave))
Z_val = np.zeros((Nval, D - ave))

for i in range(D - ave):
    Z_train[:, i] = np.mean(X_train[:, i:(i+ave)], axis=1)
    Z_val[:, i] = np.mean(X_val[:, i:(i+ave)], axis=1)

data_train_zave = np.concatenate((X_train_out.reshape((N, 1)),
                                  Z_train), axis=1)

data_test_zave = np.concatenate((X_val_out.reshape((Nval, 1)),
                                 Z_val), axis=1)

dtr_zave = pd.DataFrame(data_train_zave)
dte_zave = pd.DataFrame(data_test_zave)

dtr_zave.to_excel('dtr_zave.xlsx', index=False)
dte_zave.to_excel('dte_zave.xlsx', index=False)

# ========= #
# Datarobot #
# ========= #

API_TOKEN = '-aP9mLf539Zy_1FLr2FzZkY8ZeoI59uA'
END_POINT = 'https://app.datarobot.com/api/v2'

# Intantiate DataRobot Client
dr.Client(token= API_TOKEN, endpoint=END_POINT)
def get_projects_by_name(name):
    return list(filter(lambda x: name in x.project_name, dr.Project.list()))

# ========== #
# Model zave #
# ========== #

# create project

TRAIN_SET = '/Users/alex/Desktop/roar/test/dtr_zave.xlsx'
TEST_SET = '/Users/alex/Desktop/roar/test/dte_zave.xlsx'
# TRAIN_SET = '/Users/yihewang/Desktop/test/dtr_z.xlsx'
# TEST_SET = '/Users/yihewang/Desktop/test/dte_z.xlsx'

project_autopilot = dr.Project.create(TRAIN_SET, project_name='AE_zave')
project_autopilot.set_target(target='0', mode=dr.AUTOPILOT_MODE.QUICK, worker_count=4)
models = project_autopilot.get_models()

# prediction
projects = get_projects_by_name('AE_zave')
project_autopilot = projects[0]
dataset = project_autopilot.upload_dataset(TEST_SET)
models = project_autopilot.get_models()
predict_job = models[0].request_predictions(dataset.id)
predictions = predict_job.get_result_when_complete()

MSE_Z = np.sum((np.array(predictions.iloc[:, 0]) - X_val_out)**2)/Nval
MSE_Z
# 0.10906995778314993