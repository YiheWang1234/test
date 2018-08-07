import numpy as np
import pandas as pd
import datarobot as dr

data = np.load('output_rec_z.npz')

z_train=data['z_train']
z_test=data['z_test']

n_train, pz = z_train.shape
n_test, _ = z_test.shape

x_train=data['X_train1'][0:n_train, :]
x_test=data['X_val1'][0:n_val, :]
x_train_out=data['X_train_out'][0:n_train]
x_test_out=data['X_val_out'][0:n_val]

data_train_x = np.concatenate((x_train_out.reshape((n_train, 1)),
                               x_train), axis=1)

data_train_z = np.concatenate((x_train_out.reshape((n_train, 1)),
                               z_train), axis=1)

data_test_x = np.concatenate((x_test_out.reshape((n_test, 1)),
                              x_test), axis=1)

data_test_z = np.concatenate((x_test_out.reshape((n_test, 1)),
                              z_test), axis=1)

dtr_x = pd.DataFrame(data_train_x)
dtr_z = pd.DataFrame(data_train_z)
dte_x = pd.DataFrame(data_test_x)
dte_z = pd.DataFrame(data_test_z)

dtr_x.to_excel('dtr_x_re.xlsx', index=False)
dtr_z.to_excel('dtr_z_re.xlsx', index=False)
dte_x.to_excel('dte_x_re.xlsx', index=False)
dte_z.to_excel('dte_z_re.xlsx', index=False)

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
TRAIN_SET = '/Users/alex/Desktop/roar/test/dtr_x_re.xlsx'
TEST_SET = '/Users/alex/Desktop/roar/test/dte_x_re.xlsx'

project_autopilot = dr.Project.create(TRAIN_SET, project_name='AE_x_re')
project_autopilot.set_target(target='0', mode=dr.AUTOPILOT_MODE.QUICK, worker_count=4)
models = project_autopilot.get_models()

# prediction
projects = get_projects_by_name('AE_x_re')
project_autopilot = projects[0]
dataset = project_autopilot.upload_dataset(TEST_SET)
models = project_autopilot.get_models()
predict_job = models[0].request_predictions(dataset.id)
predictions = predict_job.get_result_when_complete()

MSE_X = np.sum((np.array(predictions.iloc[:, 0]) - x_test_out)**2)/n_test
MSE_X


# ======= #
# Model z #
# ======= #


# create project
TRAIN_SET = '/Users/alex/Desktop/roar/test/dtr_z_re.xlsx'
TEST_SET = '/Users/alex/Desktop/roar/test/dte_z_re.xlsx'

project_autopilot = dr.Project.create(TRAIN_SET, project_name='AE_z_re')
project_autopilot.set_target(target='0', mode=dr.AUTOPILOT_MODE.QUICK, worker_count=4)
models = project_autopilot.get_models()

# prediction
projects = get_projects_by_name('AE_z_re')
project_autopilot = projects[0]
dataset = project_autopilot.upload_dataset(TEST_SET)
models = project_autopilot.get_models()
predict_job = models[0].request_predictions(dataset.id)
predictions = predict_job.get_result_when_complete()

MSE_Z = np.sum((np.array(predictions.iloc[:, 0]) - x_test_out)**2)/n_test
MSE_Z

