import numpy as np
import pandas as pd
import datarobot as dr

data = np.load('output_rec_z3.npz')

z_train = data['z_train']
z_test = data['z_test']

n_train, pz = z_train.shape
n_test, _ = z_test.shape

x_train = data['x_train'][0:n_train, 10:]
x_test = data['x_test'][0:n_test, 10:]
x_train_out = data['x_train_out'][0:n_train]
x_test_out = data['x_test_out'][0:n_test]

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

project_autopilot = dr.Project.create(TRAIN_SET, project_name='AE_x_re10')
project_autopilot.set_target(target='0', mode=dr.AUTOPILOT_MODE.QUICK, worker_count=4)
models = project_autopilot.get_models()

# prediction
projects = get_projects_by_name('AE_x_re10')
project_autopilot = projects[0]
dataset = project_autopilot.upload_dataset(TEST_SET)
models = project_autopilot.get_models()
predict_job = models[0].request_predictions(dataset.id)
print(models[0])
predictions = predict_job.get_result_when_complete()

MSE_X = np.sum((np.array(predictions.iloc[:, 0]) - x_test_out)**2)/n_test
MSE_X
# with first 10: 0.19449225981392265
# without first 10: 0.18615741310322917
# 10000 epoch: 0.2184233473440311

# ============= #
# Residual data #
# ============= #

test_residual = np.array(predictions.iloc[:, 0]) - x_test_out
projects = get_projects_by_name('AE_x_re')
project_autopilot = projects[0]
dataset = project_autopilot.upload_dataset(TRAIN_SET)
models = project_autopilot.get_models()
predict_job = models[0].request_predictions(dataset.id)
predictions = predict_job.get_result_when_complete()
train_residual = np.array(predictions.iloc[:, 0]) - x_train_out

residual_train_z = np.concatenate((train_residual.reshape((n_train, 1)),
                                   z_train), axis=1)

residual_test_z = np.concatenate((test_residual.reshape((n_test, 1)),
                                  z_test), axis=1)

rtr_z = pd.DataFrame(residual_train_z)
rte_z = pd.DataFrame(residual_test_z)

rtr_z.to_excel('rtr_z.xlsx', index=False)
rte_z.to_excel('rte_z.xlsx', index=False)

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
print(models[0])
predictions = predict_job.get_result_when_complete()

MSE_Z = np.sum((np.array(predictions.iloc[:, 0]) - x_test_out)**2)/n_test
MSE_Z
# 0.5540405214213027
# 2.0267783581693655
# 10000 epoch: 0.5670585780308599

# ================ #
# residual Model z #
# ================ #

# create project
TRAIN_SET = '/Users/alex/Desktop/roar/test/rtr_z.xlsx'
TEST_SET = '/Users/alex/Desktop/roar/test/rte_z.xlsx'

project_autopilot = dr.Project.create(TRAIN_SET, project_name='AE_z_residual')
project_autopilot.set_target(target='0', mode=dr.AUTOPILOT_MODE.QUICK, worker_count=4)
models = project_autopilot.get_models()

# prediction
projects = get_projects_by_name('AE_z_residual')
project_autopilot = projects[0]
dataset = project_autopilot.upload_dataset(TEST_SET)
models = project_autopilot.get_models()
predict_job = models[0].request_predictions(dataset.id)
print(models[0])
predictions = predict_job.get_result_when_complete()

new_MSE = np.sum((np.array(predictions.iloc[:, 0]) + test_residual)**2)/n_test
new_MSE

# with first 10: 0.199
# without first 10: 0.18674056419626256
# 10000 epoch: 0.23460866458705085
