import scipy.io as sio
import numpy as np

# we're sampling at 500hz
n_points_per_sample = 250

def get_batch_from_raw_data (data, action_map):
	batch_x = []
	batch_y = []
	skip = False
	sample = [[],[],[],[]]
	curr_sample_size = 0
	curr_action = 0

	for i in range(data.shape[1]):

		if data[0,i].shape[1] == 1:
			skip = False
			continue
		elif skip == True:
			continue


		if curr_action != data[1][i][0][0]:
			# a different sample, clear everything
			# don't store the first one, it could be contaminated
			sample = [[],[],[],[]]
			curr_sample_size = 0
			curr_action = data[1][i][0][0]
		elif data[0,i].shape[1] + curr_sample_size  >= n_points_per_sample:
			for j in range(n_points_per_sample - curr_sample_size):
				sample[0].append(float(data[0][i][0][j]) - 3500)
				sample[1].append(float(data[0][i][1][j]) - 3500)
				sample[2].append(float(data[0][i][2][j]) - 3500)
				sample[3].append(float(data[0][i][3][j]) - 3500)

			data_set = sample[0] + sample[1] + sample[2] + sample[3]
			batch_x.append(np.asarray(data_set))
			batch_y.append(np.asarray(action_map[curr_action]))
			curr_sample_size = 0
			sample = [[],[],[],[]]

		else:
			for j in range(data[0,i].shape[1]):
				sample[0].append(float(data[0][i][0][j]) - 3500)
				sample[1].append(float(data[0][i][1][j]) - 3500)
				sample[2].append(float(data[0][i][2][j]) - 3500)
				sample[3].append(float(data[0][i][3][j]) - 3500)
			curr_sample_size =  curr_sample_size + data[0,i].shape[1]
		skip = True

	return np.asarray(batch_y), np.asarray(batch_x)



