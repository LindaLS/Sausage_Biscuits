import scipy.io as sio
import numpy as np

# we're sampling at 500hz
n_points_per_sample = 250

def get_batch_from_raw_data (data, action_map, inaction):
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

		if data[1][i][0][0] in inaction:
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


def get_batch_from_raw_data_normalized (data, action_map, inaction):
	batch_x = []
	batch_y = []
	skip = False
	sample = [[],[],[],[]]
	curr_sample_size = 0
	curr_action = 0

	inaction_sum = [0,0,0,0]
	inaction_average_sum = [0,0,0,0]
	n_averages = 0

	for i in range(data.shape[1]):
		if data[1][i][0][0] in inaction:
			for j in range(data[0,i].shape[1]):
				inaction_sum[0] = inaction_sum[0] + abs((float(data[0][i][0][j])) - 3500)
				inaction_sum[1] = inaction_sum[1] + abs((float(data[0][i][1][j])) - 3500)
				inaction_sum[2] = inaction_sum[2] + abs((float(data[0][i][2][j])) - 3500)
				inaction_sum[3] = inaction_sum[3] + abs((float(data[0][i][3][j])) - 3500)

			inaction_average_sum[0] += inaction_sum[0]/data[0,i].shape[1]
			inaction_average_sum[1] += inaction_sum[1]/data[0,i].shape[1]
			inaction_average_sum[2] += inaction_sum[2]/data[0,i].shape[1]
			inaction_average_sum[3] += inaction_sum[3]/data[0,i].shape[1]
			n_averages = n_averages + 1


	inaction_average_sum[0] = inaction_average_sum[0]/n_averages
	inaction_average_sum[1] = inaction_average_sum[1]/n_averages
	inaction_average_sum[2] = inaction_average_sum[2]/n_averages
	inaction_average_sum[3] = inaction_average_sum[3]/n_averages


	for i in range(data.shape[1]):

		if data[0,i].shape[1] == 1:
			skip = False
			continue
		elif skip == True:
			continue

		if data[1][i][0][0] in inaction:
			continue


		if curr_action != data[1][i][0][0]:
			# a different sample, clear everything
			# don't store the first one, it could be contaminated
			sample = [[],[],[],[]]
			curr_sample_size = 0
			curr_action = data[1][i][0][0]
		elif data[0,i].shape[1] + curr_sample_size  >= n_points_per_sample:
			for j in range(n_points_per_sample - curr_sample_size):
				sample[0].append( abs(float(data[0][i][0][j]) - 3500) - inaction_average_sum[0] )
				sample[1].append( abs(float(data[0][i][1][j]) - 3500) - inaction_average_sum[1] )
				sample[2].append( abs(float(data[0][i][2][j]) - 3500) - inaction_average_sum[2] )
				sample[3].append( abs(float(data[0][i][3][j]) - 3500) - inaction_average_sum[3] )

			data_set = sample[0] + sample[1] + sample[2] + sample[3]
			batch_x.append(np.asarray(data_set))
			batch_y.append(np.asarray(action_map[curr_action]))
			curr_sample_size = 0
			sample = [[],[],[],[]]

		else:
			for j in range(data[0,i].shape[1]):
				sample[0].append( abs(float(data[0][i][0][j]) - 3500) - inaction_average_sum[0] )
				sample[1].append( abs(float(data[0][i][1][j]) - 3500) - inaction_average_sum[1] )
				sample[2].append( abs(float(data[0][i][2][j]) - 3500) - inaction_average_sum[2] )
				sample[3].append( abs(float(data[0][i][3][j]) - 3500) - inaction_average_sum[3] )
			curr_sample_size =  curr_sample_size + data[0,i].shape[1]
		skip = True

	return np.asarray(batch_y), np.asarray(batch_x)


def get_batch_from_raw_data_new_format (data, action_map, inaction):

	n_points = 250
	batch_x = []
	batch_y = []
	sample = [[],[],[],[]]
	curr_sample_size = 0
	curr_action = 0

	inaction_sum = [0,0,0,0]
	n_averages = 0

	# for i in range(data.shape[1]):
	# 	if data[1][i][0][0] in inaction:
	# 		inaction_sum[0] = inaction_sum[0] + abs((float(data[0][i][0][0])) - 3500)
	# 		inaction_sum[1] = inaction_sum[1] + abs((float(data[0][i][0][1])) - 3500)
	# 		inaction_sum[2] = inaction_sum[2] + abs((float(data[0][i][0][2])) - 3500)
	# 		inaction_sum[3] = inaction_sum[3] + abs((float(data[0][i][0][3])) - 3500)

	# 		n_averages = n_averages + 1


	# inaction_sum[0] = inaction_sum[0]/n_averages
	# inaction_sum[1] = inaction_sum[1]/n_averages
	# inaction_sum[2] = inaction_sum[2]/n_averages
	# inaction_sum[3] = inaction_sum[3]/n_averages


	for i in range(data.shape[1]):

		if data[1][i][0][0] in inaction:
			continue


		if curr_action != data[1][i][0][0]:
			# a different sample, clear everything
			# don't store the first one, it could be contaminated
			sample = [[],[],[],[]]
			curr_sample_size = 0
			curr_action = data[1][i][0][0]
		else:
			sample[0].append((float(data[0][i][0][0])))
			sample[1].append((float(data[0][i][0][1])))
			sample[2].append((float(data[0][i][0][2])))
			sample[3].append((float(data[0][i][0][3])))
			curr_sample_size =  curr_sample_size + 1

			if (curr_sample_size == n_points):
				data_set = sample[0] + sample[1] + sample[2] + sample[3]
				batch_x.append(np.asarray(data_set))
				batch_y.append(np.asarray(action_map[curr_action]))
				curr_sample_size = 0
				sample = [[],[],[],[]]

	return batch_y, batch_x



