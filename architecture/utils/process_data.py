import scipy.io as sio

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
				sample[0].append(data[0][i][0][j])
				sample[1].append(data[0][i][1][j])
				sample[2].append(data[0][i][2][j])
				sample[3].append(data[0][i][3][j])
			batch_x.append(sample)
			batch_y.append(action_map[curr_action])
			curr_sample_size = 0

		else:
			for j in range(data[0,i].shape[1]):
				sample[0].append(data[0][i][0][j])
				sample[1].append(data[0][i][1][j])
				sample[2].append(data[0][i][2][j])
				sample[3].append(data[0][i][3][j])
			curr_sample_size =  curr_sample_size + data[0,i].shape[1]
		skip = True

	return batch_y, batch_x



mat_contents = sio.loadmat('/home/linda/school/capstone/data/set2/Frad_thumb_index1.mat')

action_map = {}
action_map[0] = [0,0,0,0,0]
action_map[1] = [0,0,0,0,1]

data = mat_contents['EMGdata']

batch_y, batch_x = get_batch_from_raw_data(data, action_map)

