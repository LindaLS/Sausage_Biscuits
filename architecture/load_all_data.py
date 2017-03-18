import tensorflow as tf
import sys
import scipy.io as sio

data_path = ("../../")
from utils import *

def get_data(data_path, batch_size):

	index_action = {}
	middles_action = {}
	ring_action = {}
	pinky_action = {}
	thumb_action = {}

	sets_x = []
	sets_y = []

	# index_action[0] =   [0,0,0,0,0,0,0,0,0,0]
	# index_action[1] =   [0,1,0,0,0,0,0,0,0,0]
	index_action[2] =   [1,0,0,0,0]
	# middles_action[0] = [0,0,0,0,0,0,0,0,0,0]
	# middles_action[1] = [0,0,0,1,0,0,0,0,0,0]
	middles_action[2] = [0,1,0,0,0]
	# ring_action[0] =    [0,0,0,0,0,0,0,0,0,0]
	# ring_action[1] =    [0,0,0,0,0,1,0,0,0,0]
	ring_action[2] =    [0,0,1,0,0]
	# pinky_action[0] =   [0,0,0,0,0,0,0,0,0,0]
	# pinky_action[1] =   [0,0,0,0,0,0,0,1,0,0]
	pinky_action[2] =   [0,0,0,1,0]
	# thumb_action[0] =   [0,0,0,0,0,0,0,0,0,0]
	# thumb_action[1] =   [0,0,0,0,0,0,0,0,0,1]
	thumb_action[2] =   [0,0,0,0,1]

	
	n_index = 3
	index_contents = [[] for i in range(n_index)]
	# index_contents[0] = sio.loadmat(data_path + "data/Feb 17/david_index1.mat")['EMGdata']
	# index_contents[1] = sio.loadmat(data_path + "data/Feb 17/david_index2.mat")['EMGdata']
	# index_contents[2] = sio.loadmat(data_path + "data/Feb 17/david_index3.mat")['EMGdata']
	# index_contents[3] = sio.loadmat(data_path + "data/Feb 22/david_index1.mat")['EMGdata']
	# index_contents[4] = sio.loadmat(data_path + "data/Feb 22/david_index2.mat")['EMGdata']
	# index_contents[5] = sio.loadmat(data_path + "data/Feb 22/david_index3.mat")['EMGdata']
	index_contents[0] = sio.loadmat(data_path + "data/Feb 2/index1.mat")['EMGdata']
	index_contents[1] = sio.loadmat(data_path + "data/Feb 2/index2.mat")['EMGdata']
	index_contents[2] = sio.loadmat(data_path + "data/Feb 2/index3.mat")['EMGdata']
	# index_contents[] = sio.loadmat(data_path + "data/Feb 17/Linda_index1.mat")['EMGdata']
	# index_contents[] = sio.loadmat(data_path + "data/Feb 17/Linda_index2.mat")['EMGdata']
	# index_contents[] = sio.loadmat(data_path + "data/Feb 17/Linda_index3.mat")['EMGdata']
	for i in range(n_index):
		s = get_batch_shifted(index_contents[i], index_action, 5, [0,1])
		sets_x.append(s[0])
		sets_y.append(s[1])


	n_middle = 3
	middles_contents = [[] for i in range(n_middle)]
	# middles_contents[0] = sio.loadmat(data_path + "data/Feb 17/david_middle1.mat")['EMGdata']
	# middles_contents[1] = sio.loadmat(data_path + "data/Feb 17/david_middle2.mat")['EMGdata']
	# middles_contents[2] = sio.loadmat(data_path + "data/Feb 17/david_middle3.mat")['EMGdata']
	# middles_contents[3] = sio.loadmat(data_path + "data/Feb 22/david_middle1.mat")['EMGdata']
	# middles_contents[4] = sio.loadmat(data_path + "data/Feb 22/david_middle2.mat")['EMGdata']
	# middles_contents[5] = sio.loadmat(data_path + "data/Feb 22/david_middle3.mat")['EMGdata']
	middles_contents[0] = sio.loadmat(data_path + "data/Feb 2/mid1.mat")['EMGdata']
	middles_contents[1] = sio.loadmat(data_path + "data/Feb 2/mid2.mat")['EMGdata']
	middles_contents[2] = sio.loadmat(data_path + "data/Feb 2/mid3.mat")['EMGdata']
	# middles_contents[] = sio.loadmat(data_path + "data/Feb 17/Linda_middle1.mat")['EMGdata']
	# middles_contents[] = sio.loadmat(data_path + "data/Feb 17/Linda_middle2.mat")['EMGdata']
	# middles_contents[] = sio.loadmat(data_path + "data/Feb 17/Linda_middle3.mat")['EMGdata']
	for i in range(n_middle):
		s = get_batch_shifted(middles_contents[i], middles_action, 5, [0,1])
		sets_x.append(s[0])
		sets_y.append(s[1])

	n_ring = 3
	ring_contents = [[] for i in range(n_ring)]
	# ring_contents[0] = sio.loadmat(data_path + "data/Feb 17/david_ring1.mat")['EMGdata']
	# ring_contents[1] = sio.loadmat(data_path + "data/Feb 17/david_ring2.mat")['EMGdata']
	# ring_contents[2] = sio.loadmat(data_path + "data/Feb 17/david_ring3.mat")['EMGdata']
	# ring_contents[3] = sio.loadmat(data_path + "data/Feb 22/david_ring1.mat")['EMGdata']
	ring_contents[0] = sio.loadmat(data_path + "data/Feb 2/ring1.mat")['EMGdata']
	ring_contents[1] = sio.loadmat(data_path + "data/Feb 2/ring2.mat")['EMGdata']
	ring_contents[2] = sio.loadmat(data_path + "data/Feb 2/ring3.mat")['EMGdata']
	# ring_contents[] = sio.loadmat(data_path + "data/Feb 17/Linda_ring1.mat")['EMGdata']
	# ring_contents[] = sio.loadmat(data_path + "data/Feb 17/Linda_ring2.mat")['EMGdata']
	for i in range(n_ring):
		s = get_batch_shifted(ring_contents[i], ring_action, 5, [0,1])
		sets_x.append(s[0])
		sets_y.append(s[1])

	n_pinky = 3
	pinky_contents = [[] for i in range(n_pinky)]
	# pinky_contents[0] = sio.loadmat(data_path + "data/Feb 17/david_pinky1.mat")['EMGdata']
	# pinky_contents[1] = sio.loadmat(data_path + "data/Feb 17/david_pinky2.mat")['EMGdata']
	# pinky_contents[2] = sio.loadmat(data_path + "data/Feb 17/david_pinky3.mat")['EMGdata']
	# pinky_contents[4] = sio.loadmat(data_path + "data/Feb 22/david_pinky1.mat")['EMGdata']
	# pinky_contents[5] = sio.loadmat(data_path + "data/Feb 22/david_pinky2.mat")['EMGdata']
	# pinky_contents[6] = sio.loadmat(data_path + "data/Feb 22/david_pinky3.mat")['EMGdata']
	pinky_contents[0] = sio.loadmat(data_path + "data/Feb 2/pinky1.mat")['EMGdata']
	pinky_contents[1] = sio.loadmat(data_path + "data/Feb 2/pinky2.mat")['EMGdata']
	pinky_contents[2] = sio.loadmat(data_path + "data/Feb 2/pinky3.mat")['EMGdata']
	# pinky_contents[] = sio.loadmat(data_path + "data/Feb 17/Linda_pinky1.mat")['EMGdata']
	# pinky_contents[] = sio.loadmat(data_path + "data/Feb 17/Linda_pinky2.mat")['EMGdata']
	# pinky_contents[] = sio.loadmat(data_path + "data/Feb 17/Linda_pinky3.mat")['EMGdata']
	for i in range(n_pinky):
		s = get_batch_shifted(pinky_contents[i], pinky_action, 5, [0,1])
		sets_x.append(s[0])
		sets_y.append(s[1])

	n_thumb = 3
	thumb_contents = [[] for i in range(n_thumb)]
	# thumb_contents[0] = sio.loadmat(data_path + "data/Feb 17/david_thumb1.mat")['EMGdata']
	# thumb_contents[1] = sio.loadmat(data_path + "data/Feb 17/david_thumb2.mat")['EMGdata']
	# thumb_contents[2] = sio.loadmat(data_path + "data/Feb 17/david_thumb3.mat")['EMGdata']
	# thumb_contents[3] = sio.loadmat(data_path + "data/Feb 22/david_thumb1.mat")['EMGdata']
	# thumb_contents[4] = sio.loadmat(data_path + "data/Feb 22/david_thumb2.mat")['EMGdata']
	# thumb_contents[5] = sio.loadmat(data_path + "data/Feb 22/david_thumb3.mat")['EMGdata']
	thumb_contents[0] = sio.loadmat(data_path + "data/Feb 2/thumb1.mat")['EMGdata']
	thumb_contents[1] = sio.loadmat(data_path + "data/Feb 2/thumb2.mat")['EMGdata']
	thumb_contents[2] = sio.loadmat(data_path + "data/Feb 2/thumb3.mat")['EMGdata']
	# thumb_contents[] = sio.loadmat(data_path + "data/Feb 17/Linda_thumb1.mat")['EMGdata']
	# thumb_contents[] = sio.loadmat(data_path + "data/Feb 17/Linda_thumb2.mat")['EMGdata']
	# thumb_contents[] = sio.loadmat(data_path + "data/Feb 17/Linda_thumb3.mat")['EMGdata']
	for i in range(n_thumb):
		s = get_batch_shifted(thumb_contents[i], thumb_action, 5, [0,1])
		sets_x.append(s[0])
		sets_y.append(s[1])



	# index_contents[6] = sio.loadmat(data_path + "data/Feb 22/david_pinchindex1.mat")['EMGdata']
	# middles_contents[6] = sio.loadmat(data_path + "data/Feb 22/david_pinchmiddle1.mat")['EMGdata']
	# pinky_contents[3] = sio.loadmat(data_path + "data/Feb 22/david_pinchpinky1.mat")['EMGdata']

	return create_batches_and_test(sets_x, sets_y, batch_size, 5)
