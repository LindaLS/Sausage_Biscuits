import numpy as np

def get_emg_data(raw_data):
    data = raw_data['EMGdata']
    _emg_data = []
    _output_data = []

    for i in range(0,len(data[0])):
        _emg_data.append(data[0][i][0])
        _output_data.append(data[1][i][0][0])

    emg_data = np.array(_emg_data).transpose()
    output_data = np.array(_output_data).transpose()

    return emg_data, output_data