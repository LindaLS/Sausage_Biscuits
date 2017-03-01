import numpy as np

def process_data_autoencoder(original, window_size, stride_length):
    processed_stream = []
    for i in range(len(original)):
        mu = np.mean(original[i], axis=1)
        sigma = np.std(original[i], axis=1)
        normalized = (original[i] - np.transpose([mu])) / np.transpose([sigma])

        for j in range(len(original[i])):
            processed_stream.append(normalized[j])

    snippets = []
    for i in range(len(processed_stream)):
        upper_index = len(processed_stream[i])

        current_index = 0
        while current_index < (upper_index - window_size + 1):
            snippets.append(processed_stream[i][current_index : current_index+window_size])
            current_index = current_index + stride_length

    return np.array(snippets)
