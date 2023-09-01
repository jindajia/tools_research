import torch
import matplotlib.pyplot as plt
import numpy as np
import math


def read_pt_to_tensor(pth_file_path):
    # Load the tensor data from the .pt file
    tensor_list = torch.load(pth_file_path, map_location=torch.device('cpu'))
    tensor_data = tensor_list[0]
    #
    # if tensor_data.dtype == torch.bfloat16:
    #     tensor_data = tensor_data.to(torch.float32)
    return tensor_data


def save_tensor_to_disk(tensor, savename):
    torch.save(tensor, savename)


def draw_tensor_histogram(tensor):
    plt.hist(tensor.numpy(), bins=128)
    plt.title("Tensor Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()


def calculate_entropy(freq):
    total = sum(freq)
    ent = 0.0
    for f in freq:
        if f > 0:
            p = f / total
            ent += p * math.log2(p)
    return -ent


def extract_bits_and_plot(data, dtype, shift_amounts, masks, num_possible_values, plot_histogram=False,
                          title='Histogram'):
    # Convert data to the corresponding uint dtype
    data_uint = np.frombuffer(data, dtype=dtype)

    extracted_bits_all = np.array([], dtype=int)

    for shift_amount, mask in zip(shift_amounts, masks):
        # Extract the bits
        extracted_bits = (data_uint >> shift_amount) & mask
        extracted_bits_all = np.concatenate((extracted_bits_all, extracted_bits))

    # Count frequencies of each possible value
    freq = np.zeros(num_possible_values, dtype=int)
    for val in extracted_bits_all:
        freq[val] += 1

    if plot_histogram:
        # Plot histogram
        plt.bar(range(num_possible_values), freq)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.show()

    return calculate_entropy(freq)
