import numpy as np
import matplotlib.pyplot as plt
import math


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

    # Compute entropy
    ent = calculate_entropy(freq)
    print(f"Entropy: {ent}")

def test():

    # Read binary float16 data from file
    with open("/Users/jindajia/Downloads/hurr_float16", "rb") as f:
        data = np.fromfile(f, dtype=np.float16)

    # Convert float16 data to uint16
    data_uint16 = np.frombuffer(data, dtype=np.uint16)

    # For 8 bits, just take the top 8 bits (after shifting down 8 positions)
    extract_bits_and_plot(data, np.uint16, [8], [0xFF], [256], False)

    # # For 5 bits, take bits 10 to 14 (after shifting down 10 positions)
    # extract_bits_and_plot(data, np.uint16, [10], [0x1F], [256], False)

    # For all 8 bits
    extract_bits_and_plot(
        data, np.uint32,
        [0, 8],
        [0xFF, 0xFF],
        256,
        False,
    )
