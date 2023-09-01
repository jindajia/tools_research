'''
Usage:
python binary_analysis.py --filepath [Binary File Path] \
--savepath [Saved Image Path] \
--table_idx [The table index of embedding vectors] \
--iteration [The iteration of embedding vectors] \
--error_bound [The relative error bound] \
--shape [(batch_size, emb_len)] \

Example:
python binary_analysis.py --filepath /home/haofeng/dlrm/embs/table1_iter_1.bin --savepath . --table_idx 1 --iteration 1 --error_bound 0.01 --shape 1048576 16
'''

import numpy as np
import matplotlib.pyplot as plt
import argparse
import struct

parser = argparse.ArgumentParser()
parser.add_argument("--filepath", type=str, help="the path of binary file")
parser.add_argument("--savepath", type=str, help="the path of saved figures")
parser.add_argument("--table_idx", type=int, help="the index of embedding table")
parser.add_argument("--iteration", type=int, help="the number of iteration")
parser.add_argument("--error_bound", type=float, help="the error bound for pre-quantization and compression")
parser.add_argument("--shape", type=int, nargs=2,
                    help="the shape of the embedding vectors, for example (batch_size, emb_len)")
parser.add_argument("--print_frequency", action="store_true", default=False,
                    help="Print top20 values after quantization")


# Read the binary file as a numpy array
def read_binary_file(filename, data_type):
    return np.fromfile(filename, dtype=data_type)


def int8_to_bits(value):
    """Convert an int8 to its 8-bit representation."""
    return [(value >> i) & 1 for i in range(7, -1, -1)]

def float32_to_bits(value):
    """Convert a float32 to its 32-bit representation."""
    packed = struct.pack('f', value)
    integers = struct.unpack('I', packed)[0]
    bits = [0] * 32
    for i in range(32):
        bits[31 - i] = (integers >> i) & 1
    return bits

def float16_to_bits(value):
    """Convert a float16 to its 16-bit representation."""
    float16_hex = struct.unpack('H', struct.pack('e', value))[0]
    bits = [0] * 16
    for i in range(16):
        bits[15 - i] = (float16_hex >> i) & 1
    return bits

def float32_to_bfloat16_bits(value):
    packed = struct.pack('f', value)
    integers = struct.unpack('I', packed)[0]

    bfloat16_integers = integers >> 16

    bits = [0] * 16

    for i in range(16):
        bits[15 - i] = (bfloat16_integers >> i) & 1

    return bits

def bit_ratio(data, bits_function, bits_num):
    bit_count = [0] * bits_num
    total_count = len(data)
    for num in data:
        bits_num = bits_function(num)
        for i, bit in enumerate(bits_num):
            bit_count[i] += bit
    bit_ratio = [ count / total_count for count in bit_count]
    return bit_ratio

def partial_count_bit_ratio(start, end, data, bits_function, bits_num):
    bit_count = [0] * bits_num

    for i in range(start, end):
        bits = bits_function(data[i])
        for j, bit in enumerate(bits):
            bit_count[j] += bit

    return bit_count

def draw_bitmaps(data, table, iter, bits_function, bits, savepath):
    """
    Draw 16 figures from a (128, 16) numpy array.
    Each figure is of shape (128, 8) representing the bit values.
    """
    rows, cols = data.shape

    for col in range(cols):
        bitmaps = np.zeros((rows, bits), dtype=int)
        for row in range(rows):
            bitmaps[row, :] = bits_function(data[row, col])

        plt.figure()
        plt.imshow(bitmaps, cmap='gray', aspect='auto', interpolation='none')
        plt.colorbar()
        plt.title(f"Bit Representation for Column {col} table: {table} iter: {iter}")
        plt.xlabel("Bit Index")
        plt.ylabel("Row")
        plt.savefig(f"{savepath}/bit_table_{table}_iter_{iter}_{col}.png")
        plt.clf()


def quantization(original_arr, eb):
    eb = (original_arr.max() - original_arr.min()) * eb
    quantization_arr = np.round(original_arr * (1 / (eb * 2))).astype(np.int8)
    # quantization_arr.tofile(filename)
    return quantization_arr


# Draw histogram
def draw_histogram(data, table, iter, savepath):
    plt.hist(data, bins=128,  facecolor='blue', alpha=0.7)
    plt.title(f"Histogram_table_{table}_iter_{iter}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(f"{savepath}/hist_table_{table}_iter_{iter}.png")
    plt.clf()


def draw_heatmap(data, table, iter, savepath):
    plt.figure(figsize=(5, 8))
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f"Heatmap table_{table}_iter_{iter}")
    plt.savefig(f"{savepath}/heat_table_{table}_iter_{iter}.png")
    plt.clf()


def draw_values_with_grid(data, table, iter):
    """
    Draw the values of the data array in a grid.
    """
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)

    # Display grid
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)

    # Remove the major tick labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Display data values
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, str(data[i, j]), ha='center', va='center', color='black')

    plt.savefig(f"{savepath}/values_table_{table}_iter_{iter}.png")
    plt.clf()


# Print the top n most frequent integers without using Counter
def print_top_frequent(data, n=20):
    value_counts = {}
    for val in data:
        if val in value_counts:
            value_counts[val] += 1
        else:
            value_counts[val] = 1

    most_common = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:n]

    print(f"Top {n} most frequent integers:")
    for val, freq in most_common:
        print(f"Value: {val}, Frequency: {freq}")


if __name__ == "__main__":
    args = parser.parse_args()
    filepath = args.filepath
    savepath = args.savepath
    table = args.table_idx
    iter = args.iteration
    batch_size, emb_len = args.shape
    eb = args.error_bound
    data = np.fromfile(filepath, dtype=np.float32)
    # data after quantization
    data = quantization(data, eb)
    draw_histogram(data, table, iter, savepath)
    subset = data[:128 * emb_len]  # Select the first 128 rows
    subset = subset.reshape(128, emb_len)
    draw_bitmaps(subset, table, iter, savepath)
    draw_heatmap(subset, table, iter)
    subset = data[:8 * emb_len]
    subset = subset.reshape(8, emb_len)
    draw_values_with_grid(subset, table, iter)
    if args.print_frequency:
        print_top_frequent(data, 20)