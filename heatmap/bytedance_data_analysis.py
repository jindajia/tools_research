import torch
import matplotlib.pyplot as plt
import numpy as np
from entropy import extract_bits_and_plot
def plot_heatmap_from_pth(pth_file_path):
    # Load the tensor data from the .pth file
    tensor_list = torch.load(pth_file_path, map_location=torch.device('cpu'))
    tensor_data = tensor_list[0]

    if tensor_data.dtype == torch.bfloat16:
        tensor_data = tensor_data.to(torch.float32)

    # Calculate Entropy
    print('Start')
    extract_bits_and_plot(
        tensor_data.numpy(), np.uint32,
        [23],
        [0xFF],
        256,
        False,
    )



    # print(tensor_data[:50])
    # tensor_data *= 1e6
    print(tensor_data.dtype, tensor_data.size(), tensor_data.shape)
    print(tensor_data[:50])

    # Check if the tensor is 1D and try to reshape to 2D if it is
    if len(tensor_data.shape) == 1:
        data_size = tensor_data.shape[0]
        side_length = int(np.sqrt(data_size))

        # Trim data to make it fit into a square 2D matrix
        trimmed_data_size = side_length * side_length
        tensor_data = tensor_data[:trimmed_data_size].reshape(side_length, side_length)

    # Scale the values by some large constant

    # Or apply logarithmic scaling (be cautious of taking log of zero)
    # tensor_data = np.log(tensor_data - tensor_data.min() + 1)

    # Or normalize the data
    # tensor_data = (tensor_data - tensor_data.min()) / (tensor_data.max() - tensor_data.min())

    # Create a heatmap using matplotlib
    plt.imshow(tensor_data, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Heatmap from .pt Tensor')
    plt.show()


# Use the function
plot_heatmap_from_pth('/Users/jindajia/Downloads/data/params/params_iter_001000/000.pt')
# plot_heatmap_from_pth('/Users/jindajia/Downloads/data/params/params_iter_001000/001.pt')
#
plot_heatmap_from_pth('/Users/jindajia/Downloads/data/grads/grads_iter_002000/000.pt')
# plot_heatmap_from_pth('/Users/jindajia/Downloads/data/grads/grads_iter_002000/001.pt')
