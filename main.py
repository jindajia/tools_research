
from tool_function import *
from binary_analysis import *
from concurrent.futures import ThreadPoolExecutor

abs_path = '/Users/jindajia/Downloads/data'
params_list = ['params_iter_001000', 'params_iter_002000']
grads_list = ['grads_iter_001000', 'grads_iter_002000']
rank_list = ['000.pt', '001.pt']
rank_list_withoutSuffix = ['000', '001']

def draw_bitmap_heatmap_histogram():
    # Calculate Entropy

    for table, parm in enumerate(params_list):
        for iter, rank in enumerate(rank_list):
            path = abs_path + '/params/' + parm + '/' + rank
            print(path)
            tensor = read_pt_to_tensor(path)
            data = tensor.to(torch.float32).numpy()
            # data = quantization(data, 0.01)
            savepath = '/Users/jindajia/PycharmProjects/tools_research/fig'
            draw_histogram(data, parm, rank, savepath)
            emb_len = 16
            subset = data[:512 * emb_len]  # Select the first 128 rows
            subset = subset.reshape(512, emb_len)
            draw_bitmaps(subset, parm, rank, float32_to_bfloat16_bits, 16, savepath)
            # draw_heatmap(subset, parm, rank, savepath)


    for grad in grads_list:
        for rank in rank_list:
            path = abs_path + '/grads/' + grad + '/' + rank
            print(path)
            tensor = read_pt_to_tensor(path)
            data = tensor.numpy()
            # data = quantization(data, 0.01)
            savepath = '/Users/jindajia/PycharmProjects/tools_research/fig'
            draw_histogram(data, grad, rank, savepath)
            emb_len = 16
            subset = data[:512 * emb_len]  # Select the first 128 rows
            subset = subset.reshape(512, emb_len)
            draw_bitmaps(subset, grad, rank, float32_to_bits, 32, savepath)
            # draw_heatmap(subset, grad, rank, savepath)
            # print(grad, rank, 'ANS Entropy',
            #       extract_bits_and_plot(
            #           tensor.numpy(), np.uint32,
            #           [0, 8, 16, 24],
            #           [0xFF, 0xFF, 0xFF, 0xFF],
            #           256,
            #           False,
            #       )
            #       )

def cal_bit_ratio(num_threads=8):
    # for table, parm in enumerate(params_list):
    #     for iter, rank in enumerate(rank_list):
    #         path = abs_path + '/params/' + parm + '/' + rank
    #         print(path)
    #         tensor = read_pt_to_tensor(path)
    #         data_set = tensor.to(torch.float32).numpy()
    #         total_count = len(data_set)
    #         step = total_count // num_threads
    #         bits_num = 16
    #         bit_count = [0] * bits_num
    #         with ThreadPoolExecutor(max_workers=num_threads) as executor:
    #             futures = [executor.submit(partial_count_bit_ratio, i * step, (i + 1) * step, data_set, float32_to_bfloat16_bits, bits_num) for i in
    #                        range(num_threads)]
    #
    #             for future in futures:
    #                 partial_count = future.result()
    #                 for i in range(bits_num):
    #                     bit_count[i] += partial_count[i]
    #
    #         bit_ratio_arr = [count / total_count for count in bit_count]
    #         print(parm, rank, bit_ratio_arr)

    for grad in grads_list:
        for rank in rank_list:
            # path = abs_path + '/grads/' + grad + '/' + rank
            # print(path)
            # tensor = read_pt_to_tensor(path)
            # data = tensor.numpy()
            # print(grad, rank, bit_ratio(data, float32_to_bits, 32))
            path = abs_path + '/grads/' + grad + '/' + rank
            print(path)
            tensor = read_pt_to_tensor(path)
            data_set = tensor.numpy()
            total_count = len(data_set)
            step = total_count // num_threads
            bits_num = 32
            bit_count = [0] * bits_num
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(partial_count_bit_ratio, i * step, (i + 1) * step, data_set, float32_to_bits, bits_num) for i in
                           range(num_threads)]

                for future in futures:
                    partial_count = future.result()
                    for i in range(bits_num):
                        bit_count[i] += partial_count[i]

            bit_ratio_arr = [count / total_count for count in bit_count]
            print(grad, rank, bit_ratio_arr)

def save_tensor():
    for grad in grads_list:
        for rank in rank_list_withoutSuffix:
            path = abs_path + '/grads/' + grad + '/' + rank + '.pt'
            tensor = read_pt_to_tensor(path)
            np_data = tensor.to(torch.float32).numpy()
            tensor = torch.from_numpy(np_data)
            print(tensor.dtype)
            save_tensor_to_disk(tensor, 'data_bin/' + grad+rank+'.bin')
    for table, parm in enumerate(params_list):
        for iter, rank in enumerate(rank_list_withoutSuffix):
            path = abs_path + '/params/' + parm + '/' + rank + '.pt'
            tensor = read_pt_to_tensor(path)
            np_data = tensor.to(torch.float32).numpy()
            tensor = torch.from_numpy(np_data)
            tensor = tensor.to(torch.bfloat16)
            print(tensor.dtype)
            save_tensor_to_disk(tensor, 'data_bin/' + parm+rank+'.bin')
def main():
    # draw_bitmap_heatmap_histogram()
    # cal_bit_ratio(8)
    save_tensor()

if __name__ == '__main__':
    main()

