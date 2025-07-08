# 这是附加题模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现triton的深度学习推理过程，请严格保持输出格式输出
import os
import h5py
import time
import numpy as np
import torch
import triton
import triton.language as tl


def read_params(dir):
    # 列出所有txt文件
    files = [f for f in os.listdir(dir) if f.endswith('.txt')]
    params = {}
    for fileName in files:
        data = []
        with open(os.path.join(dir,fileName), 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                value = float(line)
                data.append(value)
        modelName = fileName.replace(".txt","")
        params[modelName] = data
    return params

def read_h5_file(dataPath):
    list_of_points = []
    list_of_labels = []
    with h5py.File(dataPath,"r") as hf:
        for k in hf.keys():
            # list_of_points.append(hf[k]["points"][:].astype(np.float32)) #每个points是（N,3）的二维数组ndarray
            list_of_points.append(hf[k]["points"][:].astype(np.float32).flatten()) #每个points是N*3的一维ndarray
            list_of_labels.append(hf[k].attrs["label"])
    return list_of_points,list_of_labels

device_64_22500 = torch.zeros(64*22500, dtype=torch.float32).to(device='cuda')
device_64_22500_2 = torch.zeros(64*22500, dtype=torch.float32).to(device='cuda')
device_22500_64_copy = torch.zeros(22500*64, dtype=torch.float32).to(device='cuda')
device_128_22500 = torch.zeros(128*22500, dtype=torch.float32).to(device='cuda')
device_1024_22500 = torch.zeros(1024*22500, dtype=torch.float32).to(device='cuda')
device_1024_1 = torch.zeros(1024*1, dtype=torch.float32).to(device='cuda')
device_512_1 = torch.zeros(512*1, dtype=torch.float32).to(device='cuda')
device_256_1 = torch.zeros(256*1, dtype=torch.float32).to(device='cuda')
device_9_1 = torch.zeros(9*1, dtype=torch.float32).to(device='cuda')
device_3_22500 = torch.zeros(3*22500, dtype=torch.float32).to(device='cuda')
device_64_64 = torch.zeros(64*64, dtype=torch.float32).to(device='cuda')
device_4096_1 = torch.zeros(4096*1, dtype=torch.float32).to(device='cuda')
device_10_1 = torch.zeros(10*1, dtype=torch.float32).to(device='cuda')
device_1000_10_1 = torch.zeros(1000*10*1, dtype=torch.float32).to(device='cuda')

# 定义Triton内核函数
@triton.jit
def conv1d_norm_relu(output, a_row, a_col, b_col, weight, input_array, bias, bn_weight, bn_bias, bn_running_mean, bn_running_var,
                     BLOCK_SIZE: tl.constexpr = 1024, BIG_BLOCK_SIZE: tl.constexpr = 16):
    idx = tl.program_id(0)
    idy = tl.program_id(1)

    # 检查线程是否在有效范围内
    if idx < a_row and idy < b_col:
        sum = 0.0
        col_array = tl.arange(0, BLOCK_SIZE)
        mask = col_array < a_col
        sum = tl.sum(tl.load(weight + (idx * a_col + col_array), mask) * tl.load(input_array + (col_array * b_col + idy), mask))
        sum += tl.load(bias + idx)
        x_mean = tl.load(bn_running_mean + idx)
        x_var = tl.load(bn_running_var + idx)
        bn1_w_val = tl.load(bn_weight + idx)
        bn1_b_val = tl.load(bn_bias + idx)
        norm_res = (sum - x_mean) / tl.sqrt(x_var + 1e-5)
        ans = norm_res * bn1_w_val + bn1_b_val
        ans = tl.maximum(ans, 0.0)
        tl.store(output + (idx * b_col + idy), ans)

# import triton.language as tl

@triton.jit
def conv1d_norm_relu_1(
    output, a_row, a_col, b_col, weight, input_array, bias, 
    bn_weight, bn_bias, bn_running_mean, bn_running_var,
    BLOCK_SIZE: tl.constexpr = 1024, BIG_BLOCK_SIZE: tl.constexpr = 16
):
    idx = tl.program_id(0)  # 当前线程块的行索引
    blocky = tl.program_id(1)  # 当前线程块的列块索引

    # 计算当前线程块的输出起始列
    idy = blocky * BIG_BLOCK_SIZE
    range_y = idy + tl.arange(0, BIG_BLOCK_SIZE)
    mask_y = range_y < b_col

    if idx < a_row:
        col_array = tl.arange(0, BLOCK_SIZE)
        mask = col_array < a_col

        # 遍历 BIG_BLOCK_SIZE 范围内的列
        for i in range(BIG_BLOCK_SIZE):
            current_y = idy + i
            if current_y < b_col:
                # 计算当前元素的卷积求和
                sum_val = tl.sum(
                    tl.load(weight + (idx * a_col + col_array), mask) *
                    tl.load(input_array + (col_array * b_col + current_y), mask)
                )
                # 加上偏置
                sum_val += tl.load(bias + idx)

                # 批归一化
                x_mean = tl.load(bn_running_mean + idx)
                x_var = tl.load(bn_running_var + idx)
                bn1_w_val = tl.load(bn_weight + idx)
                bn1_b_val = tl.load(bn_bias + idx)
                norm_res = (sum_val - x_mean) / tl.sqrt(x_var + 1e-5)
                
                # 批归一化后的值
                sum_val = norm_res * bn1_w_val + bn1_b_val
                
                # ReLU 激活
                sum_val = tl.maximum(sum_val, 0.0)

                # 将结果直接写入输出
                tl.store(output + (idx * b_col + current_y), sum_val)




@triton.jit
def conv1d_norm_relu_2(output, a_row, a_col, b_col, weight, input_array, bias, bn_weight, bn_bias, bn_running_mean, bn_running_var,
                       BLOCK_SIZE: tl.constexpr = 1024, BIG_BLOCK_SIZE: tl.constexpr = 16):
    
    block_idx = tl.program_id(0)
    block_idy = tl.program_id(1)

    block_idx_start = block_idx * BIG_BLOCK_SIZE
    block_idy_start = block_idy * BIG_BLOCK_SIZE

    # Compute indices for the current block
    idx_array = tl.arange(0, BIG_BLOCK_SIZE) + block_idx_start
    idy_array = tl.arange(0, BIG_BLOCK_SIZE) + block_idy_start

    # Apply masks to limit indices to tensor dimensions
    idx_mask = idx_array < a_row
    idy_mask = idy_array < b_col

    # Initialize accumulator with shape (BIG_BLOCK_SIZE, BIG_BLOCK_SIZE)
    sum = tl.zeros((BIG_BLOCK_SIZE, BIG_BLOCK_SIZE), dtype=tl.float32)

    # Column array for computation
    col_array = tl.arange(0, BLOCK_SIZE)
    mask = col_array < a_col

    # Perform 1D convolution, ensure shapes align correctly
    weight_vals = tl.load(weight + idx_array[:, None] * a_col + col_array, mask=mask[None, :] & idx_mask[:, None])
    input_vals = tl.load(input_array + col_array[:, None] * b_col + idy_array, mask=mask[:, None] & idy_mask[None, :])
    sum += tl.sum(weight_vals * input_vals, axis=1)

    # Add bias
    sum += tl.load(bias + idx_array, mask=idx_mask)

    # Batch normalization parameters
    x_mean = tl.load(bn_running_mean + idx_array, mask=idx_mask)
    x_var = tl.load(bn_running_var + idx_array, mask=idx_mask)
    bn1_w_val = tl.load(bn_weight + idx_array, mask=idx_mask)
    bn1_b_val = tl.load(bn_bias + idx_array, mask=idx_mask)

    # Batch normalization
    norm_res = (sum - x_mean) / tl.sqrt(x_var + 1e-5)

    # Apply BN weights and biases
    ans = norm_res * bn1_w_val + bn1_b_val

    # ReLU activation
    ans = tl.maximum(ans, 0.0)

    # Store results in the output tensor
    tl.store(output + idx_array[:, None] * b_col + idy_array, ans, mask=idx_mask[:, None] & idy_mask[None, :])


@triton.jit
def max_matrix(output,input, row, col, BLOCK_SIZE: tl.constexpr = 22500):
# # #     @triton.jit
# def max_matrix(output, input_tensor, a_row, a_col, BLOCK_SIZE: tl.constexpr = 1024):
#     row_idx = tl.program_id(0)  # 当前行索引
#     row_start = row_idx * a_col
#     idx = tl.arange(0, BLOCK_SIZE)
#     mask = idx < a_col
#     vals_1 = tl.load(input_tensor + row_start + idx, mask=mask)
#     vals_2 = tl.load(input_tensor + row_start + idx + BLOCK_SIZE, mask=mask)

#     # 进行块内最大值归约
#     for stride in range(2, (a_col + BLOCK_SIZE - 1) // BLOCK_SIZE):
#         vals_1 = tl.maximum(vals_1, vals_2)
#         vals_2 = tl.load(input_tensor + row_start + idx + stride * BLOCK_SIZE, mask=mask)

#     # 最终最大值
#     max_val = vals_1[0]
#     for i in range(1, BLOCK_SIZE):
#         max_val = tl.max(max_val, vals_1[i])

#     # 将每行的最大值存储到输出张量中
#     tl.store(output + row_idx, max_val)

    # max_vals = tl.max(input, axis=1)
    
    # # 存储结果
    # tl.store(output, max_vals)

    id = tl.program_id(0)
    # array = tl.arange(0, BLOCK_SIZE)
    # mask = array < col
    # max_val = tl.load(input + id*col)
    if (id<row):
        # array = tl.arange(0, BLOCK_SIZE)
        # mask = array < col
        # if (id<row):
        #     max_val = tl.load(input + id*col)
        #     max_val = tl.max(max_val, tl.load(input + id*col+array, mask))
        max_val = tl.load(input + id*col)
        for i in range(1, col):
            max_val = tl.maximum(max_val, tl.load(input + id*col +i))
        # array = tl.arange(0, 1024)
        # max_val = tl.maximum(input + id*col + array)
        tl.store(output + id, max_val)

@triton.jit
def conv1d_9(output,  weight, input, bias):
    idx = tl.program_id(0)
    if (idx<9):
        sum = 0.0
        array_add = tl.arange(0, 256)
        mask = array_add < 256
        sum = tl.sum(tl.load(weight + idx*256 + array_add, mask) * tl.load(input + array_add, mask))
        sum += tl.load(bias + idx)
        if (idx==0 or idx==4) or idx==8:
            sum += 1.0
        tl.store(output + idx,sum)

@triton.jit
def matrix_array_9(output, a,  b,  a_row, a_col,  b_col,  BLOCK_SIZE: tl.constexpr = 4):
    idx = tl.program_id(0)
    idy = tl.program_id(1)
    if (idx<a_row and idy<b_col):
        id_T = idy*a_row+idx
        sum = 0.0
        array_mul = tl.arange(0, BLOCK_SIZE)
        mask = array_mul < a_col
        sum = tl.sum(tl.load(a + idx*a_col + array_mul, mask) * tl.load(b + array_mul*b_col + idy, mask))
        tl.store(output + id_T, sum)

@triton.jit
def conv1d_norm_relu_copy_T(output,  output_T,  a_row,  a_col,  b_col,  weight, input, bias, bn_weight,bn_bias,  bn_running_mean, bn_running_var,
                            BLOCK_SIZE: tl.constexpr = 4):
    idx = tl.program_id(0)
    idy = tl.program_id(1)
    if (idx<a_row and idy<b_col):
        id = idx*b_col+idy
        id_T = idy*a_row+idx
        sum = 0.0
        array_mul = tl.arange(0, BLOCK_SIZE)
        mask = array_mul < a_col
        sum = tl.sum(tl.load(weight + idx*a_col + array_mul, mask) * tl.load(input + array_mul*b_col + idy, mask))
        sum += tl.load(bias + idx)
        x_mean = tl.load(bn_running_mean + idx)
        x_var = tl.load(bn_running_var + idx)
        bn1_w_val = tl.load(bn_weight + idx)
        bn1_b_val = tl.load(bn_bias + idx)
        norm_res = (sum - x_mean) / tl.sqrt(x_var + 1e-5)
        ans = norm_res * bn1_w_val + bn1_b_val
        ans = tl.maximum(ans, 0.0)
        tl.store(output + id, ans)
        tl.store(output_T + id_T, ans)


@triton.jit
def conv1d_4096_add_64_64(output,   a_row,  a_col,  b_col,  weight, input,  bias, 
                          BLOCK_SIZE: tl.constexpr = 1024):
    idx = tl.program_id(0)
    if (idx<a_row):
        sum = 0.0
        array_mul = tl.arange(0, BLOCK_SIZE)
        mask = array_mul < a_col
        sum = tl.sum(tl.load(weight + idx*a_col + array_mul, mask) * tl.load(input + array_mul, mask))
        sum += tl.load(bias + idx)
        current_row = idx // 64
        if (idx == current_row + current_row * 64):
            sum += 1.0
        tl.store(output + idx, sum)

@triton.jit
def conv1d_22500_64_64_trans_result( output,  a_row,  a_col,  b_col, input_a, input_b,
                                      BLOCK_SIZE: tl.constexpr = 1024):
    idx = tl.program_id(0)
    idy = tl.program_id(1)
    if (idx<a_row and idy<a_col):
        sum = 0.0
        array_mul = tl.arange(0, BLOCK_SIZE)
        mask = array_mul < b_col
        sum = tl.sum(tl.load(input_a + idx*a_col + array_mul, mask) * tl.load(input_b + array_mul*b_col + idy, mask))
        tl.store(output + idy*a_row+idx, sum)

@triton.jit
def conv1d_log_softmax_ans(output,  a_row,  a_col,  weight,  input,  bias,
                           BLOCK_SIZE: tl.constexpr = 1024):
    idx = tl.program_id(0)
    if (idx<a_row):
        num = 0.0
        array_mul = tl.arange(0, BLOCK_SIZE)
        mask = array_mul < a_col
        num = tl.sum(tl.load(weight + idx*a_col + array_mul, mask) * tl.load(input + array_mul, mask))
        num += tl.load(bias + idx)
        tl.store(output + idx, num)

def get_max_index (result):
    max_index = 0
    max_value = result[0]
    for i in range(1, 10):
        current_value = result[i]
        if (max_value < current_value):
            max_value = current_value
            max_index = i
    return max_index

from tqdm import tqdm
def do_inference(list_of_points,list_of_labels,params): #请在本函数下使用triton实现推理操作
    correct_count = 0
    for i in tqdm(range(list_of_points.shape[0])):
        # if i>10:
        #     break
        device_points = torch.from_numpy(list_of_points[i]).reshape(-1).to(device='cuda', dtype=torch.float32).contiguous()
        # output_ptr = torch.zeros(64*22500, dtype=torch.float32).to(device='cuda')
        device_points_T = torch.from_numpy(list_of_points[i].T).reshape(-1).to(device='cuda', dtype=torch.float32).contiguous()  
        # print(f"device_points_T {device_points_T.shape}")
        # 调用Triton内核函数
        # conv1d_norm_relu_2[4, 1407](
        BIG_BLOCK_SIZE = 16
        # conv1d_norm_relu_1[64, (22500 + BIG_BLOCK_SIZE -1 ) // BIG_BLOCK_SIZE](
        conv1d_norm_relu[64, 22500](
            device_64_22500, 64, 3, 22500, 
            # torch.from_numpy(np.array(params["feat.stn.conv1.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
            params["feat.stn.conv1.weight"], 
            device_points_T, 
            # torch.from_numpy(np.array(params["feat.stn.conv1.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
            params["feat.stn.conv1.bias"], 
#"feat.stn.bn1.weight","feat.stn.bn1.bias","feat.stn.bn1.running_mean","feat.stn.bn1.running_var"
            # torch.from_numpy(np.array(params["feat.stn.bn1.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
            params["feat.stn.bn1.weight"], 
            params["feat.stn.bn1.bias"], 
            params["feat.stn.bn1.running_mean"], 
            params["feat.stn.bn1.running_var"],
            # num_warps=1,
            # torch.from_numpy(np.array(params["feat.stn.bn1.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
            # torch.from_numpy(np.array(params["feat.stn.bn1.running_mean"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
            # torch.from_numpy(np.array(params["feat.stn.bn1.running_var"], dtype=np.float32)).reshape(-1).to(device='cuda'),
            # num_warps=64,
            # BLOCK_SIZE=32
            )
        # break
        # 打印输出结果，仅用于调试！
        # print(f"output_ptr {device_64_22500.data[22400:22410]}")
        # print(f"output_ptr {device_64_22500.data[0:10]}")
        # break
        # continue
        # last_result = output_ptr
        # current_result = torch.zeros(128*22500, dtype=torch.float32).to(device='cuda')
        conv1d_norm_relu[128, 22500](
            device_128_22500, 128, 64, 22500, 
            params["feat.stn.conv2.weight"], 
            # torch.from_numpy(np.array(params["feat.stn.conv2.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
            device_64_22500, 
            params["feat.stn.conv2.bias"], 
            params["feat.stn.bn2.weight"], 
            params["feat.stn.bn2.bias"], 
            params["feat.stn.bn2.running_mean"], 
            params["feat.stn.bn2.running_var"],
#             torch.from_numpy(np.array(params["feat.stn.conv2.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
# #"feat.stn.bn2.weight","feat.stn.bn2.bias","feat.stn.bn2.running_mean","feat.stn.bn2.running_var"
#             torch.from_numpy(np.array(params["feat.stn.bn2.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.stn.bn2.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.stn.bn2.running_mean"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.stn.bn2.running_var"], dtype=np.float32)).reshape(-1).to(device='cuda'),
            )
        # last_result = current_result
        # current_result = torch.zeros(1024*22500, dtype=torch.float32).to(device='cuda')
        conv1d_norm_relu[500, 22500](
            device_1024_22500, 1024, 128, 22500, 
            params["feat.stn.conv3.weight"], 
            # torch.from_numpy(np.array(params["feat.stn.conv3.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
            device_128_22500, 
            params["feat.stn.conv3.bias"], 
            params["feat.stn.bn3.weight"], 
            params["feat.stn.bn3.bias"], 
            params["feat.stn.bn3.running_mean"], 
            params["feat.stn.bn3.running_var"],
#             torch.from_numpy(np.array(params["feat.stn.conv3.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
# #"feat.stn.bn3.weight","feat.stn.bn3.bias","feat.stn.bn3.running_mean","feat.stn.bn3.running_var"
#             torch.from_numpy(np.array(params["feat.stn.bn3.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.stn.bn3.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.stn.bn3.running_mean"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.stn.bn3.running_var"], dtype=np.float32)).reshape(-1).to(device='cuda'),
            )
        # last_result = current_result
        # current_result = torch.zeros(1024*1, dtype=torch.float32).to(device='cuda')
        # max_matrix<<<1024,1>>>(device_1024_1, device_1024_22500, 1024, 22500);
        max_matrix[1024, 1](
            device_1024_1, device_1024_22500, 1024, 22500
            )
        # last_result = current_result
        # current_result = torch.zeros(512*1, dtype=torch.float32).to(device='cuda')
        conv1d_norm_relu[512, 1](
            device_512_1, 512, 1024, 1, 
# //"feat.stn.fc1.weight","feat.stn.fc1.bias"
            params["feat.stn.fc1.weight"], 
            # torch.from_numpy(np.array(params["feat.stn.fc1.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
            device_1024_1,
            params["feat.stn.fc1.bias"], 
            params["feat.stn.bn4.weight"], 
            params["feat.stn.bn4.bias"], 
            params["feat.stn.bn4.running_mean"], 
            params["feat.stn.bn4.running_var"],
#             torch.from_numpy(np.array(params["feat.stn.fc1.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
# #"feat.stn.bn4.weight","feat.stn.bn4.bias","feat.stn.bn4.running_mean","feat.stn.bn4.running_var"
#             torch.from_numpy(np.array(params["feat.stn.bn4.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.stn.bn4.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.stn.bn4.running_mean"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.stn.bn4.running_var"], dtype=np.float32)).reshape(-1).to(device='cuda'),
            )
        # last_result = current_result
        # current_result = torch.zeros(256*1, dtype=torch.float32).to(device='cuda')
        conv1d_norm_relu[256, 1](
            device_256_1, 256, 512, 1, 
# //"feat.stn.fc2.weight","feat.stn.fc2.bias"
            params["feat.stn.fc2.weight"], 
            # torch.from_numpy(np.array(params["feat.stn.fc2.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
            device_512_1,
            params["feat.stn.fc2.bias"], 
            params["feat.stn.bn5.weight"], 
            params["feat.stn.bn5.bias"], 
            params["feat.stn.bn5.running_mean"], 
            params["feat.stn.bn5.running_var"],
#             torch.from_numpy(np.array(params["feat.stn.fc2.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
# #"feat.stn.bn5.weight","feat.stn.bn5.bias","feat.stn.bn5.running_mean","feat.stn.bn5.running_var"
#             torch.from_numpy(np.array(params["feat.stn.bn5.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.stn.bn5.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.stn.bn5.running_mean"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.stn.bn5.running_var"], dtype=np.float32)).reshape(-1).to(device='cuda'),
            )
        # last_result = current_result
        # current_result = torch.zeros(9*1, dtype=torch.float32).to(device='cuda')
        # conv1d_9<<<9,1>>>(device_9_1, device_params["feat.stn.fc3.weight"], device_256_1, device_params["feat.stn.fc3.bias"]);
        conv1d_9[9, 1](
            device_9_1, 
            params["feat.stn.fc3.weight"], 
            # torch.from_numpy(np.array(params["feat.stn.fc3.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
            device_256_1,
            params["feat.stn.fc3.bias"], 
            # torch.from_numpy(np.array(params["feat.stn.fc3.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'),
            )
        # print(f"current_result {current_result.data[0:10]}")
        # last_result = current_result
        # current_result = torch.zeros(22500*3, dtype=torch.float32).to(device='cuda')
        # matrix_array_9<<<blockPerGrid_6,threadPerBlock_6>>>(device_3_22500 , device_array_1000_22500_3[i], device_9_1, 22500, 3, 3);
        matrix_array_9[22500,3](
            device_3_22500, 
            device_points, 
            device_9_1, 
            22500, 3, 3
            )
        # print(f"current_result {current_result.data[0:10]}")
        # last_result = current_result
        # current_result = torch.zeros(64*22500, dtype=torch.float32).to(device='cuda')
        # device_22500_64_copy = torch.zeros(22500*64, dtype=torch.float32).reshape(-1).to(device='cuda')
        conv1d_norm_relu_copy_T[64, 22500](
            device_64_22500, device_22500_64_copy, 64, 3, 22500, 
# //"feat.conv1.weight","feat.conv1.bias"
            params["feat.conv1.weight"], 
            # torch.from_numpy(np.array(params["feat.conv1.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
            device_3_22500,
            params["feat.conv1.bias"], 
            params["feat.bn1.weight"], 
            params["feat.bn1.bias"], 
            params["feat.bn1.running_mean"], 
            params["feat.bn1.running_var"],
#             torch.from_numpy(np.array(params["feat.conv1.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
# #"feat.bn1.weight","feat.bn1.bias","feat.bn1.running_mean","feat.bn1.running_var"
#             torch.from_numpy(np.array(params["feat.bn1.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.bn1.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.bn1.running_mean"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.bn1.running_var"], dtype=np.float32)).reshape(-1).to(device='cuda'),
            )
        # print(f"conv1d_norm_relu_copy_T current_result {current_result.data[0:10]}")
        # last_result = current_result
        # current_result = torch.zeros(64*22500, dtype=torch.float32).to(device='cuda')
        conv1d_norm_relu[64, 22500](
            device_64_22500_2, 
            64, 64, 22500, 
            # // "feat.fstn.conv1.weight","feat.fstn.conv1.bias"
            params["feat.fstn.conv1.weight"], 
            # torch.from_numpy(np.array(params["feat.fstn.conv1.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
            device_64_22500,
            params["feat.fstn.conv1.bias"], 
            params["feat.fstn.bn1.weight"], 
            params["feat.fstn.bn1.bias"], 
            params["feat.fstn.bn1.running_mean"], 
            params["feat.fstn.bn1.running_var"],
#             torch.from_numpy(np.array(params["feat.fstn.conv1.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
# #"feat.fstn.bn1.weight","feat.fstn.bn1.bias","feat.fstn.bn1.running_mean","feat.fstn.bn1.running_var"
#             torch.from_numpy(np.array(params["feat.fstn.bn1.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.fstn.bn1.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.fstn.bn1.running_mean"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.fstn.bn1.running_var"], dtype=np.float32)).reshape(-1).to(device='cuda'),
            )
        # last_result = current_result
        # current_result = torch.zeros(128*22500, dtype=torch.float32).to(device='cuda')
        conv1d_norm_relu[128, 22500](
            device_128_22500, 
            128, 64, 22500, 
# // "feat.fstn.conv2.weight","feat.fstn.conv2.bias"
            params["feat.fstn.conv2.weight"], 
            # torch.from_numpy(np.array(params["feat.fstn.conv2.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
            device_64_22500_2,
            params["feat.fstn.conv2.bias"], 
            params["feat.fstn.bn2.weight"], 
            params["feat.fstn.bn2.bias"], 
            params["feat.fstn.bn2.running_mean"], 
            params["feat.fstn.bn2.running_var"],
#             torch.from_numpy(np.array(params["feat.fstn.conv2.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
# #"feat.fstn.bn2.weight","feat.fstn.bn2.bias","feat.fstn.bn2.running_mean","feat.fstn.bn2.running_var"
#             torch.from_numpy(np.array(params["feat.fstn.bn2.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.fstn.bn2.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.fstn.bn2.running_mean"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.fstn.bn2.running_var"], dtype=np.float32)).reshape(-1).to(device='cuda'),
            )
        # last_result = current_result
        # current_result = torch.zeros(1024*22500, dtype=torch.float32).to(device='cuda')
        conv1d_norm_relu[500, 22500](
            device_1024_22500, 
            1024, 128, 22500, 
# // "feat.fstn.conv3.weight","feat.fstn.conv3.bias"
            params["feat.fstn.conv3.weight"], 
            # torch.from_numpy(np.array(params["feat.fstn.conv3.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
            device_128_22500,
            params["feat.fstn.conv3.bias"], 
            params["feat.fstn.bn3.weight"], 
            params["feat.fstn.bn3.bias"], 
            params["feat.fstn.bn3.running_mean"], 
            params["feat.fstn.bn3.running_var"],
#             torch.from_numpy(np.array(params["feat.fstn.conv3.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
# #"feat.fstn.bn3.weight","feat.fstn.bn3.bias","feat.fstn.bn3.running_mean","feat.fstn.bn3.running_var"
#             torch.from_numpy(np.array(params["feat.fstn.bn3.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.fstn.bn3.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.fstn.bn3.running_mean"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.fstn.bn3.running_var"], dtype=np.float32)).reshape(-1).to(device='cuda'),
            )
        # last_result = current_result
        # current_result = torch.zeros(1024*1, dtype=torch.float32).to(device='cuda')
        # max_matrix<<<1024,1>>>(device_1024_1, device_1024_22500, 1024, 22500);
        max_matrix[1024, 1](
            device_1024_1, device_1024_22500, 1024, 22500
            )
        # last_result = current_result
        # current_result = torch.zeros(512*1, dtype=torch.float32).to(device='cuda')
        conv1d_norm_relu[512, 1](
            device_512_1, 
            512, 1024, 1, 
            # // "feat.fstn.fc1.weight","feat.fstn.fc1.bias"
            params["feat.fstn.fc1.weight"], 
            # torch.from_numpy(np.array(params["feat.fstn.fc1.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
            device_1024_1,
            params["feat.fstn.fc1.bias"], 
            params["feat.fstn.bn4.weight"], 
            params["feat.fstn.bn4.bias"], 
            params["feat.fstn.bn4.running_mean"], 
            params["feat.fstn.bn4.running_var"],
#             torch.from_numpy(np.array(params["feat.fstn.fc1.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
# #"feat.fstn.bn4.weight","feat.fstn.bn4.bias","feat.fstn.bn4.running_mean","feat.fstn.bn4.running_var"
#             torch.from_numpy(np.array(params["feat.fstn.bn4.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.fstn.bn4.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.fstn.bn4.running_mean"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.fstn.bn4.running_var"], dtype=np.float32)).reshape(-1).to(device='cuda'),
            )
        # last_result = current_result
        # current_result = torch.zeros(256*1, dtype=torch.float32).to(device='cuda')
        conv1d_norm_relu[256, 1](
            device_256_1, 
            256, 512, 1, 
# // "feat.fstn.fc2.weight","feat.fstn.fc2.bias"
            params["feat.fstn.fc2.weight"], 
            # torch.from_numpy(np.array(params["feat.fstn.fc2.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
            device_512_1,
            params["feat.fstn.fc2.bias"], 
            params["feat.fstn.bn5.weight"], 
            params["feat.fstn.bn5.bias"], 
            params["feat.fstn.bn5.running_mean"], 
            params["feat.fstn.bn5.running_var"],
#             torch.from_numpy(np.array(params["feat.fstn.fc2.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
# #"feat.fstn.bn5.weight","feat.fstn.bn5.bias","feat.fstn.bn5.running_mean","feat.fstn.bn5.running_var"
#             torch.from_numpy(np.array(params["feat.fstn.bn5.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.fstn.bn5.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.fstn.bn5.running_mean"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.fstn.bn5.running_var"], dtype=np.float32)).reshape(-1).to(device='cuda'),
            )
        # last_result = current_result
        # current_result = torch.zeros(4096*1, dtype=torch.float32).to(device='cuda')
        conv1d_4096_add_64_64[4096, 1](
            device_4096_1, 
            4096, 256, 1, 
# // "feat.fstn.fc3.weight","feat.fstn.fc3.bias"
            params["feat.fstn.fc3.weight"], 
            # torch.from_numpy(np.array(params["feat.fstn.fc3.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
            device_256_1,
            params["feat.fstn.fc3.bias"], 
            # torch.from_numpy(np.array(params["feat.fstn.fc3.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
            )
        # last_result = current_result
        # current_result = torch.zeros(22500*64, dtype=torch.float32).to(device='cuda')
        conv1d_22500_64_64_trans_result[22500, 64](
            device_64_22500, 
            22500, 64, 64, 
            device_22500_64_copy,
            device_4096_1,
            )
        # last_result = current_result
        # current_result = torch.zeros(128*22500, dtype=torch.float32).to(device='cuda')
        conv1d_norm_relu[128, 22500](
            device_128_22500, 
            128, 64, 22500, 
# // "feat.conv2.weight","feat.conv2.bias"
            params["feat.conv2.weight"], 
            # torch.from_numpy(np.array(params["feat.conv2.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
            device_64_22500,
            params["feat.conv2.bias"], 
            params["feat.bn2.weight"], 
            params["feat.bn2.bias"], 
            params["feat.bn2.running_mean"], 
            params["feat.bn2.running_var"],
#             torch.from_numpy(np.array(params["feat.conv2.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
# #"feat.bn2.weight","feat.bn2.bias","feat.bn2.running_mean","feat.bn2.running_var"
#             torch.from_numpy(np.array(params["feat.bn2.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.bn2.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.bn2.running_mean"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.bn2.running_var"], dtype=np.float32)).reshape(-1).to(device='cuda'),
            )
        # last_result = current_result
        # current_result = torch.zeros(1024*22500, dtype=torch.float32).to(device='cuda')
        conv1d_norm_relu[1024, 22500](
            device_1024_22500, 
            1024, 128, 22500, 
# // "feat.conv3.weight","feat.conv3.bias"
            params["feat.conv3.weight"], 
            # torch.from_numpy(np.array(params["feat.conv3.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
            device_128_22500,
            params["feat.conv3.bias"],
            params["feat.bn3.weight"], 
            params["feat.bn3.bias"], 
            params["feat.bn3.running_mean"], 
            params["feat.bn3.running_var"],
#             torch.from_numpy(np.array(params["feat.conv3.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
# #"feat.bn3.weight","feat.bn3.bias","feat.bn3.running_mean","feat.bn3.running_var"
#             torch.from_numpy(np.array(params["feat.bn3.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.bn3.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.bn3.running_mean"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["feat.bn3.running_var"], dtype=np.float32)).reshape(-1).to(device='cuda'),
            )
        # last_result = current_result
        # current_result = torch.zeros(1024*1, dtype=torch.float32).to(device='cuda')
        # max_matrix<<<1024,1>>>(device_1024_1, device_1024_22500, 1024, 22500);
        max_matrix[1024, 1](
            device_1024_1, device_1024_22500, 1024, 22500
            )
        # last_result = current_result
        # current_result = torch.zeros(512*1, dtype=torch.float32).to(device='cuda')
        conv1d_norm_relu[512, 1](
            device_512_1, 
            512, 1024, 1, 
# // "fc1.weight","fc1.bias"
            params["fc1.weight"], 
            # torch.from_numpy(np.array(params["fc1.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
            device_1024_1,
            params["fc1.bias"], 
            params["bn1.weight"], 
            params["bn1.bias"], 
            params["bn1.running_mean"], 
            params["bn1.running_var"],
#             torch.from_numpy(np.array(params["fc1.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
# #"bn1.weight","bn1.bias","bn1.running_mean","bn1.running_var"
#             torch.from_numpy(np.array(params["bn1.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["bn1.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["bn1.running_mean"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["bn1.running_var"], dtype=np.float32)).reshape(-1).to(device='cuda'),
            )
        # last_result = current_result
        # current_result = torch.zeros(256*1, dtype=torch.float32).to(device='cuda')
        conv1d_norm_relu[256, 1](
            device_256_1, 
            256, 512, 1, 
# // "fc2.weight","fc2.bias"
            params["fc2.weight"], 
            # torch.from_numpy(np.array(params["fc2.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
            device_512_1,
            params["fc2.bias"], 
            params["bn2.weight"], 
            params["bn2.bias"], 
            params["bn2.running_mean"], 
            params["bn2.running_var"],
#             torch.from_numpy(np.array(params["fc2.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
# #"bn2.weight","bn2.bias","bn2.running_mean","bn2.running_var"
#             torch.from_numpy(np.array(params["bn2.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["bn2.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["bn2.running_mean"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
#             torch.from_numpy(np.array(params["bn2.running_var"], dtype=np.float32)).reshape(-1).to(device='cuda'),
            )
        # last_result = current_result
        # current_result = torch.zeros(10*1, dtype=torch.float32).to(device='cuda')
        conv1d_log_softmax_ans[10, 1](
            device_10_1, 
            10, 256,
# // "fc3.weight","fc3.bias"
            params["fc3.weight"], 
            # torch.from_numpy(np.array(params["fc3.weight"], dtype=np.float32)).reshape(-1).to(device='cuda'), 
            device_256_1,
            params["fc3.bias"],
            # torch.from_numpy(np.array(params["fc3.bias"], dtype=np.float32)).reshape(-1).to(device='cuda'),
            )
        # print(f"current_result {current_result}")
        max_index = get_max_index(device_10_1)
        # print(f"max_index {max_index}")
        if max_index == list_of_labels[i]:
            correct_count += 1
        # break

    accuracy_rate = correct_count / len(list_of_labels)
    return accuracy_rate

if __name__ == '__main__':
    dir = os.path.dirname(__file__) # 保存模型参数文件(.txt)的文件夹路径
    # print(f"dir {dir}")
    # 读取模型参数 
    params = read_params(dir)
    # print(str(params["feat.stn.conv1.weight"][0:3]))
    # 读取训练集数据
    dataPath = "./data/test_point_clouds.h5"
    list_of_points,list_of_labels = read_h5_file(dataPath)
    target_points = 22500
    padding_value = (0, 0, 0)

    # 创建一个形状为 (len(list_of_points), target_points, 3) 的数组，初始化为填充值
    padded_points = np.zeros((len(list_of_points), target_points, 3))

    # device_of_points

    for i, points in enumerate(list_of_points):
        # print(points.shape)
        # break
        num_points = points.reshape(-1, 3).shape[0]
        current_point = points.reshape(-1, 3)
        if num_points < target_points:
            # 用 0 填充缺少的点
            padded_points[i, :num_points, :] = current_point[:num_points, :]
        else:
            # 裁剪到目标点数
            padded_points[i, :target_points, :] = current_point[:target_points, :]


    device_params = {}
    for key in params:
        device_params[key] = torch.from_numpy(np.array(params[key], dtype=np.float32)).reshape(-1).to(device='cuda')

    # 开始计时
    # global start
    start = time.time()
    accuracy_rate = do_inference(padded_points,list_of_labels,device_params)
    # 结束计时
    end = time.time()
    ms = end - start

    # 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    print(f"{ms:.4f}:{accuracy_rate:.4f}")