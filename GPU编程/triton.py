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
            # list_of_points.append(hf[k]["points"][:].astype(np.float16)) #每个points是（N,3）的二维数组ndarray
            list_of_points.append(hf[k]["points"][:].astype(np.float16).flatten()) #每个points是N*3的一维ndarray
            list_of_labels.append(hf[k].attrs["label"])
    return list_of_points,list_of_labels

device_64_22500 = torch.zeros(64*22500, dtype=torch.float16).to(device='cuda')
device_64_22500_2 = torch.zeros(64*22500, dtype=torch.float16).to(device='cuda')
device_22500_64_copy = torch.zeros(22500*64, dtype=torch.float16).to(device='cuda')
device_128_22500 = torch.zeros(128*22500, dtype=torch.float16).to(device='cuda')
device_1024_22500 = torch.zeros(1024*22500, dtype=torch.float16).to(device='cuda')
device_1024_1 = torch.zeros(1024*1, dtype=torch.float16).to(device='cuda')
device_512_1 = torch.zeros(512*1, dtype=torch.float16).to(device='cuda')
device_256_1 = torch.zeros(256*1, dtype=torch.float16).to(device='cuda')
device_9_1 = torch.zeros(9*1, dtype=torch.float16).to(device='cuda')
device_3_22500 = torch.zeros(3*22500, dtype=torch.float16).to(device='cuda')
device_64_64 = torch.zeros(64*64, dtype=torch.float16).to(device='cuda')
device_4096_1 = torch.zeros(4096*1, dtype=torch.float16).to(device='cuda')
device_10_1 = torch.zeros(10*1, dtype=torch.float16).to(device='cuda')
device_1000_10_1 = torch.zeros(1000*10*1, dtype=torch.float16).to(device='cuda')

# 定义Triton内核函数
@triton.jit
def conv1d_norm_relu(output, a_row, a_col, b_col, weight, input_array, bias, bn_weight, bn_bias, bn_running_mean, bn_running_var,
                     BLOCK_SIZE: tl.constexpr = 1024):
    idx = tl.program_id(0)
    idy = tl.program_id(1)

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

@triton.jit
def conv1d_norm_relu_start(
    output, a_row, a_col, b_col, weight, input_array, bias, 
    bn_weight, bn_bias, bn_running_mean, bn_running_var,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    idx = tl.program_id(0)
    store_idy = tl.program_id(1)
    idy = tl.program_id(1) * 10

    if idx < a_row and idy < 22500:
        sum = 0.0
        col_array = tl.arange(0, BLOCK_SIZE)
        mask = col_array < a_col
        sum = tl.sum(tl.load(weight + (idx * a_col + col_array), mask) * tl.load(input_array + (col_array * 22500 + idy), mask))
        sum += tl.load(bias + idx)
        x_mean = tl.load(bn_running_mean + idx)
        x_var = tl.load(bn_running_var + idx)
        bn1_w_val = tl.load(bn_weight + idx)
        bn1_b_val = tl.load(bn_bias + idx)
        norm_res = (sum - x_mean) / tl.sqrt(x_var + 1e-5)
        ans = norm_res * bn1_w_val + bn1_b_val
        ans = tl.maximum(ans, 0.0)
        tl.store(output + (idx * b_col + store_idy), ans)
        # tl.store(output + (idx * b_col + idy), sum)

@triton.jit
def max_matrix(output, input, row, col, BLOCK_SIZE: tl.constexpr = 2048):
    id = tl.program_id(0)
    row_data = tl.load(input + id * col + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < col)
    max_val = tl.max(row_data)
    tl.store(output + id, max_val)

@triton.jit
def conv1d_9(output,  weight, input, bias, constant_9_tensor):
    idx = tl.program_id(0)
    # if (idx<9):
    array_add = tl.arange(0, 256)
    mask = array_add < 256
    sum = tl.sum(tl.load(weight + idx*256 + array_add, mask) * tl.load(input + array_add, mask))
    sum += tl.load(bias + idx)
    sum += tl.load(constant_9_tensor + idx)
    tl.store(output + idx,sum)

@triton.jit
def matrix_array_9(output, a,  b,  a_row, a_col,  b_col,  
                    BLOCK_SIZE: tl.constexpr = 4):
    idx = tl.program_id(0)
    idy = tl.program_id(1)
    # if (idx<a_row and idy<b_col):
    id_T = idy*a_row+idx
    array_mul = tl.arange(0, BLOCK_SIZE)
    mask = array_mul < a_col
    sum = tl.sum(tl.load(a + idx*a_col + array_mul, mask) * tl.load(b + array_mul*b_col + idy, mask))
    tl.store(output + id_T, sum)

@triton.jit
def conv1d_norm_relu_copy_T(output,  output_T,  a_row,  a_col,  b_col,  weight, input, bias, bn_weight,bn_bias,  bn_running_mean, bn_running_var,
                            BLOCK_SIZE: tl.constexpr = 4):
    idx = tl.program_id(0)
    idy = tl.program_id(1)
    # if (idx<a_row and idy<b_col):
    id = idx*b_col+idy
    id_T = idy*a_row+idx
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
                          constant_64_tensor, BLOCK_SIZE: tl.constexpr = 1024):
    idx = tl.program_id(0)
    # if (idx<a_row):
    array_mul = tl.arange(0, BLOCK_SIZE)
    mask = array_mul < a_col
    sum = tl.sum(tl.load(weight + idx*a_col + array_mul, mask) * tl.load(input + array_mul, mask))
    sum += tl.load(bias + idx)
    sum += tl.load(constant_64_tensor + idx)
    tl.store(output + idx, sum)

@triton.jit
def conv1d_22500_64_64_trans_result( output,  a_row,  a_col,  b_col, input_a, input_b,
                                      BLOCK_SIZE: tl.constexpr = 1024):
    idx = tl.program_id(0)
    idy = tl.program_id(1)
    # if (idx<a_row and idy<a_col):
    array_mul = tl.arange(0, BLOCK_SIZE)
    mask = array_mul < b_col
    sum = tl.sum(tl.load(input_a + idx*a_col + array_mul, mask) * tl.load(input_b + array_mul*b_col + idy, mask))
    tl.store(output + idy*a_row+idx, sum)

@triton.jit
def conv1d_log_softmax_ans(output,  a_row,  a_col,  weight,  input,  bias,
                           BLOCK_SIZE: tl.constexpr = 256):
    idx = tl.program_id(0)
    # if (idx<a_row):
    array_mul = tl.arange(0, BLOCK_SIZE)
    mask = array_mul < a_col
    num = tl.sum(tl.load(weight + idx*a_col + array_mul, mask) * tl.load(input + array_mul, mask))
    num += tl.load(bias + idx)
    tl.store(output + idx, num)

def get_max_index (result):
    max_index = 0
    max_value = result[0]
    for i in range(1, 10):
        if (max_value < result[i]):
            max_value = result[i]
            max_index = i
    return max_index


from tqdm import tqdm
def do_inference(list_of_points,list_of_points_T, list_of_labels,params): #请在本函数下使用triton实现推理操作
    correct_count = 0
    # define a tensor
    constant_9_tensor = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float16).to(device='cuda')
    constant_64_tensor = torch.eye(64).flatten().type(torch.float16).to(device='cuda')
    down_size = 4000
    # for i in tqdm(range(len(list_of_points))):
    for i in range(len(list_of_points)):
    # for i in tqdm(range()):
        # if i>10:
        #     break
        device_points = list_of_points[i]
        device_points_T     = list_of_points_T[i]
        grid = (64, down_size)
        conv1d_norm_relu_start[grid](
            device_64_22500, 64, 3, down_size, 
            params["feat.stn.conv1.weight"], 
            device_points_T, 
            params["feat.stn.conv1.bias"], 
            params["feat.stn.bn1.weight"], 
            params["feat.stn.bn1.bias"], 
            params["feat.stn.bn1.running_mean"], 
            params["feat.stn.bn1.running_var"],
            # num_warps=32,
            )
        # continue
        # break
        # 打印输出结果，仅用于调试！
        # print(f"output_ptr {device_64_22500.data[0]}")
        # print(f"output_ptr {device_64_22500.data[223]}")
        # print(f"output_ptr {device_64_22500.data[225]}")
        # print(f"output_ptr {device_64_22500.data[450]}")
        # break
        # continue
        # last_result = output_ptr
        # current_result = torch.zeros(128*22500, dtype=torch.float16).to(device='cuda')
        # conv1d_norm_relu[128, 22500](
        grid = (128, down_size)
        # conv1d_norm_relu[128, tl.cdiv(22500, BIG_BLOCK_SIZE)](
        conv1d_norm_relu[grid](
            device_128_22500, 128, 64, down_size, 
            params["feat.stn.conv2.weight"], 
            device_64_22500, 
            params["feat.stn.conv2.bias"], 
            params["feat.stn.bn2.weight"], 
            params["feat.stn.bn2.bias"], 
            params["feat.stn.bn2.running_mean"], 
            params["feat.stn.bn2.running_var"],
            # num_warps=32,
            )
        # print(f"output_ptr {device_128_22500.data[0]}")
        # print(f"output_ptr {device_128_22500.data[223]}")
        # print(f"output_ptr {device_128_22500.data[446]}")
        # print(f"output_ptr {device_128_22500.data[892]}")
        # break
        grid = (1024, down_size)
        conv1d_norm_relu[grid](
        # conv1d_norm_relu[1024, 22500](
            device_1024_22500, 1024, 128, down_size, 
            params["feat.stn.conv3.weight"], 
            # torch.from_numpy(np.array(params["feat.stn.conv3.weight"], dtype=np.float16)).reshape(-1).to(device='cuda'), 
            device_128_22500, 
            params["feat.stn.conv3.bias"], 
            params["feat.stn.bn3.weight"], 
            params["feat.stn.bn3.bias"], 
            params["feat.stn.bn3.running_mean"], 
            params["feat.stn.bn3.running_var"],
            # num_warps=16,
            )
        # print(f"output_ptr {device_1024_22500.data}")
        # print(f"output_ptr {device_1024_22500.data[225:229]}")
        # break
        max_matrix[1024, 1](
            device_1024_1, device_1024_22500, 1024, down_size
            )
        
        
        # print(f"output_ptr {device_1024_1.data[0:10]}")
        # break
        # continue

        conv1d_norm_relu[512, 1](
            device_512_1, 512, 1024, 1, 
            params["feat.stn.fc1.weight"], 
            device_1024_1,
            params["feat.stn.fc1.bias"], 
            params["feat.stn.bn4.weight"], 
            params["feat.stn.bn4.bias"], 
            params["feat.stn.bn4.running_mean"], 
            params["feat.stn.bn4.running_var"],
            # num_warps=2,
            )
        conv1d_norm_relu[256, 1](
            device_256_1, 256, 512, 1, 
            params["feat.stn.fc2.weight"], 
            device_512_1,
            params["feat.stn.fc2.bias"], 
            params["feat.stn.bn5.weight"], 
            params["feat.stn.bn5.bias"], 
            params["feat.stn.bn5.running_mean"], 
            params["feat.stn.bn5.running_var"],
            num_warps=2,
            )
        conv1d_9[9, 1](
            device_9_1, 
            params["feat.stn.fc3.weight"], 
            device_256_1,
            params["feat.stn.fc3.bias"], 
            constant_9_tensor
            )
        # print(f"output_ptr {device_9_1.data[0:8]}")
        # break
        matrix_array_9[22500,3](
            device_3_22500, 
            device_points, 
            device_9_1, 
            22500, 3, 3,
            )
        # print(f"output_ptr {device_3_22500.data[0:10]}")
        # break
        # continue
        conv1d_norm_relu_copy_T[64, 22500](
            device_64_22500, device_22500_64_copy, 64, 3, 22500, 
            params["feat.conv1.weight"], 
            device_3_22500,
            params["feat.conv1.bias"], 
            params["feat.bn1.weight"], 
            params["feat.bn1.bias"], 
            params["feat.bn1.running_mean"], 
            params["feat.bn1.running_var"],
            )
        # print(f"output_ptr {device_64_22500.data[0:10]}")
        # print(f"output_ptr {device_64_22500.data[22400:22410]}")
        # break
        # 这里的device_64_22500和device_22500_64_copy都是正确的
        grid = (64, down_size)
        conv1d_norm_relu_start[grid](
        # conv1d_norm_relu[64, 22500](
            device_64_22500_2, 
            64, 64, down_size, 
            params["feat.fstn.conv1.weight"], 
            device_64_22500,
            params["feat.fstn.conv1.bias"], 
            params["feat.fstn.bn1.weight"], 
            params["feat.fstn.bn1.bias"], 
            params["feat.fstn.bn1.running_mean"], 
            params["feat.fstn.bn1.running_var"],
            )

        grid = (128, down_size)
        conv1d_norm_relu[grid](
        # conv1d_norm_relu[128, 22500](
            device_128_22500, 
            128, 64, down_size, 
            params["feat.fstn.conv2.weight"], 
            device_64_22500_2,
            params["feat.fstn.conv2.bias"], 
            params["feat.fstn.bn2.weight"], 
            params["feat.fstn.bn2.bias"], 
            params["feat.fstn.bn2.running_mean"], 
            params["feat.fstn.bn2.running_var"],
            )
        grid = (1024, down_size)
        conv1d_norm_relu[grid](
        # conv1d_norm_relu[1024, 22500](
            device_1024_22500, 
            1024, 128, down_size, 
            params["feat.fstn.conv3.weight"], 
            device_128_22500,
            params["feat.fstn.conv3.bias"], 
            params["feat.fstn.bn3.weight"], 
            params["feat.fstn.bn3.bias"], 
            params["feat.fstn.bn3.running_mean"], 
            params["feat.fstn.bn3.running_var"],
            )
        # max_matrix<<<1024,1>>>(device_1024_1, device_1024_22500, 1024, 22500);
        max_matrix[1024, 1](
            device_1024_1, device_1024_22500, 1024, down_size
            )
        conv1d_norm_relu[512, 1](
            device_512_1, 
            512, 1024, 1, 
            params["feat.fstn.fc1.weight"], 
            device_1024_1,
            params["feat.fstn.fc1.bias"], 
            params["feat.fstn.bn4.weight"], 
            params["feat.fstn.bn4.bias"], 
            params["feat.fstn.bn4.running_mean"], 
            params["feat.fstn.bn4.running_var"],
            )
        conv1d_norm_relu[256, 1](
            device_256_1, 
            256, 512, 1, 
            params["feat.fstn.fc2.weight"], 
            device_512_1,
            params["feat.fstn.fc2.bias"], 
            params["feat.fstn.bn5.weight"], 
            params["feat.fstn.bn5.bias"], 
            params["feat.fstn.bn5.running_mean"], 
            params["feat.fstn.bn5.running_var"],
            )
        conv1d_4096_add_64_64[4096, 1](
            device_4096_1, 
            4096, 256, 1, 
            params["feat.fstn.fc3.weight"], 
            device_256_1,
            params["feat.fstn.fc3.bias"], 
            constant_64_tensor,
            )
        # print(f"output_ptr {device_4096_1.data[0:10]}")
        # break
        conv1d_22500_64_64_trans_result[22500, 64](
            device_64_22500, 
            22500, 64, 64, 
            device_22500_64_copy,
            device_4096_1,
            )
        grid = (128, down_size)
        conv1d_norm_relu_start[grid](
            device_128_22500, 
            128, 64, down_size, 
            params["feat.conv2.weight"], 
            device_64_22500,
            params["feat.conv2.bias"], 
            params["feat.bn2.weight"], 
            params["feat.bn2.bias"], 
            params["feat.bn2.running_mean"], 
            params["feat.bn2.running_var"],
            )
        grid = (1024, down_size)
        conv1d_norm_relu[grid](
            device_1024_22500, 
            1024, 128, down_size, 
            params["feat.conv3.weight"], 
            device_128_22500,
            params["feat.conv3.bias"],
            params["feat.bn3.weight"], 
            params["feat.bn3.bias"], 
            params["feat.bn3.running_mean"], 
            params["feat.bn3.running_var"],
            )
        max_matrix[1024, 1](
            device_1024_1, device_1024_22500, 1024, down_size
            )
        conv1d_norm_relu[512, 1](
            device_512_1, 
            512, 1024, 1, 
            params["fc1.weight"], 
            device_1024_1,
            params["fc1.bias"], 
            params["bn1.weight"], 
            params["bn1.bias"], 
            params["bn1.running_mean"], 
            params["bn1.running_var"],
            )
        conv1d_norm_relu[256, 1](
            device_256_1, 
            256, 512, 1, 
            params["fc2.weight"], 
            device_512_1,
            params["fc2.bias"], 
            params["bn2.weight"], 
            params["bn2.bias"], 
            params["bn2.running_mean"], 
            params["bn2.running_var"],
            )
        conv1d_log_softmax_ans[10, 1](
            device_10_1, 
            10, 256,
            params["fc3.weight"], 
            device_256_1,
            params["fc3.bias"],
            )
        max_index = get_max_index(device_10_1)
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

    device_of_points = []
    device_of_points_T = []

    for i, points in enumerate(list_of_points):
        # print(points.shape)
        # break
        num_points = points.reshape(-1, 3).shape[0]
        current_point = points.reshape(-1, 3)
        if num_points < target_points:
            # 用 0 填充缺少的点
            padded_points[i, :num_points, :] = current_point[:num_points, :]
            device_of_points.append( torch.from_numpy(padded_points[i]).reshape(-1).to(device='cuda'))
            device_of_points_T.append( torch.from_numpy(padded_points[i].T).reshape(-1).to(device='cuda'))
        else:
            # 裁剪到目标点数
            padded_points[i, :target_points, :] = current_point[:target_points, :]
            device_of_points.append( torch.from_numpy(padded_points[i]).reshape(-1).to(device='cuda'))
            device_of_points_T.append( torch.from_numpy(padded_points[i].T).reshape(-1).to(device='cuda'))



    device_params = {}
    for key in params:
        device_params[key] = torch.from_numpy(np.array(params[key], dtype=np.float16)).reshape(-1).to(device='cuda')


    # 开始计时
    # global start
    start = time.time()
    accuracy_rate = do_inference(device_of_points,device_of_points_T,list_of_labels,device_params)
    # 结束计时
    end = time.time()
    ms = end - start

    # 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    print(f"{ms:.4f}:{accuracy_rate:.4f}")