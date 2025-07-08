// 这是程序二的模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现CUDA的深度学习推理过程，请严格保持输出格式输出
// 编译的命令为：nvcc test.cu -o test -Xcompiler "-O3 -std=c++14" -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -lhdf5 -lhdf5_cpp
// 编译的命令为：
// nvcc batch_process.cu -o batxh_process -Xcompiler "-O3 -std=c++14" -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -lhdf5 -lhdf5_cpp

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <fstream>
#include <map>
#include <dirent.h>
#include <cstring>
#include <hdf5/serial/H5Cpp.h>
using namespace std;

/****************************************************************************************
 * 读取模型参数
 ****************************************************************************************/
// 获取目录中的所有 .txt 文件
std::vector<std::string> get_files_in_directory(const std::string& dir) {
    std::vector<std::string> files;
    DIR* dp;
    struct dirent* entry;
    if ((dp = opendir(dir.c_str())) != NULL) {
        while ((entry = readdir(dp)) != NULL) {
            std::string filename = entry->d_name;
            if (filename.find(".txt") != std::string::npos) {
                files.push_back(filename);
            }
        }
        closedir(dp);
    } else {
        perror("opendir");
    }
    return files;
}

// 读取 .txt 文件并转换为 std::vector<float>
std::vector<float> read_param(const std::string& filepath) {
    std::vector<float> data;
    std::ifstream file(filepath);
    if (file.is_open()) {
        float value;
        while (file >> value) {
            data.push_back(value);
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filepath << std::endl;
    }
    return data;
}

std::map<std::string, std::vector<float>> read_params(std::string dir) {
    // std::string dir = "."; // 当前目录
    std::map<std::string, std::vector<float>> params;

    // 获取目录中的所有 .txt 文件
    std::vector<std::string> param_files = get_files_in_directory(dir);
    for (const auto& file : param_files) {
        std::string filename = file.substr(0, file.find_last_of(".")); // 获取不带扩展名的文件名
        params[filename] = read_param(dir + "/" + file);
    }

    // // 访问参数时可以使用 params["conv1_weight"]
    // for (const auto& kv : params) {
    //     std::cout << "Key: " << kv.first << ", Values: ";
    //     // for (const auto& value : kv.second) {
    //     //     std::cout << value << " ";
    //     // }
    //     std::cout << std::endl;
    // }

    return params;
}

/****************************************************************************************
 * 读取训练集数据
 ****************************************************************************************/

using namespace H5;
void read_h5_file(const std::string& file_path, std::vector<std::vector<float>>& list_of_points, std::vector<int>& list_of_labels) {
    try {
        // 打开文件
        H5File file(file_path, H5F_ACC_RDONLY);

        // 获取文件中的所有数据集名称
        std::vector<std::string> dataset_names;
        hsize_t num_objs = file.getNumObjs();
        for (hsize_t i = 0; i < num_objs; i++) {
            dataset_names.push_back(file.getObjnameByIdx(i));
        }

        // 读取每个数据集
        for (const auto& name : dataset_names) {
            DataSet dataset = file.openDataSet(name + "/points");
            DataSpace dataspace = dataset.getSpace();

            // 获取数据集的维度
            hsize_t dims[2];
            dataspace.getSimpleExtentDims(dims, NULL);

            // 读取数据
            std::vector<float> points(dims[0] * dims[1]);
            dataset.read(points.data(), PredType::NATIVE_FLOAT);

            // 存储点云数据
            list_of_points.push_back(points);

            // 读取标签
            Attribute label_attr = file.openGroup(name).openAttribute("label");
            int label;
            label_attr.read(PredType::NATIVE_INT, &label);

            // 存储标签
            list_of_labels.push_back(label);
        }
    } catch (FileIException& error) {
        error.printErrorStack();
    } catch (DataSetIException& error) {
        error.printErrorStack();
    } catch (DataSpaceIException& error) {
        error.printErrorStack();
    } catch (DataTypeIException& error) {
        error.printErrorStack();
    }
}
// #define devtype int

//clerations for all the kernel functions
int log_softmax(float *result);
void print_device_array(float *array, int row, int col, bool from22400 = true);
void print_device_array_64(float *array, int row, int col, bool from22400);

#define print_int(x) printf("%d\n", (int)(x))
#define print_float(x) printf("%f\n", (float)(x))
#define forloop(i, start, end) for (int i = (start); i <= (end); ++i)

__global__ void conv1d_norm_relu(float *output, int a_row, int a_col, int b_col, float *weight, float *input, float *bias, float *bn_weight, float *bn_bias, float *bn_running_mean, float *bn_running_var);

__global__ void conv1d_log_softmax_ans(float *output_num, int a_row, int a_col, float *weight, float *input, float *bias);

void print_device_float(float *num);

__global__ void conv1d_norm(float *output, int a_row, int a_col, int b_col, float *weight, float *input, float *bias, float *bn_weight, float *bn_bias, float *bn_running_mean, float *bn_running_var);

__global__ void conv1d_4096_add_64_64(float *output, int a_row, int a_col, int b_col, float *weight, float *input, float *bias);

__global__ void conv1d_22500_64_64_trans_result(float *output, int a_row, int a_col, int b_col, float *input_a, float *input_b);

__global__ void conv1d_norm_relu_copy_T(float *output, float *output_T, int a_row, int a_col, int b_col, float *weight, float *input, float *bias, float *bn_weight, float *bn_bias, float *bn_running_mean, float *bn_running_var);

__global__ void max_matrix(float *output, float *input, int row, int col);

__global__ void conv1d_9(float *output, float *weight, float *input, float *bias);

__global__ void matrix_array_9(float *output, float *a, float *b, int a_row, int a_col, int b_col);

void HANDLE_ERROR(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        // 处理错误，例如退出程序等
    }
}

int main(int argc, char *argv[]) {
    // std::string dir = "parameters";  // 第一个参数是程序所在的目录，这个目录是存放前一步训练模型参数文件的目录，从这个目录下读取模型参数文件，相对于这个目录读取测试集点云数据和标签
    
    std::string dir = argv[1];  // 第一个参数是程序所在的目录，这个目录是存放前一步训练模型参数文件的目录，从这个目录下读取模型参数文件，相对于这个目录读取测试集点云数据和标签
    // cout << dir;
    
    // 读取模型参数
    auto params = read_params(dir);

    std::string file_path = "./data/test_point_clouds.h5";
    std::vector<std::vector<float>> list_of_points;
    std::vector<int> list_of_labels;
    // 读取训练集数据
    read_h5_file(file_path, list_of_points, list_of_labels);

    // 开始计时，使用chrono计时，不支持其它计时方式

    int correct_count = 0;
    int total_count = list_of_points.size();
    // __constant__ float * data;
    // cudaMalloc(&data,1000*22500*3*sizeof(float));

    // freopen("out", "w", stdout);

    // 将vector的数据全部转化为cuda的device端的数组，存储转置前和转置后的点云数据
    // float array_of_points[total_count][22500*3];
    // memset(array_of_points, 0, total_count * 22500*3 * sizeof(float));
    // float array_of_points_T[total_count][22500*3];
    // puts("=======================================================");
    //赋值
    float * device_array_1000_22500_3[total_count];
    float * device_array_1000_22500_3_T[total_count];
    for (int i=0;i<total_count;++i){
        vector<float> current_point = list_of_points[i];
        float array_of_points[22500*3];
        int size = current_point.size();
        for (int j=0;j<22500*3;++j){
            if (j<size)
                array_of_points[j] = current_point[j];
            else 
                array_of_points[j] = 0;
        }
        //为转置矩阵赋值
        float array_of_points_T[22500*3];
        int count = 0;
        for (int j=0;j<3;++j)
            for (int k=j;k<22500*3;k+=3){
                array_of_points_T[count ++ ] = array_of_points[k];
            } 
        
        // // 转置前和转置后的点云数据转化为cuda的device端的数组
        float * tmp_1; HANDLE_ERROR( cudaMalloc((void**)&tmp_1, 22500*3*sizeof(float)));
        float * tmp_2; HANDLE_ERROR( cudaMalloc((void**)&tmp_2, 22500*3*sizeof(float)));
        HANDLE_ERROR( cudaMemcpy(tmp_1, array_of_points, 22500*3*sizeof(float), cudaMemcpyHostToDevice));
        HANDLE_ERROR( cudaMemcpy(tmp_2, array_of_points_T, 22500*3*sizeof(float), cudaMemcpyHostToDevice));
        device_array_1000_22500_3[i] = tmp_1;
        device_array_1000_22500_3_T[i] = tmp_2;
    }
    // puts("=======================================================");
    // list_of_points.clear();


    map<string, float*> device_params;
    for (auto& kv : params){
        string name = kv.first;
        vector<float> values = kv.second;
        float * tmp;
        HANDLE_ERROR( cudaMalloc((void**)&tmp, values.size()*sizeof(float)));
        HANDLE_ERROR( cudaMemcpy(tmp, values.data(), values.size()*sizeof(float), cudaMemcpyHostToDevice));
        device_params[name] = tmp;
    }
    // params.clear();

    //Free all the memory
    // for (auto& kv : device_params){
    //     HANDLE_ERROR( cudaFree(kv.second));
    // }

    //Free all the 22500*3*sizeof(float) memory
    // for (int i=0;i<total_count;++i){
    //     HANDLE_ERROR( cudaFree(device_array_1000_22500_3[i]));
    //     HANDLE_ERROR( cudaFree(device_array_1000_22500_3_T[i]));
    // }

    // cudaDeviceSynchronize();

    // // 结束计时
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> diff = end - start;

    const int batch_size = 10;

    // float * device_64_22500;
    // HANDLE_ERROR( cudaMalloc((void**)&device_64_22500, 64*22500*sizeof(float)));

    float * device_64_22500_BATCH;
    HANDLE_ERROR( cudaMalloc((void**)&device_64_22500_BATCH, batch_size*64*22500*sizeof(float)));

    // float * device_64_22500_2;
    // HANDLE_ERROR( cudaMalloc((void**)&device_64_22500_2, 64*22500*sizeof(float)));

    float * device_64_22500_2_BATCH;
    HANDLE_ERROR( cudaMalloc((void**)&device_64_22500_2_BATCH, batch_size*64*22500*sizeof(float)));

    // float * device_22500_64_copy;
    // HANDLE_ERROR( cudaMalloc((void**)&device_22500_64_copy, 22500*64*sizeof(float)));

    float * device_22500_64_copy_BATCH;
    HANDLE_ERROR( cudaMalloc((void**)&device_22500_64_copy_BATCH, batch_size*22500*64*sizeof(float)));

    // float * device_128_22500;
    // HANDLE_ERROR( cudaMalloc((void**)&device_128_22500, 128*22500*sizeof(float)));

    float * device_128_22500_BATCH;
    HANDLE_ERROR( cudaMalloc((void**)&device_128_22500_BATCH, batch_size*128*22500*sizeof(float)));

    // float * device_1024_22500;
    // HANDLE_ERROR( cudaMalloc((void**)&device_1024_22500, 1024*22500*sizeof(float)));

    float * device_1024_22500_BATCH;
    HANDLE_ERROR( cudaMalloc((void**)&device_1024_22500_BATCH, batch_size*1024*22500*sizeof(float)));

    // float * device_1024_1;
    // HANDLE_ERROR( cudaMalloc((void**)&device_1024_1, 1024*1*sizeof(float)));

    float * device_1024_1_BATCH;
    HANDLE_ERROR( cudaMalloc((void**)&device_1024_1_BATCH, batch_size*1024*1*sizeof(float)));

    // float * device_512_1;
    // HANDLE_ERROR( cudaMalloc((void**)&device_512_1, 512*1*sizeof(float)));
    float * device_512_1_BATCH;
    HANDLE_ERROR( cudaMalloc((void**)&device_512_1_BATCH, batch_size*512*1*sizeof(float)));

    // float * device_256_1;
    // HANDLE_ERROR( cudaMalloc((void**)&device_256_1, 256*1*sizeof(float)));
    float * device_256_1_BATCH;
    HANDLE_ERROR( cudaMalloc((void**)&device_256_1_BATCH, batch_size*256*1*sizeof(float)));

    // float * device_9_1;
    // HANDLE_ERROR( cudaMalloc((void**)&device_9_1, 9*1*sizeof(float)));
    float * device_9_1_BATCH;
    HANDLE_ERROR( cudaMalloc((void**)&device_9_1_BATCH, batch_size*9*1*sizeof(float)));

    // float * device_3_22500;
    // HANDLE_ERROR( cudaMalloc((void**)&device_3_22500, 3*22500*sizeof(float)));
    float * device_3_22500_BATCH;
    HANDLE_ERROR( cudaMalloc((void**)&device_3_22500_BATCH, batch_size*3*22500*sizeof(float)));

    // float * device_64_64;
    // HANDLE_ERROR( cudaMalloc((void**)&device_64_64, 64*64*sizeof(float)));
    float * device_64_64_BATCH;
    HANDLE_ERROR( cudaMalloc((void**)&device_64_64_BATCH, batch_size*64*64*sizeof(float)));

    // float * device_4096_1;
    // HANDLE_ERROR( cudaMalloc((void**)&device_4096_1, 4096*1*sizeof(float)));
    float * device_4096_1_BATCH;
    HANDLE_ERROR( cudaMalloc((void**)&device_4096_1_BATCH, batch_size*4096*1*sizeof(float)));

    // float * device_10_1;
    // HANDLE_ERROR( cudaMalloc((void**)&device_10_1, 10*1*sizeof(float)));
    float * device_10_1_BATCH;
    HANDLE_ERROR( cudaMalloc((void**)&device_10_1_BATCH, batch_size*10*1*sizeof(float)));

    // float * device_1000_10_1;
    // HANDLE_ERROR( cudaMalloc((void**)&device_1000_10_1, 1000*10*1*sizeof(float)));
    float * device_1000_10_1_BATCH;
    HANDLE_ERROR( cudaMalloc((void**)&device_1000_10_1_BATCH, batch_size*1000*10*1*sizeof(float)));

    auto start = std::chrono::high_resolution_clock::now();


    // const int numStreams = 10;
    // cudaStream_t streams[numStreams];
    // for (int i = 0; i < numStreams; ++i) {
    //     HANDLE_ERROR(cudaStreamCreate(&streams[i])) ;
    // }

    float * host_1000_10_1 = (float *)malloc(sizeof(float)*1000*10*1);
    float * host_10_1 = (float *)malloc(sizeof(float)*10*1);


    forloop(i, 0, total_count-1){
        // int streamIdx = i % numStreams;  // 将每个迭代分配到一个流中
        // cudaStream_t stream = streams[streamIdx];
        int label = list_of_labels[i];
        // conferencing(device_array_1000_22500_3[i], device_array_1000_22500_3_T[i], params);
        // float * result = NULL;
        dim3 threadPerBlock_1 (4, 16);  // 2，16的话 2 会变慢
        dim3 blockPerGrid_1 ((64+threadPerBlock_1.x-1)/threadPerBlock_1.x,(22500+threadPerBlock_1.y-1)/threadPerBlock_1.y);


        conv1d_norm_relu<<<blockPerGrid_1,threadPerBlock_1>>>(
            device_64_22500,
            64, 3, 22500,
            device_params["feat.stn.conv1.weight"],
            device_array_1000_22500_3_T[i],
            device_params["feat.stn.conv1.bias"],
//"feat.stn.bn1.weight","feat.stn.bn1.bias","feat.stn.bn1.running_mean","feat.stn.bn1.running_var"
            device_params["feat.stn.bn1.weight"],
            device_params["feat.stn.bn1.bias"],
            device_params["feat.stn.bn1.running_mean"],
            device_params["feat.stn.bn1.running_var"]
        );

        // print_device_array(device_64_22500, 64, 22500, true);

        // break;

        dim3 threadPerBlock_2 (4, 16);
        dim3 blockPerGrid_2 ((128+threadPerBlock_2.x-1)/threadPerBlock_2.x,(22500+threadPerBlock_2.y-1)/threadPerBlock_2.y);


        conv1d_norm_relu<<<blockPerGrid_2,threadPerBlock_2>>>(
            device_128_22500,
            128, 64, 22500,
//"feat.stn.conv2.weight","feat.stn.conv2.bias"
            device_params["feat.stn.conv2.weight"],
            device_64_22500,
            device_params["feat.stn.conv2.bias"],
// "feat.stn.bn2.weight", "feat.stn.bn2.bias", "feat.stn.bn2.running_mean", "feat.stn.bn2.running_var"
            device_params["feat.stn.bn2.weight"],
            device_params["feat.stn.bn2.bias"],
            device_params["feat.stn.bn2.running_mean"],
            device_params["feat.stn.bn2.running_var"]
        );

        // print_device_array(device_128_22500, 128, 22500, true);

        // break;

        dim3 threadPerBlock_3 (4, 64);
        dim3 blockPerGrid_3 ((1024+threadPerBlock_3.x-1)/threadPerBlock_3.x,(22500+threadPerBlock_3.y-1)/threadPerBlock_3.y);


        conv1d_norm_relu<<<blockPerGrid_3,threadPerBlock_3>>>(
            device_1024_22500,
            1024, 128, 22500,
//"feat.stn.conv3.weight","feat.stn.conv3.bias"
            device_params["feat.stn.conv3.weight"],
            device_128_22500,
            device_params["feat.stn.conv3.bias"],
// "feat.stn.bn3.weight", "feat.stn.bn3.bias", "feat.stn.bn3.running_mean", "feat.stn.bn3.running_var"
            device_params["feat.stn.bn3.weight"],
            device_params["feat.stn.bn3.bias"],
            device_params["feat.stn.bn3.running_mean"],
            device_params["feat.stn.bn3.running_var"]
        );

        max_matrix<<<1024,16>>>(device_1024_1, device_1024_22500, 1024, 22500);

        // print_device_array(device_1024_1, 1024, 1, false);

        // break;

        dim3 threadPerBlock_4 (4, 4);
        dim3 blockPerGrid_4 ((512+threadPerBlock_4.x-1)/threadPerBlock_4.x,(1+threadPerBlock_4.y-1)/threadPerBlock_4.y);


        conv1d_norm_relu<<<blockPerGrid_4,threadPerBlock_4>>>(
            device_512_1,
            512, 1024, 1,
//"feat.stn.fc1.weight","feat.stn.fc1.bias"
            device_params["feat.stn.fc1.weight"],
            device_1024_1,
            device_params["feat.stn.fc1.bias"],
// "feat.stn.bn4.weight", "feat.stn.bn4.bias", "feat.stn.bn4.running_mean", "feat.stn.bn4.running_var"
            device_params["feat.stn.bn4.weight"],
            device_params["feat.stn.bn4.bias"],
            device_params["feat.stn.bn4.running_mean"],
            device_params["feat.stn.bn4.running_var"]
        );

        // print_device_array(device_512_1, 512, 1, false);

        // break;

        dim3 threadPerBlock_5 (4, 4);
        dim3 blockPerGrid_5 ((256+threadPerBlock_5.x-1)/threadPerBlock_5.x,(1+threadPerBlock_5.y-1)/threadPerBlock_5.y);


        conv1d_norm_relu<<<blockPerGrid_5,threadPerBlock_5>>>(
            device_256_1,
            256, 512, 1,
//"feat.stn.fc2.weight","feat.stn.fc2.bias"
            device_params["feat.stn.fc2.weight"],
            device_512_1,
            device_params["feat.stn.fc2.bias"],
// "feat.stn.bn5.weight", "feat.stn.bn5.bias", "feat.stn.bn5.running_mean", "feat.stn.bn5.running_var"
            device_params["feat.stn.bn5.weight"],
            device_params["feat.stn.bn5.bias"],
            device_params["feat.stn.bn5.running_mean"],
            device_params["feat.stn.bn5.running_var"]
        );

        // print_device_array(device_256_1, 256, 1, false);

        // break;

        
        conv1d_9<<<9,1>>>(device_9_1, device_params["feat.stn.fc3.weight"], device_256_1, device_params["feat.stn.fc3.bias"]);

        // print_device_array(device_9_1, 9, 1, false);

        dim3 threadPerBlock_6 (16, 4);
        dim3 blockPerGrid_6 ((22500+threadPerBlock_6.x-1)/threadPerBlock_6.x,(3+threadPerBlock_6.y-1)/threadPerBlock_6.y);
        matrix_array_9<<<blockPerGrid_6,threadPerBlock_6>>>(device_3_22500 , device_array_1000_22500_3[i], device_9_1, 22500, 3, 3);

        // print_device_array(device_3_22500, 3, 22500, true);
        // break;


        dim3 threadPerBlock_7 (4, 16);
        dim3 blockPerGrid_7 ((64+threadPerBlock_7.x-1)/threadPerBlock_7.x,(22500+threadPerBlock_7.y-1)/threadPerBlock_7.y);
        conv1d_norm_relu_copy_T<<<blockPerGrid_7,threadPerBlock_7>>>(
            device_64_22500,
            device_22500_64_copy,
            64, 3, 22500,
//"feat.conv1.weight","feat.conv1.bias"
            device_params["feat.conv1.weight"],
            device_3_22500,
            device_params["feat.conv1.bias"],
// "feat.bn1.weight", "feat.bn1.bias", "feat.bn1.running_mean", "feat.bn1.running_var"
            device_params["feat.bn1.weight"],
            device_params["feat.bn1.bias"],
            device_params["feat.bn1.running_mean"],
            device_params["feat.bn1.running_var"]
        );

        dim3 threadPerBlock_8 (4, 16);
        dim3 blockPerGrid_8 ((64+threadPerBlock_8.x-1)/threadPerBlock_8.x,(22500+threadPerBlock_8.y-1)/threadPerBlock_8.y);

        conv1d_norm_relu<<<blockPerGrid_8,threadPerBlock_8>>>(
            device_64_22500_2,
            64, 64, 22500,
// "feat.fstn.conv1.weight","feat.fstn.conv1.bias"
            device_params["feat.fstn.conv1.weight"],
            device_64_22500,
            device_params["feat.fstn.conv1.bias"],
// "feat.fstn.bn1.weight","feat.fstn.bn1.bias","feat.fstn.bn1.running_mean","feat.fstn.bn1.running_var"
            device_params["feat.fstn.bn1.weight"],
            device_params["feat.fstn.bn1.bias"],
            device_params["feat.fstn.bn1.running_mean"],
            device_params["feat.fstn.bn1.running_var"]
        );


        dim3 threadPerBlock_9 (4, 16);
        dim3 blockPerGrid_9 ((128+threadPerBlock_9.x-1)/threadPerBlock_9.x,(22500+threadPerBlock_9.y-1)/threadPerBlock_9.y);

        conv1d_norm_relu<<<blockPerGrid_9,threadPerBlock_9>>>(
            device_128_22500,
            128, 64, 22500,
// "feat.fstn.conv2.weight","feat.fstn.conv2.bias"
            device_params["feat.fstn.conv2.weight"],
            device_64_22500_2,
            device_params["feat.fstn.conv2.bias"],
// "feat.fstn.bn2.weight","feat.fstn.bn2.bias","feat.fstn.bn2.running_mean","feat.fstn.bn2.running_var"
            device_params["feat.fstn.bn2.weight"],
            device_params["feat.fstn.bn2.bias"],
            device_params["feat.fstn.bn2.running_mean"],
            device_params["feat.fstn.bn2.running_var"]
        );

        dim3 threadPerBlock_10 (4, 64); // 这里不能放6 64 会变慢
        dim3 blockPerGrid_10 ((1024+threadPerBlock_10.x-1)/threadPerBlock_10.x,(22500+threadPerBlock_10.y-1)/threadPerBlock_10.y);

        conv1d_norm_relu<<<blockPerGrid_10,threadPerBlock_10>>>(
            device_1024_22500,
            1024, 128, 22500,
// "feat.fstn.conv3.weight","feat.fstn.conv3.bias"
            device_params["feat.fstn.conv3.weight"],
            device_128_22500,
            device_params["feat.fstn.conv3.bias"],
// "feat.fstn.bn3.weight","feat.fstn.bn3.bias","feat.fstn.bn3.running_mean","feat.fstn.bn3.running_var"
            device_params["feat.fstn.bn3.weight"],
            device_params["feat.fstn.bn3.bias"],
            device_params["feat.fstn.bn3.running_mean"],
            device_params["feat.fstn.bn3.running_var"]
        );

        max_matrix<<<1024,16>>>(device_1024_1, device_1024_22500, 1024, 22500);

        // print_device_array(device_1024_1, 1024, 1, false);
        // break;

        dim3 threadPerBlock_11 (4, 4);
        dim3 blockPerGrid_11 ((512+threadPerBlock_11.x-1)/threadPerBlock_11.x,(1+threadPerBlock_11.y-1)/threadPerBlock_11.y);

        conv1d_norm_relu<<<blockPerGrid_11,threadPerBlock_11>>>(
            device_512_1,
            512, 1024, 1,
// "feat.fstn.fc1.weight","feat.fstn.fc1.bias"
            device_params["feat.fstn.fc1.weight"],
            device_1024_1,
            device_params["feat.fstn.fc1.bias"],
// "feat.fstn.bn4.weight","feat.fstn.bn4.bias","feat.fstn.bn4.running_mean","feat.fstn.bn4.running_var"
            device_params["feat.fstn.bn4.weight"],
            device_params["feat.fstn.bn4.bias"],
            device_params["feat.fstn.bn4.running_mean"],
            device_params["feat.fstn.bn4.running_var"]
        );


        dim3 threadPerBlock_12 (4, 4);
        dim3 blockPerGrid_12 ((256+threadPerBlock_12.x-1)/threadPerBlock_12.x,(1+threadPerBlock_12.y-1)/threadPerBlock_12.y);

        conv1d_norm_relu<<<blockPerGrid_12,threadPerBlock_12>>>(
            device_256_1,
            256, 512, 1,
// "feat.fstn.fc2.weight","feat.fstn.fc2.bias"
            device_params["feat.fstn.fc2.weight"],
            device_512_1,
            device_params["feat.fstn.fc2.bias"],
// "feat.fstn.bn5.weight","feat.fstn.bn5.bias","feat.fstn.bn5.running_mean","feat.fstn.bn5.running_var"
            device_params["feat.fstn.bn5.weight"],
            device_params["feat.fstn.bn5.bias"],
            device_params["feat.fstn.bn5.running_mean"],
            device_params["feat.fstn.bn5.running_var"]
        );

        conv1d_4096_add_64_64<<<4096,1>>>(
            device_4096_1,
            4096, 256, 1,
// "feat.fstn.fc3.weight","feat.fstn.fc3.bias"
            device_params["feat.fstn.fc3.weight"],
            device_256_1,
            device_params["feat.fstn.fc3.bias"]
        );

        // print_device_array_64(device_4096_1, 64, 64, false);
        // break;

        dim3 threadPerBlock_13 (16, 4);
        dim3 blockPerGrid_13 ((22500+threadPerBlock_13.x-1)/threadPerBlock_13.x,(64+threadPerBlock_13.y-1)/threadPerBlock_13.y);

        conv1d_22500_64_64_trans_result<<<blockPerGrid_13,threadPerBlock_13>>>(
            device_64_22500, // trans_result
            22500, 64, 64,
            device_22500_64_copy,
            device_4096_1
        );

        // print_device_array(device_64_22500, 64, 22500, true);
        // break;

        dim3 threadPerBlock_14 (4, 16);
        dim3 blockPerGrid_14 ((128+threadPerBlock_14.x-1)/threadPerBlock_14.x,(22500+threadPerBlock_14.y-1)/threadPerBlock_14.y);
        // dim3 threadPerBlock_14 (16, 16);
        // dim3 blockPerGrid_14 ((128+16-1)/16,(22500+16-1)/16);
        conv1d_norm_relu<<<blockPerGrid_14,threadPerBlock_14>>>(
            device_128_22500,
            128, 64, 22500,
// "feat.conv2.weight","feat.conv2.bias"
            device_params["feat.conv2.weight"],
            device_64_22500,
            device_params["feat.conv2.bias"],
// "feat.bn2.weight","feat.bn2.bias","feat.bn2.running_mean","feat.bn2.running_var"
            device_params["feat.bn2.weight"],
            device_params["feat.bn2.bias"],
            device_params["feat.bn2.running_mean"],
            device_params["feat.bn2.running_var"]
        );

        dim3 threadPerBlock_15 (4, 64);
        dim3 blockPerGrid_15 ((1024+threadPerBlock_15.x-1)/threadPerBlock_15.x,(22500+threadPerBlock_15.y-1)/threadPerBlock_15.y);
        // dim3 threadPerBlock_15 (16, 16);
        // dim3 blockPerGrid_15 ((1024+16-1)/16,(22500+16-1)/16);
        conv1d_norm<<<blockPerGrid_15,threadPerBlock_15>>>(
            device_1024_22500,
            1024, 128, 22500,
// "feat.conv3.weight","feat.conv3.bias"
            device_params["feat.conv3.weight"],
            device_128_22500,
            device_params["feat.conv3.bias"],
// "feat.bn3.weight","feat.bn3.bias","feat.bn3.running_mean","feat.bn3.running_var"
            device_params["feat.bn3.weight"],
            device_params["feat.bn3.bias"],
            device_params["feat.bn3.running_mean"],
            device_params["feat.bn3.running_var"]
        );

        max_matrix<<<1024,100>>>(device_1024_1, device_1024_22500, 1024, 22500);

        // print_device_array(device_1024_1, 1024, 1, false);
        // break;

        // dim3 threadPerBlock_16 (16, 1);
        // dim3 blockPerGrid_16 ((512+16-1)/16,1);
        dim3 threadPerBlock_16 (4, 4);
        dim3 blockPerGrid_16 ((512+threadPerBlock_16.x-1)/threadPerBlock_16.x,(1+threadPerBlock_16.y-1)/threadPerBlock_16.y);

        conv1d_norm_relu<<<blockPerGrid_16,threadPerBlock_16>>>(
            device_512_1,
            512, 1024, 1,
// "fc1.weight","fc1.bias"
            device_params["fc1.weight"],
            device_1024_1,
            device_params["fc1.bias"],
// "bn1.weight","bn1.bias","bn1.running_mean","bn1.running_var"
            device_params["bn1.weight"],
            device_params["bn1.bias"],
            device_params["bn1.running_mean"],
            device_params["bn1.running_var"]
        );

        dim3 threadPerBlock_17 (4, 4);
        // dim3 blockPerGrid_17 ((256+16-1)/16,1);
        dim3 blockPerGrid_17 ((256+threadPerBlock_17.x-1)/threadPerBlock_17.x,(1+threadPerBlock_17.y-1)/threadPerBlock_17.y);

        conv1d_norm_relu<<<blockPerGrid_17,threadPerBlock_17>>>(
            device_256_1,
            256, 512, 1,
// "fc2.weight","fc2.bias"
            device_params["fc2.weight"],
            device_512_1,
            device_params["fc2.bias"],
// "bn2.weight","bn2.bias","bn2.running_mean","bn2.running_var"
            device_params["bn2.weight"],
            device_params["bn2.bias"],
            device_params["bn2.running_mean"],
            device_params["bn2.running_var"]
        );

        // print_device_array(device_256_1, 256, 1, false);

        // break;

        // result = conv1d_launch(result, device_params, 512, 1024, 1, "fc1.weight","fc1.bias");
        // result = batchnorm1d_launch(result, device_params, 512, 1, "bn1.weight","bn1.bias","bn1.running_mean","bn1.running_var");
        // result = relu_launch(result, 512, 1);

        // result = conv1d_launch(result, device_params, 256, 512, 1, "fc2.weight","fc2.bias");
        // result = batchnorm1d_launch(result, device_params, 256, 1, "bn2.weight","bn2.bias","bn2.running_mean","bn2.running_var");
        // result = relu_launch(result, 256, 1);


        conv1d_log_softmax_ans<<<10,1>>>(
            device_10_1,
            10, 256,    //b_col = 1
// "fc3.weight","fc3.bias"
            device_params["fc3.weight"],
            device_256_1,
            device_params["fc3.bias"]
        );

        cudaMemcpyAsync(host_10_1, device_10_1, sizeof(float)*10, cudaMemcpyDeviceToHost);
        int ans = log_softmax(host_10_1);
        // print_int(ans);
        if (ans == label)
            correct_count++;
        // print_device_array(device_10_1, 10, 1, false);
        // break;
        // print_device_float(result_num);
        // break;
    }
    
    // for (int i = 0; i < numStreams; ++i) {
    //     cudaStreamSynchronize(streams[i]);
    // }

    // // 释放流资源
    // for (int i = 0; i < numStreams; ++i) {
    //     cudaStreamDestroy(streams[i]);
    // }

    // 向主机端同步以等待所有异步调用的GPU kernel执行完毕，这句必须要有
    cudaDeviceSynchronize();

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" <<(float)correct_count / (float)total_count;

    return 0;
}

__global__ void conv1d_norm_relu(float * output,  int a_row, int a_col, int b_col, float * weight, float * input, float * bias, float * bn_weight, float * bn_bias, float * bn_running_mean, float * bn_running_var){

    const int LENGTH = 16;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int id = idx*b_col+idy;
    __shared__ float x_mean_array[LENGTH] ;
    __shared__ float x_var_array[LENGTH] ;
    __shared__ float bn1_w_val_array[LENGTH] ;
    __shared__ float bn1_b_val_array[LENGTH] ;
    __shared__ float bias_array[LENGTH];
    int threadx = threadIdx.x;
    int thready = threadIdx.y;
    if (idx < a_row){

        switch (thready){
            case 0:
                x_mean_array[threadx] = bn_running_mean[idx];
                // bias_array[threadx] = bias[idx];
                break;
            case 1:
                x_var_array[threadx] = bn_running_var[idx];
                break;
            case 2:
                bn1_w_val_array[threadx] = bn_weight[idx];
                break;
            case 3:
                bn1_b_val_array[threadx] = bn_bias[idx];
                break;
            default:
                break;
        }

        if (blockDim.y <= 4 && thready == 0)
            bias_array[threadx] = bias[idx];
        else if (blockDim.y > 4 && thready == 4)
            bias_array[threadx] = bias[idx];
    }


    __syncthreads();

    float x_mean = x_mean_array[threadx];
    float x_var = x_var_array[threadx];
    float bn1_w_val = bn1_w_val_array[threadx];
    float bn1_b_val = bn1_b_val_array[threadx];
    float bias_value = bias_array[threadx];

    if (idx<a_row && idy<b_col){
        float sum = 0;
        for (int i=0;i<a_col;i++){
            sum += weight[idx*a_col+i] * input[i*b_col+idy];
        }
        sum += bias_value;
        // float x_mean = bn_running_mean[idx];
        // float x_var = bn_running_var[idx];
        // float bn1_w_val = bn_weight[idx];
        // float bn1_b_val = bn_bias[idx];

        // float norm_res = ;
        float ans = ( (sum - x_mean) / sqrt(x_var + (1e-5)) ) * bn1_w_val + bn1_b_val;
        ans = ans > 0 ? ans : 0;
        output[id] = ans;
        // output[id] = sum;
    }
}

__global__ void conv1d_log_softmax_ans(float * output, int a_row, int a_col, float * weight, float * input, float * bias){
    int idx = blockIdx.x;
    if (idx<a_row){
        float num = 0;
        for (int i=0;i<a_col;i++)
            num += weight[idx*a_col+i] * input[i];
        num += bias[idx];
        output[idx] = num;
        }
}

void print_device_float(float * num){
    float * host_num = (float *)malloc(sizeof(float));
    cudaMemcpy(host_num, num, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << *host_num << std::endl;
    free(host_num);
}

__global__ void conv1d_norm(float * output,  int a_row, int a_col, int b_col, float * weight, float * input, float * bias, float * bn_weight, float * bn_bias, float * bn_running_mean, float * bn_running_var){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int id = idx*b_col+idy;
    if (idx<a_row && idy<b_col){
        float sum = 0;
        for (int i=0;i<a_col;i++){
            sum += weight[idx*a_col+i] * input[i*b_col+idy];
        }
        sum += bias[idx];
        float x_mean = bn_running_mean[idx];
        float x_var = bn_running_var[idx];
        float bn1_w_val = bn_weight[idx];
        float bn1_b_val = bn_bias[idx];

        float norm_res = (sum - x_mean) / sqrt(x_var + (1e-5));
        output[id] = norm_res * bn1_w_val + bn1_b_val;
        // output[id] = sum;
    }
}

__global__ void conv1d_4096_add_64_64(float * output,  int a_row, int a_col, int b_col, float * weight, float * input, float * bias){

    // __shared__ float * array_256_1 = input;

    int idx = blockIdx.x;

    if (idx<a_row){
        float sum = 0;
        for (int i=0;i<a_col;i++){
            sum += weight[idx*a_col+i] * input[i];
        }
        sum += bias[idx];
        output[idx] = sum;
        int current_row = idx / 64;
        if (idx == current_row + current_row * 64)
            output[idx] ++;
    }
}

__global__ void conv1d_22500_64_64_trans_result(float * output, int a_row, int a_col, int b_col, float * input_a, float * input_b){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx<a_row && idy<a_col){
        float sum = 0;
        for (int i=0;i<a_col;i++){
            sum += input_a[idx*a_col+i] * input_b[i*b_col+idy];
        }
        output[idy*a_row+idx] = sum;
    }

}

__global__ void conv1d_norm_relu_copy_T(float * output,  float * output_T, int a_row, int a_col, int b_col, float * weight, float * input, float * bias, float * bn_weight, float * bn_bias, float * bn_running_mean, float * bn_running_var){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx<a_row && idy<b_col){
        int id = idx*b_col+idy;
        int id_T = idy*a_row+idx;
        float sum = 0;
        for (int i=0;i<a_col;i++){
            sum += weight[idx*a_col+i] * input[i*b_col+idy];
        }
        sum += bias[idx];
        float x_mean = bn_running_mean[idx];
        float x_var = bn_running_var[idx];
        float bn1_w_val = bn_weight[idx];
        float bn1_b_val = bn_bias[idx];

        float norm_res = (sum - x_mean) / sqrt(x_var + (1e-5));
        float ans = norm_res * bn1_w_val + bn1_b_val;
        ans = ans > 0 ? ans : 0;
        output[id] = ans;
        output_T[id_T] = ans;
        // output[id] = sum;
    }
}

__global__ void max_matrix(float * output, float * input, int row, int col){

    // const int shared_memory_size = 20;




    int id = blockIdx.x;

    int tid = threadIdx.x;

    int thread_num = blockDim.x;

    __shared__ float max_val;
    if (id<row){
        int each_thread_col = col / thread_num;
        int start_col = each_thread_col * tid;
        int end_col = start_col + each_thread_col;
        for (int i=start_col;i<end_col;i++){
            float val = input[id*col+i];
            if (i==0){
                max_val = val;
            }
            else if (max_val < val){
                max_val = val;
            }
        }
        __syncthreads();
        output[id] = max_val;
    }
}

__global__ void conv1d_9(float * output, float * weight, float * input, float * bias){
    int idx = blockIdx.x ;
    if (idx<9){
        float sum = 0;
        for (int i=0;i<256;i++){
            sum += weight[idx*256+i] * input[i];
        }
        sum += bias[idx];
        output[idx] = sum;
        if (idx==0 || idx==4 || idx==8)
            output[idx] += 1;
    }
}

__global__ void matrix_array_9(float * output, float * a, float * b, int a_row, int a_col, int b_col){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx<a_row && idy<b_col){
        // int id = idx*a_col+idy;
        int id_T = idy*a_row+idx;
        float sum = 0;
        for (int i=0;i<a_col;i++){
            sum += a[idx*a_col+i] * b[i*b_col+idy];
        }
        output[id_T] = sum;
    }

}

int log_softmax(float * result){

    int max_index = 0;
    float max_value = result[0];
    for (int i=1;i<10;++i){
        if (max_value < result[i]){
            max_value = result[i];
            max_index = i;
        }
    }
    return max_index;
}

void print_device_array( float * array, int row, int col, bool from22400){
    int length = row * col;
    float * print_array = (float *)malloc (sizeof (float) * length);
    cudaMemcpy(print_array, array, sizeof (float) * length, cudaMemcpyDeviceToHost);

    // float * print_array = createHostArrayFromDeviceArray(array, sizeof (float) * row * col);
    puts("===============================Check: 0-9 ===========================");

    for (int i=0;i<10;++i){
            printf("%.8f ", print_array[i]);
    }


    
    puts("================================Check: 22400 - 22410=================");
    if (from22400)
        forloop(i, 22400, 22410)
            printf("%.8f ", print_array[i]);

    puts("================================End====================================");
    free(print_array);
}


void print_device_array_64( float * array, int row, int col, bool from22400){
    int length = row * col;
    float * print_array = (float *)malloc (sizeof (float) * length);
    cudaMemcpy(print_array, array, sizeof (float) * length, cudaMemcpyDeviceToHost);

    // float * print_array = createHostArrayFromDeviceArray(array, sizeof (float) * row * col);
    puts("===============================Check: 0-9 ===========================");

    for (int i=0;i<64;++i){
        for (int j=0;j<64;++j)
            printf("%.5f ", print_array[i*64+j]);
        puts("");
    }


    
    puts("================================Check: 22400 - 22410=================");
    if (from22400)
        forloop(i, 22400, 22410)
            print_float(print_array[i]);
    puts("================================End====================================");
    free(print_array);
}