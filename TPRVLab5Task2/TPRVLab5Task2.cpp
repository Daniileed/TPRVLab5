// TPRVLab5Task2.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <stdexcept>
#include <chrono>

#define MATRIX_SIZE 2048

// Функция для инициализации матрицы
void init_matrix(std::vector<cl_int>& matrix)
{
    for (auto& element : matrix) {
        element = rand() % 10;
    }
}

// Функция для вывода матрицы
void print_matrix(const std::vector<cl_int>& matrix)
{
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            std::cout << matrix[i * MATRIX_SIZE + j] << " ";
        }
        std::cout << std::endl;
    }
}

// Функция умножения матриц на CPU
void cpu_matrix_multiply(const std::vector<cl_int>& A, const std::vector<cl_int>& B, std::vector<cl_int>& C)
{
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            cl_int sum = 0;
            for (int k = 0; k < MATRIX_SIZE; ++k) {
                sum += A[i * MATRIX_SIZE + k] * B[k * MATRIX_SIZE + j];
            }
            C[i * MATRIX_SIZE + j] = sum;
        }
    }
}

const char* kernelSource = R"(
    __kernel void matrix_multiply(__global int* A, __global int* B, __global int* C) {
        int row = get_global_id(0);
        int col = get_global_id(1);
        int value = 0;
        for (int k = 0; k < 2048 ; ++k) {
            value += A[row * 2048 + k] * B[k * 2048 + col];
        }
        C[row * 2048 + col] = value;
    }
    )";

int main()
{
    setlocale(LC_ALL, "Ru");
    // Инициализация матриц



        std::vector<cl_int> A(MATRIX_SIZE * MATRIX_SIZE);
        std::vector<cl_int> B(MATRIX_SIZE * MATRIX_SIZE);
        std::vector<cl_int> C_gpu(MATRIX_SIZE * MATRIX_SIZE);
        std::vector<cl_int> C_cpu(MATRIX_SIZE * MATRIX_SIZE);

        init_matrix(A);
        init_matrix(B);

        auto start_gpu = std::chrono::high_resolution_clock::now();

        cl_platform_id platform_id = nullptr;
        cl_device_id device_id = nullptr;
        cl_uint ret_num_devices;
        cl_uint ret_num_platforms;
        cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

        cl_context context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &ret);
        cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);

        // Создание буферов для матриц
        cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, MATRIX_SIZE * MATRIX_SIZE * sizeof(cl_int), nullptr, &ret);
        cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, MATRIX_SIZE * MATRIX_SIZE * sizeof(cl_int), nullptr, &ret);
        cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, MATRIX_SIZE * MATRIX_SIZE * sizeof(cl_int), nullptr, &ret);

        // Копирование данных матриц в буферы
        ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(cl_int), A.data(), 0, nullptr, nullptr);
        ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(cl_int), B.data(), 0, nullptr, nullptr);

        // Подготовка и компиляция OpenCL ядра
        cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, nullptr, &ret);
        ret = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);

        // Создание ядра
        cl_kernel kernel = clCreateKernel(program, "matrix_multiply", &ret);

        // Установка аргументов ядра
        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_mem_obj);
        ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b_mem_obj);
        ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&c_mem_obj);

        // Выполнение ядра
        size_t global_item_size[2] = { MATRIX_SIZE, MATRIX_SIZE };
        size_t local_item_size[2] = { 32, 32 };
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, nullptr, global_item_size, local_item_size, 0, nullptr, nullptr);

        // Чтение результата обратно в память CPU
        ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(cl_int), C_gpu.data(), 0, nullptr, nullptr);

        // Очистка ресурсов
        ret = clFlush(command_queue);
        ret = clFinish(command_queue);
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(program);
        ret = clReleaseMemObject(a_mem_obj);
        ret = clReleaseMemObject(b_mem_obj);
        ret = clReleaseMemObject(c_mem_obj);
        ret = clReleaseCommandQueue(command_queue);
        ret = clReleaseContext(context);

        auto stop_gpu = std::chrono::high_resolution_clock::now();

        std::cout << "Gpu multiplication time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_gpu - start_gpu).count() << " ms\n";

        auto start_cpu = std::chrono::high_resolution_clock::now();

        cpu_matrix_multiply(A, B, C_cpu);

        auto stop_cpu = std::chrono::high_resolution_clock::now();

        std::cout << "Cpu multiplication time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_cpu - start_cpu).count() << " ms\n";


        bool equal = true;
        for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i) {
            if (C_gpu[i] != C_cpu[i]) {
                equal = false;
                break;
            }
        }

        if (equal) {
            std::cout << "Matrices are equal\n";
        }
        else {
            std::cout << "Matrices aren't equal\n";
        }



    return 0;
}