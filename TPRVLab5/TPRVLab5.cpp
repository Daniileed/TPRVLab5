// TPRVLab5.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace cv;
using namespace std;

#define Height 1536
#define Length 2048
string filename = "C://Users//danii//source//repos//TPRVLab5//" + std::to_string(Length) + "x" + std::to_string(Height) + ".jpg";
#define output_path "C://Users//danii//source//repos//TPRVLab5//result.jpg"
#define output_path_2 "C://Users//danii//source//repos//TPRVLab5//result_OpenCL.jpg"

int Rv[Height][Length];
int Gv[Height][Length];
int Bv[Height][Length];
int Iv[Height][Length];

int MRv[Height][Length];
int MX[Height][Length];
int MY[Height][Length];

int MRMax = 1;

const int FSX[3][3] = {
{ -1,0,1 },
    {-2,0,2},
{-1,0,1 }
};

const int FSY[3][3] = {
    {-1,-2,-1},
    {0,0,0},
    {1,2,1 } };

void check_elementX(int curI, int curJ, int i, int j, int elX, int elY) {
    if ((i >= 0) && (i < Height) && (j >= 0) && (j < Length))
        MX[curI][curJ] += Iv[i][j] * FSX[elX][elY];
}

void check_elementY(int curI, int curJ, int i, int j, int elX, int elY) {
    if ((i >= 0) && (i < Height) && (j >= 0) && (j < Length))
        MY[curI][curJ] += Iv[i][j] * FSY[elX][elY];
}

const char* kernalSource = R"(
__kernel void modifyChannels(__global const int* Iv, __global const int* FSX, __global const int* FSY,__global int* MX, __global int* MY) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int Length = 2048;
    int Height = 1536;
    if (x < Length && y < Height) {
            int index = x * Height + y;
    if (index - Length - 1 >= 0 && index - Length - 1 < Length*Height)
            MX[index] += Iv[index-Length - 1] * FSX[0];
        if (index - Length + 1 >= 0 && index - Length + 1 < Length * Height)
            MX[index] += Iv[index - Length + 1] * FSX[2];
        if (index - 1 >= 0 && index - 1 < Length * Height)
            MX[index] += Iv[index - 1] * FSX[3];
        if (index + 1 >= 0 && index + 1 < Length * Height)
            MX[index] += Iv[index + 1] * FSX[5];
        if (index + Length - 1 >= 0 && index + Length - 1 < Length * Height)
            MX[index] += Iv[index + Length - 1] * FSX[6];
        if (index + Length + 1 >= 0 && index + Length + 1 < Length * Height)
            MX[index] += Iv[index + Length + 1] * FSX[8];

        if (index - Length - 1 >= 0 && index - Length - 1 < Length * Height)
            MY[index] += Iv[index - Length - 1] * FSY[0];
        if (index - Length >= 0 && index - Length < Length * Height)
            MY[index] += Iv[index - Length] * FSY[1];
        if (index - Length + 1 >= 0 && index - Length + 1 < Length * Height)
            MY[index] += Iv[index - Length + 1] * FSY[2];
        if (index + Length - 1 >= 0 && index + Length - 1 < Length * Height)
            MY[index] += Iv[index + Length - 1] * FSY[6];
        if (index + Length >= 0 && index + Length < Length * Height)
            MY[index] += Iv[index + Length] * FSY[7];
        if (index + Length + 1 >= 0 && index + Length + 1 < Length * Height)
            MY[index] += Iv[index + Length + 1] * FSY[8];
    }
}
)";

int main()
{
    Mat img_color = imread(filename);


    Mat Sobel_scale = Mat::zeros(Height, Length, CV_8UC1);

    //CPU single
    auto start_1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < Height; i++) {
        for (int j = 0; j < Length; j++) {
            Bv[i][j] = img_color.at<cv::Vec3b>(i, j)[0];
            Gv[i][j] = img_color.at<cv::Vec3b>(i, j)[1];
            Rv[i][j] = img_color.at<cv::Vec3b>(i, j)[2];
            Iv[i][j] = floor((Rv[i][j] + Gv[i][j] + Bv[i][j]) / 3);
        }
    }

    for (int i = 0; i < Height; i++) {
        for (int j = 0; j < Length; j++) {
            check_elementX(i, j, i - 1, j - 1, 0, 0);
            check_elementX(i, j, i, j - 1, 1, 0);
            check_elementX(i, j, i + 1, j - 1, 2, 0);

            check_elementX(i, j, i - 1, j + 1, 0, 2);
            check_elementX(i, j, i, j + 1, 1, 2);
            check_elementX(i, j, i + 1, j + 1, 2, 2);

            check_elementY(i, j, i - 1, j - 1, 0, 0);
            check_elementY(i, j, i - 1, j, 0, 1);
            check_elementY(i, j, i - 1, j + 1, 0, 2);

            check_elementY(i, j, i + 1, j - 1, 2, 0);
            check_elementY(i, j, i + 1, j, 2, 1);
            check_elementY(i, j, i + 1, j + 1, 2, 2);

        }
    }

    for (int i = 0; i < Height; i++) {
        for (int j = 0; j < Length; j++) {
            MRv[i][j] = floor(sqrt(pow(MX[i][j], 2) + pow(MY[i][j], 2)));
        }
    }

    for (int i = 0; i < Height; i++) {
        for (int j = 0; j < Length; j++) {
            if (MRv[i][j] >= MRMax)
                MRMax = MRv[i][j];
        }
    }

    for (int i = 0; i < Height; i++) {
        for (int j = 0; j < Length; j++) {
            MRv[i][j] = MRv[i][j] * 255 / MRMax;
        }
    }


    for (int i = 0; i < Height; i++) {
        for (int j = 0; j < Length; j++) {
            Sobel_scale.at<uchar>(i, j) = MRv[i][j];
        }
    }


    auto end_1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> dur1 = (end_1 - start_1);
    std::cout << "Procesor Time: " << dur1.count() << " seconds\n\n";

  //  namedWindow("Sobel_scale", WINDOW_NORMAL);
   // imshow("Sobel_scale", Sobel_scale);
   // waitKey(0);
   // destroyAllWindows();

    imwrite(output_path, Sobel_scale);

    for (int i = 0; i < Height; i++) {
        for (int j = 0; j < Length; j++) {
            Iv[i][j] = 0;
            MX[i][j] = 0;
            MY[i][j] = 0;
            MRv[i][j] = 0;
        }
    }
    
    //GPUOpenCL

    cl_platform_id platform_id = nullptr;
    cl_device_id device_id = nullptr;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

    cl_context context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &ret);
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);

    // Создание буферов 
    cl_mem IvCL = clCreateBuffer(context, CL_MEM_READ_ONLY, Height * Length * sizeof(int), nullptr, &ret);
    cl_mem FSXCL = clCreateBuffer(context, CL_MEM_READ_ONLY, 9*sizeof(int), nullptr, &ret);
    cl_mem FSYCL = clCreateBuffer(context, CL_MEM_READ_ONLY, 9* sizeof(int), nullptr, &ret);
    cl_mem MXCL = clCreateBuffer(context, CL_MEM_WRITE_ONLY, Height * Length * sizeof(int), nullptr, &ret);
    cl_mem MYCL = clCreateBuffer(context, CL_MEM_WRITE_ONLY, Height * Length * sizeof(int), nullptr, &ret);

    for (int i = 0; i < Height; i++) {
        for (int j = 0; j < Length; j++) {
            Bv[i][j] = img_color.at<cv::Vec3b>(i, j)[0];
            Gv[i][j] = img_color.at<cv::Vec3b>(i, j)[1];
            Rv[i][j] = img_color.at<cv::Vec3b>(i, j)[2];
            Iv[i][j] = floor((Rv[i][j] + Gv[i][j] + Bv[i][j]) / 3);
        }
    }

    // Копирование данных матриц в буферы
    ret = clEnqueueWriteBuffer(command_queue, IvCL, CL_TRUE, 0, Height * Length * sizeof(int),Iv , 0, nullptr, nullptr);
    ret = clEnqueueWriteBuffer(command_queue, FSXCL, CL_TRUE, 0, 9 * sizeof(int), FSX, 0, nullptr, nullptr);
    ret = clEnqueueWriteBuffer(command_queue, FSYCL, CL_TRUE, 0, 9 * sizeof(int), FSY, 0, nullptr, nullptr);


    auto start_2 = std::chrono::high_resolution_clock::now();

    // Подготовка и компиляция OpenCL ядра
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernalSource, nullptr, &ret);
    ret = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);

    // Создание ядра
    cl_kernel kernel = clCreateKernel(program, "modifyChannels", &ret);

    // Установка аргументов ядра
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&IvCL);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&FSXCL);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&FSYCL);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&MXCL);
    ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&MYCL);

    // Выполнение ядра
    size_t global_item_size[2] = { Height, Length};
    size_t local_item_size[2] = { 32, 32 };
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, nullptr, global_item_size, local_item_size, 0, nullptr, nullptr);

    ret = clEnqueueReadBuffer(command_queue, MXCL, CL_TRUE, 0, Height * Length * sizeof(int), MX, 0, nullptr, nullptr);
    ret = clEnqueueReadBuffer(command_queue, MYCL, CL_TRUE, 0, Height * Length * sizeof(int), MY, 0, nullptr, nullptr);

    for (int i = 0; i < Height; i++) {
        for (int j = 0; j < Length; j++) {
            MRv[i][j] = floor(sqrt(pow(MX[i][j], 2) + pow(MY[i][j], 2)));
        }
    }

    for (int i = 0; i < Height; i++) {
        for (int j = 0; j < Length; j++) {
            if (MRv[i][j] >= MRMax)
                MRMax = MRv[i][j];
        }
    }

    for (int i = 0; i < Height; i++) {
        for (int j = 0; j < Length; j++) {
            MRv[i][j] = MRv[i][j] * 255 / MRMax;
        }
    }


    for (int i = 0; i < Height; i++) {
        for (int j = 0; j < Length; j++) {
            Sobel_scale.at<uchar>(i, j) = MRv[i][j];
        }
    }

    auto end_2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> dur2 = (end_2 - start_2);
    std::cout << "OpenCL Time: " << dur2.count() << " seconds\n\n";

   // namedWindow("Sobel_scale", WINDOW_NORMAL);
   // imshow("Sobel_scale", Sobel_scale);
   // waitKey(0);
   // destroyAllWindows();

    imwrite(output_path_2, Sobel_scale);



    // Очистка ресурсов
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(IvCL);
    ret = clReleaseMemObject(FSXCL);
    ret = clReleaseMemObject(FSYCL);
    ret = clReleaseMemObject(MXCL);
    ret = clReleaseMemObject(MYCL);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
}

