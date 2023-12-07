#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <vector>

#include <opencv2/opencv.hpp>

#include "video_reader.h"
#include "video_writer.h"

#define INDEX_TO_BLOCK_X(i, frame_width, block_size) ((i) % (frame_width / block_size))
#define INDEX_TO_BLOCK_Y(i, frame_width, block_size) ((i) / (frame_width / block_size))
#define INDEX_TO_BLOCK(i, frame_width, block_size) INDEX_TO_BLOCK_X(i, frame_width, block_size), INDEX_TO_BLOCK_Y(i, frame_width, block_size)

#define XYZ_TO_I(x, y, z, frame_width, frame_height, frame_channels) ((x) * (frame_channels) + (y) * (frame_channels) * (frame_width) + (z))

__device__ inline bool checkBlockInBounds(int block_x, int block_y, int block_size, int width, int height){
    return block_x >= 0 &&
           block_x + block_size < width &&
           block_y >= 0 &&
           block_y + block_size < height;
}

__device__ double mean_average_difference(int frame_width, int frame_height, int frame_channels, int block1_x, int block1_y, int block2_x, int block2_y, int block_size, uint8_t* frame1, uint8_t* frame2){
    double difference = 0;
    for(int c = 0; c < frame_channels; c++){
        double channel_difference = 0;
        for(int i = 0; i < block_size; i++){
            for(int j = 0; j < block_size; j++){
                channel_difference += abs(
                    (int)frame1[XYZ_TO_I(block1_x + j, block1_y + i, c, frame_width, frame_height, frame_channels)] -
                    (int)frame2[XYZ_TO_I(block2_x + j, block2_y + i, c, frame_width, frame_height, frame_channels)]
                );
            }
        }
        difference += channel_difference / (block_size * block_size);
    }
    difference /= frame_channels;
    return difference;
}

__device__ void find_match(int frame_width, int frame_height, int frame_channels, int block_x, int block_y, int block_size, int step_size, uint8_t* frame1, uint8_t* frame2, int* vector){
    if(step_size == 0){
        vector[0] = block_x;
        vector[1] = block_y;
        return;
    }

    int min_x = block_x;
    int min_y = block_y;
    double min_diff = mean_average_difference(
        frame_width,
        frame_height,
        frame_channels,
        block_x,
        block_y,
        block_x,
        block_y,
        block_size,
        frame1,
        frame2
    );
    
    for(int i = -step_size; i <= step_size; i += step_size){
        for(int j = -step_size; j <= step_size; j += step_size){
            if(i == 0 && j == 0) continue;
            if(!checkBlockInBounds(block_x + j, block_y + i, block_size, frame_width, frame_height)) continue;

            double difference = mean_average_difference(
                frame_width,
                frame_height,
                frame_channels,
                block_x,
                block_y,
                block_x + j,
                block_y + i,
                block_size,
                frame1,
                frame2
            );

            if(difference < min_diff){
                min_diff = difference;
                min_x = block_x + j;
                min_y = block_y + i;
            }
        }
    }

    find_match(frame_width, frame_height, frame_channels, min_x, min_y, block_size, step_size / 2, frame1, frame2, vector);
}

__global__ void find_match(int frame_width, int frame_height, int frame_channels, int block_size, int step_size, uint8_t* frame1, uint8_t* frame2, int* vectors){
    int start = ((float)frame_width / block_size) * ((float)frame_height / block_size) * (threadIdx.x + blockIdx.x * blockDim.x) / (blockDim.x * gridDim.x);
    int end = ((float)frame_width / block_size) * ((float)frame_height / block_size) * (threadIdx.x + 1 + blockIdx.x * blockDim.x) / (blockDim.x * gridDim.x);

    for(int i = start; i < end; i++){
        find_match(
            frame_width,
            frame_height,
            frame_channels,
            INDEX_TO_BLOCK_X(i, frame_width, block_size) * block_size,
            INDEX_TO_BLOCK_Y(i, frame_width, block_size) * block_size,
            block_size,
            step_size,
            frame1,
            frame2,
            &vectors[i * 2]
        );
    }
}

int main(int argc, char** argv){
    VideoReader reader("data/mario2.mp4");

    cv::Size frameSize = reader.getFrameSize();
    int fps = reader.getFrameRate();

    std::cout << "(" << frameSize.width << ", " << frameSize.height << ")" << std::endl;

    VideoWriter writer("out/data/mario2.mp4", frameSize, fps);

    int block_size = 8;

    int vectors_size = (frameSize.width / block_size) * (frameSize.height / block_size) * 2;
    int* vectors = (int*) malloc(sizeof(int) * vectors_size);
    int* d_vectors;
    cudaMalloc(&d_vectors, sizeof(int) * vectors_size);

    cv::Mat* frame = reader.nextFrame();
    cv::Mat* frameNext = reader.nextFrame();

    int frame_width = frameSize.width;
    int frame_height = frameSize.height;
    int frame_channels = frame->channels();
    uint8_t* d_frame;
    uint8_t* d_frameNext;
    cudaMalloc(&d_frame, sizeof(uint8_t) * frame_width * frame_height * frame_channels);
    cudaMalloc(&d_frameNext, sizeof(uint8_t) * frame_width * frame_height * frame_channels);

    int idx = 0;
    while(frameNext != nullptr && idx < 1750){
        std::cout << idx << std::endl;
        
        cudaMemcpy(d_frame, frame->isContinuous() ? frame->data : frame->clone().data, sizeof(uint8_t) * frame_width * frame_height * frame_channels, cudaMemcpyHostToDevice);
        cudaMemcpy(d_frameNext, frameNext->isContinuous() ? frameNext->data : frameNext->clone().data, sizeof(uint8_t) * frame_width * frame_height * frame_channels, cudaMemcpyHostToDevice);

        find_match<<<16, 128>>>(frame_width, frame_height, frame_channels, block_size, 3, d_frame, d_frameNext, d_vectors);
        
        cudaDeviceSynchronize();

        cudaMemcpy(vectors, d_vectors, sizeof(int) * vectors_size, cudaMemcpyDeviceToHost);

        for(int i = 0; i < vectors_size / 2; i++){
            // std::cout << "(i: " << i << ", " << "(" << INDEX_TO_BLOCK_X(i, frame_width, block_size) * block_size << ", " << INDEX_TO_BLOCK_Y(i, frame_width, block_size) * block_size  << "), " << "(" << vectors[i * 2] << ", " << vectors[i * 2 + 1] << "))" << std::endl;
            cv::arrowedLine(*frame,
                cv::Point(INDEX_TO_BLOCK_X(i, frameSize.width, block_size) * block_size, INDEX_TO_BLOCK_Y(i, frameSize.width, block_size) * block_size),
                cv::Point(vectors[i * 2], vectors[i * 2 + 1]),
                1
            );
        }
        writer.writeFrame(*frame);

        free(frame);
        frame = frameNext;
        frameNext = reader.nextFrame();

        idx++;
    }

}