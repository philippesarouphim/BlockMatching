#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <vector>

#include <opencv2/opencv.hpp>

#include "gputimer.h"
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

__device__ void find_match(int frame_width, int frame_height, int frame_channels, int block_x, int block_y, int search_loc_x, int search_loc_y, int block_size, int step_size, uint8_t* frame1, uint8_t* frame2, int* vector){
    // Base case, return block center as vector
    if(step_size == 0){
        vector[0] = search_loc_x;
        vector[1] = search_loc_y;
        return;
    }

    // Set center block as best match
    int min_x = search_loc_x;
    int min_y = search_loc_y;
    double min_diff = mean_average_difference(
        frame_width,
        frame_height,
        frame_channels,
        block_x,
        block_y,
        search_loc_x,
        search_loc_y,
        block_size,
        frame1,
        frame2
    );
    
    // For each block, step_size away in each direction, check for similarity, and update best match
    // if needed.
    for(int i = -step_size; i <= step_size; i += step_size){
        for(int j = -step_size; j <= step_size; j += step_size){
            if(i == 0 && j == 0) continue;
            if(!checkBlockInBounds(search_loc_x + j, search_loc_y + i, block_size, frame_width, frame_height)) continue;

            double difference = mean_average_difference(
                frame_width,
                frame_height,
                frame_channels,
                block_x,
                block_y,
                search_loc_x + j,
                search_loc_y + i,
                block_size,
                frame1,
                frame2
            );

            if(difference < min_diff || (difference == min_diff && (abs(j) + abs(i)) < (abs(min_x - block_x) + abs(min_y - block_y)))){
                min_diff = difference;
                min_x = search_loc_x + j;
                min_y = search_loc_y + i;
            }
        }
    }

    // Call function recursively, with best match as block center, and half the step size
    find_match(frame_width, frame_height, frame_channels, block_x, block_y, min_x, min_y, block_size, step_size / 2, frame1, frame2, vector);
}

__global__ void find_match(int frame_width, int frame_height, int frame_channels, int block_size, int step_size, uint8_t* frame1, uint8_t* frame2, int* vectors){
    // Determine which blocks have to be processed by this thread
    int start = ((float)frame_width / block_size) * ((float)frame_height / block_size) * (threadIdx.x + blockIdx.x * blockDim.x) / (blockDim.x * gridDim.x);
    int end = ((float)frame_width / block_size) * ((float)frame_height / block_size) * (threadIdx.x + 1 + blockIdx.x * blockDim.x) / (blockDim.x * gridDim.x);

    // Find matches for each block
    for(int i = start; i < end; i++){
        find_match(
            frame_width,
            frame_height,
            frame_channels,
            INDEX_TO_BLOCK_X(i, frame_width, block_size) * block_size,
            INDEX_TO_BLOCK_Y(i, frame_width, block_size) * block_size,
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
    // Check if correct number of inputs, or print correct usage and exit
    if(argc != 8){
        std::cout << "Usage:" << std::endl;
        std::cout << "tss <video_in_path> <video_out_path> <block_size> <step_size> <num_blocks> <num_threads> <frames_limit>" << std::endl;
        exit(1);
    }

    // Parse input
    string video_in_path = argv[1];
    string video_out_path = argv[2];
    int block_size = atoi(argv[3]);
    int step_size = atoi(argv[4]);
    int num_blocks = atoi(argv[5]);
    int num_threads = atoi(argv[6]);
    int frames_limit = atoi(argv[7]);

    // Open video reader, and get basic information
    VideoReader reader(video_in_path);
    cv::Size frameSize = reader.getFrameSize();
    int fps = reader.getFrameRate();

    // Open video writer
    VideoWriter writer(video_out_path, frameSize, fps);

    // Create motion vectors array on CPU and GPU
    int vectors_size = (frameSize.width / block_size) * (frameSize.height / block_size) * 2;
    int* vectors = (int*) malloc(sizeof(int) * vectors_size);
    int* d_vectors;
    cudaMalloc(&d_vectors, sizeof(int) * vectors_size);

    // Get the first two frames of the video
    cv::Mat* frame = reader.nextFrame();
    cv::Mat* frameNext = reader.nextFrame();

    // Allocate memory for two frames to the GPU
    int frame_width = frameSize.width;
    int frame_height = frameSize.height;
    int frame_channels = frame->channels();
    uint8_t* d_frame;
    uint8_t* d_frameNext;
    cudaMalloc(&d_frame, sizeof(uint8_t) * frame_width * frame_height * frame_channels);
    cudaMalloc(&d_frameNext, sizeof(uint8_t) * frame_width * frame_height * frame_channels);

    double timeElapsed;
    int idx = 0;
    while(frameNext != nullptr && idx < frames_limit){
        // Upload the two current frames to GPU
        cudaMemcpy(d_frame, frame->isContinuous() ? frame->data : frame->clone().data, sizeof(uint8_t) * frame_width * frame_height * frame_channels, cudaMemcpyHostToDevice);
        cudaMemcpy(d_frameNext, frameNext->isContinuous() ? frameNext->data : frameNext->clone().data, sizeof(uint8_t) * frame_width * frame_height * frame_channels, cudaMemcpyHostToDevice);

        // Start timer
        GpuTimer timer;
        timer.Start();

        // Execute kernel call and synchronize
        find_match<<<num_blocks, num_threads>>>(frame_width, frame_height, frame_channels, block_size, step_size, d_frame, d_frameNext, d_vectors);
        cudaDeviceSynchronize();

        // End timer and record time
        timer.Stop();
        timeElapsed += timer.Elapsed();

        // Fetch the resulting motion vectors from the GPU
        cudaMemcpy(vectors, d_vectors, sizeof(int) * vectors_size, cudaMemcpyDeviceToHost);

        // Display arrows on the frame and write it to output video
        for(int i = 0; i < vectors_size / 2; i++){
            cv::arrowedLine(*frame,
                cv::Point(INDEX_TO_BLOCK_X(i, frameSize.width, block_size) * block_size, INDEX_TO_BLOCK_Y(i, frameSize.width, block_size) * block_size),
                cv::Point(vectors[i * 2], vectors[i * 2 + 1]),
                1
            );
        }
        writer.writeFrame(*frame);

        // Fetch next two frames
        free(frame);
        frame = frameNext;
        frameNext = reader.nextFrame();

        // Increment frame index
        idx++;
    }

    // Average time over frames and display
    std::cout << "Time per frame: " << (timeElapsed / idx) << "ms" << std::endl;

    // Free pointers on GPU and CPU
    free(frame);
    free(frameNext);
    free(vectors);
    cudaFree(d_frame);
    cudaFree(d_frameNext);
    cudaFree(d_vectors);
}