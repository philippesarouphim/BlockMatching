#include <cmath>
#include <vector>

#include <opencv2/opencv.hpp>

#include "cost_functions.h"
#include "utils.h"
#include "video_reader.h"
#include "video_writer.h"

std::pair<int, int> find_match(int block_x, int block_y, int block_size, int step_size, cv::Mat* frame1, cv::Mat* frame2){
    if(step_size == 0) return std::pair<int, int>(block_x, block_y);

    std::pair<int, int> min_center(block_x, block_y);
    double min_diff = mean_average_difference(
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
            if(!checkBlockInBounds(block_x + i, block_y + j, block_size, frame1->cols, frame1->rows)) continue;

            double difference = mean_average_difference(
                block_x,
                block_y ,
                block_x + i,
                block_y + j,
                block_size,
                frame1,
                frame2
            );

            if(difference < min_diff){
                min_diff = difference;
                min_center.first = block_x + i;
                min_center.second = block_y + j;
            }
        }
    }

    return find_match(min_center.first, min_center.second, block_size, step_size / 2, frame1, frame2);
}

int main(int argc, char** argv){
    VideoReader reader("data/train.mov");

    cv::Size frameSize = reader.getFrameSize();
    int fps = reader.getFrameRate();

    std::cout << "(" << frameSize.width << ", " << frameSize.height << ")" << std::endl;

    VideoWriter writer("out/data/train.mp4", frameSize, fps);

    int block_size = 16;

    cv::Mat* frame = reader.nextFrame();
    cv::Mat* nextFrame = reader.nextFrame();

    int idx = 0;
    while(nextFrame != nullptr && idx < 5){
        std::cout << idx << std::endl;
        std::vector<std::pair<int, int>>* vectors = new std::vector<std::pair<int, int>>();
        for(int y = 0; y < frameSize.height / block_size; y++){
            for(int x = 0; x < frameSize.width / block_size; x++){
                std::pair<int, int> match = find_match(x * block_size, y * block_size, block_size, 4, frame, nextFrame);
                vectors->push_back(match);
            }
        }

        for(int i = 0; i < vectors->size(); i++){
            // std::cout << "(i: " << i << ", " << "(" << i % (frameSize.width / block_size) * block_size << ", " << i * block_size * block_size / frameSize.width << "), " << "(" << vectors->at(i).first << ", " << vectors->at(i).second << "))" << std::endl;
            cv::arrowedLine(*frame, cv::Point(i % (frameSize.width / block_size) * block_size, i / (frameSize.width / block_size) * block_size), cv::Point(vectors->at(i).first, vectors->at(i).second), 1);
        }
        writer.writeFrame(*frame);

        frame = nextFrame;
        nextFrame = reader.nextFrame();

        idx++;
    }

}