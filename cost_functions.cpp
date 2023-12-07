#include "cost_functions.h"

double mean_average_difference(int block1_x, int block1_y, int block2_x, int block2_y, int block_size, cv::Mat* frame1, cv::Mat* frame2){
    int channels = frame1->channels();

    double difference = 0;
    for(int c = 0; c < channels; c++){
        double channel_difference = 0;
        for(int i = 0; i < block_size; i++){
            for(int j = 0; j < block_size; j++){
                channel_difference += abs(
                    frame1->at<cv::Vec3b>(block1_y + j, block1_x + i)[c] -
                    frame2->at<cv::Vec3b>(block2_y + j, block2_x + i)[c]
                );
            }
        }
        difference += channel_difference / (block_size * block_size);
    }
    difference /= channels;
    return difference;
}
