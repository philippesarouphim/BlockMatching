#include <opencv2/opencv.hpp>

#include "video_reader.h"
#include "video_writer.h"

int main(int argc, char** argv){
    VideoReader reader("data/train.mov");

    cv::Size frameSize = reader.getFrameSize();
    int fps = reader.getFrameRate();

    VideoWriter writer("out/data/train.mp4", frameSize, fps);

    for(int i = 0; i < 100; i++){
        cv::Mat* frame = reader.nextFrame();
        writer.writeFrame(*frame);
    }

}