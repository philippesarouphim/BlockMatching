#include "video_writer.h"

VideoWriter::VideoWriter(string filename, cv::Size frameSize, int fps){
    this->writer = new cv::VideoWriter(filename, CODEC, fps, frameSize);
}

VideoWriter::~VideoWriter(){
    writer->release();
    free(writer);
}

void VideoWriter::writeFrame(cv::Mat frame){
    writer->write(frame);
}