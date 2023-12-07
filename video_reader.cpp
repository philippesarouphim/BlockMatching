#include "video_reader.h"

VideoReader::VideoReader(string filename){
    this->cap = new cv::VideoCapture(filename);

    if (!this->cap->isOpened()){
        cerr << "Error opening video file" << endl;
        exit(1);
    }
}

VideoReader::~VideoReader(){
    this->cap->release();
    delete this->cap;
}

cv::Mat* VideoReader::nextFrame(){
    if(!this->cap->isOpened()) return nullptr;

    cv::Mat* frame = new cv::Mat();
    bool readSuccess = this->cap->read(*frame);

    if(!readSuccess) return nullptr;
    return frame;
}

cv::Size VideoReader::getFrameSize(){
    return cv::Size(
        static_cast<int>(this->cap->get(cv::CAP_PROP_FRAME_WIDTH)),
        static_cast<int>(this->cap->get(cv::CAP_PROP_FRAME_HEIGHT))
    );
}

int VideoReader::getFrameRate(){
    return static_cast<int>(this->cap->get(cv::CAP_PROP_FPS));
}