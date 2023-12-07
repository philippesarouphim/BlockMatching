#include <string>

#include <opencv2/opencv.hpp>

using namespace std;

#define CODEC cv::VideoWriter::fourcc('m', 'p', '4', 'v')

class VideoWriter{

    public:
        VideoWriter(string filename, cv::Size frameSize, int fps);
        ~VideoWriter();

        void writeFrame(cv::Mat frame);
    
    private:
        cv::VideoWriter* writer;
};