#include <string>
#include <opencv2/opencv.hpp>

using namespace std;

class VideoReader{

    public:
        VideoReader(string filename);
        ~VideoReader();

        cv::Mat* nextFrame();
        cv::Size getFrameSize();
        int getFrameRate();

    private:
        cv::VideoCapture* cap;
};
