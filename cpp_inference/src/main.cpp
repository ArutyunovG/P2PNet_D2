#include <iostream>
#include <algorithm>
#include <iterator>

#include <opencv2/opencv.hpp>

struct Point
{
    float x;
    float y;
    float conf;
};

std::ostream& operator<< (std::ostream& os, const Point& point)
{
    os << point.conf << " (" << point.x << ", " << point.y << ")" << std::endl;
    return os;
}


int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cout << "Usage: p2p_demo <model.onnx> <input_image>" << std::endl;
        return 0;
    }

    const std::string& modelPath = argv[1];
    const std::string& imagePath = argv[2];

    // confidenceThreshold can be read from config
    const float confidenceThreshold = 0.5f;

    const cv::Mat& readImage = cv::imread(imagePath);

    const int sizeDivisibility = 128;

    const int inputWidth = (readImage.cols / sizeDivisibility + 1) * sizeDivisibility;
    const int inputHeight = (readImage.cols / sizeDivisibility + 1) * sizeDivisibility;

    cv::Mat inputImage;
    cv::resize(readImage, inputImage, cv::Size(inputWidth, inputHeight));

    cv::dnn::Net net = cv::dnn::readNet(modelPath); 
    const cv::Mat& inputBlob = cv::dnn::blobFromImage(inputImage);

    net.setInput(inputBlob);
    std::vector<cv::Mat> outputBlobs;
    const std::vector<cv::String> outputBlobNames{ "regression", "classification" };

    net.forward(outputBlobs, outputBlobNames);

    const cv::Mat& classificationBlob = outputBlobs[1];

    // rows_, lines_ can be read from config
    const int rows_ = 2;
    const int lines_ = 2;

    const int numAnchors = rows_ * lines_;
    const cv::Mat& regressionBlob = outputBlobs[0];
    const float numberOfFMPoints  = static_cast<float>(regressionBlob.total() / numAnchors / 2);
    const float stride = std::sqrt((inputWidth * inputHeight)  / numberOfFMPoints);
    const float rowStep = stride / rows_;
    const float lineStep = stride / lines_;

    std::vector<float> anchorX, anchorY;
    std::generate_n(std::back_inserter(anchorX), rows_,
                    [rowStep, stride, n = 1.0f]() mutable  { return (n++ - 0.5f) * rowStep - stride / 2.0f; });
    std::generate_n(std::back_inserter(anchorY),  lines_,
                    [lineStep, stride, n = 1.0f]() mutable { return (n++ - 0.5f) * lineStep - stride / 2.0f; });


    const int numClasses = classificationBlob.total() / classificationBlob.size().area();
    const int fgClassIdx = 1;
    const int fmW = static_cast<int>(inputWidth / stride + 0.5f);
    const int fmH = static_cast<int>(inputHeight / stride + 0.5f);
    std::vector<Point> outputPoints;
    for (int y = 0; y < fmH; ++y) 
    {
        for (int x = 0; x < fmW; ++x)
        {
            for (int anchorIdx = 0; anchorIdx < numAnchors; ++anchorIdx)
            {
                const int fmIdx = (y * fmW + x) * numAnchors + anchorIdx;
                const float score = *(reinterpret_cast<float*>(classificationBlob.data) + numClasses * fmIdx + fgClassIdx);

                if (score < confidenceThreshold)
                {
                    continue;
                }

                const float offsetX = (static_cast<float>(x) + 0.5f) * stride + anchorX[anchorIdx];
                const float offsetY = (static_cast<float>(y) + 0.5f) * stride + anchorY[anchorIdx];

                outputPoints.push_back(
                    {
                        offsetX + *(reinterpret_cast<float*>(regressionBlob.data) + 2 * fmIdx + 0),
                        offsetY + *(reinterpret_cast<float*>(regressionBlob.data) + 2 * fmIdx + 1),
                        score
                    }
                );
            }
        }
    }

    std::copy(outputPoints.begin(), outputPoints.end(), std::ostream_iterator<Point>(std::cout, " "));
    
    cv::Mat dispImg = inputImage.clone();
    for (const Point& pt: outputPoints)
    {
        cv::circle(dispImg, cv::Point(pt.x, pt.y), 2, {0, 0, 255}, -1);
    }

    cv::imshow("Points", dispImg);
    cv::waitKey();

    return 0;
}
