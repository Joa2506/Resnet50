#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvOnnxConfig.h>

#include </usr/src/tensorrt/samples/common/buffers.h>
#include <memory>

#include "Logger.hpp"
#include "common.hpp"
#include <cuda_runtime.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/cuda.hpp>
#include "opencv2/imgproc.hpp"

using namespace std;

#define RESNET "/usr/src/tensorrt/data/resnet50/ResNet50.onnx"
#define RESNET_CLASSNAMES "/usr/src/tensorrt/data/resnet50/class_labels.txt"



class Engine
{
    private:
        //Some utility methods
        bool engineExists(string filename);
        bool fileExists(string filename);

        string serializeEngineName(const Configurations& config);

        //This function takes in the frame and resizes it to fit the Resnet50 model. It also normalizes the image
        bool resizeAndNormalize(cv::Mat frame, float* gpu_input, const nvinfer1::Dims& dims);
        size_t getSizeByDim(const nvinfer1::Dims& dims);
        //This function calculates the softmax to find the most probable classes for the image
        bool calculateProbability(float* gpu_output, const nvinfer1::Dims& dims, int batchSize);

        //Gets the class name from the file that is associated with the image
        vector<std::string> getClassNames(const std::string& imagenet_classes);

        //Engine for inference
        shared_ptr<nvinfer1::ICudaEngine> m_engine;
        //Context for running inference
        shared_ptr<nvinfer1::IExecutionContext> m_context;

        //ILogger
        Logger m_logger;

        const Configurations& m_config;

        //Cuda stream incase asynchronous
        cudaStream_t m_cudaStream;

        //Name of engine stored on disk
        string m_engineName;

        const char * m_inputName;
        const char * m_outputName;

        Dims m_inputDims;
        Dims m_outputDims;

        //NCHW
        int32_t m_batchSize; //Also known as N
        int32_t m_inputChannel;
        int32_t m_inputHeight;
        int32_t m_inputWidth;

        
    public:
        bool build(string onnxfilename);
        bool loadNetwork();
        bool inference(cv::Mat &img, int batchSize);

        Engine(const Configurations& config);
        ~Engine();


};