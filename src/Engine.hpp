#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvOnnxConfig.h>

#include </usr/src/tensorrt/samples/common/buffers.h>
#include <memory>

#include "Logger.hpp"
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
#define RESNET_CLASSNAMES "/usr/src/tensorrt/data/resnet50/"

struct Configurations {
    //Using 16 point floats for inference
    bool FP16 = false;
    //Using int8
    bool INT8 = false;
    //Batch size for optimization
    vector<int32_t> optBatchSize;
    // Maximum allowed batch size
    int32_t maxBatchSize = 16;
    //Max GPU memory allowed for the model.
    long int maxWorkspaceSize = 400000000;//
    //GPU device index number, might be useful for more Tegras in the future
    int deviceIndex = 0;
    // DLA
    int dlaCore = 0;

};

class Engine
{
    private:
        //Some utility methods
        bool engineExists(string filename);
        bool fileExists(string filename);

        string serializeEngineName(const Configurations& config);

        bool processInput();
        bool processOutput();

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
        bool inference(cv::Mat &image);

        Engine(const Configurations& config);
        ~Engine();


};