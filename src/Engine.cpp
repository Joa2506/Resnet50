#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//TRT
#include <NvOnnxConfig.h>
#include <NvOnnxParser.h>

//C++
#include <iostream>
#include <fstream>
#include <time.h>
//Own
#include "Engine.hpp"

//OpenCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>

Engine::Engine(const Configurations& config) : m_config(config) {}

Engine::~Engine()
{
    if(m_cudaStream)
    {
        cudaStreamDestroy(m_cudaStream);
    }
}

string Engine::serializeEngineName(const Configurations& config)
{
    printf("Serializing engine name...\n");
    fflush(stdout);
    string name = "trt.engine";

    if(config.FP16)
    {
        name += ".fp16";
    }
    else
    {
        name += ".fp32";
    }
    name += "." + to_string(config.maxBatchSize); + ".";
    for (int i = 0; i < config.optBatchSize.size(); ++i)
    {
        name += to_string(config.optBatchSize[i]);
        if(i != config.optBatchSize.size() - 1)
        {
            name += "_";
        } 
    }

    name += "." + to_string(config.maxWorkspaceSize);
    
    printf("Serialed engine name...\n");
    fflush(stdout);

    return name;
}

bool Engine::fileExists(string FILENAME)
{
    ifstream f(FILENAME.c_str());
    return f.good();
}

bool Engine::build(string onnxfilename)
{
    printf("Starting builder...\n");
    fflush(stdout);
    m_engineName = serializeEngineName(m_config);
    if(fileExists(m_engineName))
    {
        cout << "Engine already exists..." << endl;
        cout <<" Loading from disk..." << endl;

        return true;
    }

    //No engine found on disk
    cout << "Building engine from Onnx file..." << endl;

    auto builder = unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if(!builder)
    {
        cout << "Builder creation failed!" << endl;
        cout << "Exiting..." << endl;
        return false;
    }

    //Set max batch size
    builder->setMaxBatchSize(m_config.maxBatchSize);

    cout << "Buider successful!" << endl;
    cout << "Building the Network..." << endl;
    //Need to cast enum
    auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if(!network)
    {
        cout << "Network creation failed!" << endl;
        cout << "Exiting..." << endl;
        return false;
    }
    cout << "Network built successfully!" << endl;
    //Creating the parser
    cout << "Building the parser..." << endl;
    auto parser = unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if(!parser)
    {
        cout << "Parser creation failed!" << endl;
        cout << "Exiting..." << endl;

        return false;
    }
    cout << "Parser built successfully!" << endl;
    ifstream file(onnxfilename, ios::binary | ios::ate);
    streamsize fileSize = file.tellg();
    file.seekg(0, ios::beg);
    
    vector<char> buffer(fileSize);
    if(!file.read(buffer.data(), fileSize))
    {
        throw runtime_error("Was not able to parse the model");
    }
    cout << "Parsing the parser..." << endl;
    auto parsed = parser->parse(buffer.data(), buffer.size());
    for (size_t i = 0; i < parser->getNbErrors(); i++)
    {
        cout << parser->getError(i)->desc() << endl;
    }
    if(!parsed)
    {
        cout << "Parsing failed!" << endl;
        cout << "Exiting..." << endl;
        return false;
    }

    auto config = unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    cout << "Configuring the builder..." << endl;
    if(!config)
    {
        cout << "Was not able to build the config" << endl;
        return false;
    }

    auto input = network->getInput(0);
    auto output = network->getOutput(0);
    auto inputName = input->getName();
    m_inputDims = input->getDimensions();
    int32_t channel = m_inputDims.d[1];
    int32_t height = m_inputDims.d[2];
    int32_t width = m_inputDims.d[3];

    cout << "Adding optimization profile..." << endl;
    IOptimizationProfile *defaultProfile = builder->createOptimizationProfile();
    printf("Hei\n");
    fflush(stdout);
    defaultProfile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(1, channel, height, width));
    defaultProfile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4(1, channel, height, width));
    defaultProfile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4(1, channel, height, width));
    config->addOptimizationProfile(defaultProfile);

    cout << "Optimization profile added" << endl;
    cout << "Setting max workspace size..." << endl;
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, m_config.maxWorkspaceSize);
    cout << "Builder configured successfully" << endl;
    cout << "Making cuda stream..." << endl;
    auto cudaStream = samplesCommon::makeCudaStream();
    if(!cudaStream)
    {
        cout << "Could not create cudaStream." << endl;
        return false;
    }
    cout << "Cuda stream made succsessully" << endl;
    //Setting the profile stream
    config->setProfileStream(*cudaStream);
    cout << "Making serialized model..." << endl;
    unique_ptr<IHostMemory> serializedModel{builder->buildSerializedNetwork(*network, *config)};
    if(!serializedModel)
    {
        cout << "Could not build serialized model" << endl;
        return false;
    }
    cout << "Model serialized" << endl;

    /*TODO ADD DLA for Tegra 
    */
   
    cout << "Writing serialized model to disk..." << endl;
    //write the engine to disk
    ofstream outfile(m_engineName, ofstream::binary);
    outfile.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
    cout << "The engine has been built and saved to disk successfully" << endl;

    return true;


}

bool Engine::loadNetwork()
{
    ifstream file(m_engineName, ios::binary | ios::ate);
    streamsize size = file.tellg();
    file.seekg(0, ios::beg);

    vector<char> buffer(size);
    cout << "Trying to read engine file..." << endl;
    if(!file.read(buffer.data(), size))
    {
        cout << "Could not read engine from disk..." << endl;
        return false;
    }
    cout << "Engine was successfully read from disk" << endl;

    cout << "Creating a runtime object..." << endl;
    unique_ptr<IRuntime> runtime{createInferRuntime(m_logger)};
    if(!runtime)
    {
        cout << "Could not create runtime object" << endl;
        return false;
    }
    cout << "Network object was created successfully" << endl;

    auto ret = cudaSetDevice(m_config.deviceIndex);
    //Checks if device index exists. Should only be one device for this program. Might test with more at a later time
    if(ret != 0)
    {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        auto errMsg = "Unable to set GPU device index to: " + std::to_string(m_config.deviceIndex) +
                ". Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).";
        throw std::runtime_error(errMsg);
    }

    //Let's create the engine
    cout << "Creting cuda engine... " << endl;
    m_engine = shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if(!m_engine)
    {
        cout << "Creating the cuda engine failed" << endl;
        return false;
    }
    cout << "Cuda engine was created successfully" << endl;

    //Getting input and output names plus dimensions
    m_inputName = m_engine->getBindingName(0);
    m_outputName = m_engine->getBindingName(1);
    
    m_inputDims = m_engine->getBindingDimensions(0);
    m_outputDims = m_engine->getBindingDimensions(1);

    m_batchSize = m_inputDims.d[0];
    m_inputChannel = m_inputDims.d[1];
    m_inputHeight = m_inputDims.d[2];
    m_inputWidth = m_inputDims.d[3];
    
    printf("m_inputname == %s\n", m_inputName);
    printf("m_outputname == %s\n", m_outputName);
    printf("m_batchSize == %d\n", m_batchSize);
    printf("m_inputChannel == %d\n", m_inputChannel);
    printf("m_inputHeight == %d\n", m_inputHeight);
    printf("m_inputWidth == %d\n", m_inputWidth);

    cout << "Creating execution context..." << endl;
    m_context = shared_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if(!m_context)
    {
        cout << "Creating execution context failed" << endl;
        return false;
    }

    cout << "Execution context was created successfully" << endl;
    cout << "Creating CudaStream..." << endl;
    auto cudaRet = cudaStreamCreate(&m_cudaStream);
    printf("%d\n", cudaRet);
    if(cudaRet != 0)
    {
        throw std::runtime_error("Unable to create cuda stream");
    }
    cout << "Cuda stream created successfully!" << endl;
    return true;
}

bool Engine::inference(cv::Mat &image)
{

}

bool Engine::resizeAndNormalize(cv::Mat frame, float* gpu_input, const nvinfer1::Dims& dims)
{
    cv::cuda::GpuMat gpu_frame;
    // upload image to GPU
    gpu_frame.upload(frame);

    auto input_width = dims.d[2];
    auto input_height = dims.d[1];
    auto channels = dims.d[0];
    auto input_size = cv::Size(input_width, input_height);
    // resize
    cv::cuda::GpuMat resized;
    cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_NEAREST);
    // normalize
    cv::cuda::GpuMat flt_image;
    resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
    cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
    cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);
    // to tensor
    std::vector<cv::cuda::GpuMat> chw;
    for (size_t i = 0; i < channels; ++i)
    {
        chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + i * input_width * input_height));
    }
    cv::cuda::split(flt_image, chw);
}