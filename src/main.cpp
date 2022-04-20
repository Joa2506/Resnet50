#include <getopt.h>
#include <opencv2/opencv.hpp>
#include "Engine.hpp"
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

int main(int argc, char **argv)
{
    Configurations config;
    
    //Number of iterations to run
    int numberOfIterations = 100;
  
    //Precision
    int i;
    while((i = getopt(argc, argv, "p:d:s:n:h")) != -1)
    {
        switch (i)
        {
            case 'p':
                printf("Precision\n");
                //printf("%s\n", optarg);
                //fflush(stdout);
                set_precision(optarg, config);
                printf("fp16: %d    |   ", config.FP16);
                printf("INT8: %d    |   \n", config.INT8);
                break;
            case 'd':
                printf("DLA\n");
                set_dla(atoi(optarg), config);
                printf("DLA cores %d\n", config.dlaCore);
                break;
            case 's':
                printf("Workspace size");
                config.maxWorkspaceSize = atoi(optarg);
                break;
            case 'h':
                print_help();
                break;
            case 'n':
                numberOfIterations = atoi(optarg);

        }
    }

    vector<float> inferenceTime;
    
    Engine engine(config);

    bool succ = engine.build(RESNET);
    if(!succ)
    {
        throw runtime_error("Could not built TRT engine");
    }
    succ = engine.loadNetwork();
    if(!succ)
    {
        throw runtime_error("Could not load TRT engine from disk");
    }
    const size_t batchSize = 1;
    std::vector<cv::Mat> images;
    //const std::string InputImage = "images/turkish_coffee.jpg";
    //const std::string InputImage = "images/golden_retriever.jpg";
    //const std::string InputImage = "images/cat.jpg";
    //const std::string InputImage = "images/wardrobe.jpg";
    const std::string InputImage = "images/cheetah.jpg";
    const std::string InputImage2 = "images/jaguar2.jpg";
    const std::string InputImage3 = "images/leopard.jpg";
    


    auto img = cv::imread(InputImage);
    auto img2 = cv::imread(InputImage2);
    auto img3 = cv::imread(InputImage3);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::cvtColor(img2, img2, cv::COLOR_BGR2RGB);
    cv::cvtColor(img3, img3, cv::COLOR_BGR2RGB);
    for (size_t i = 0; i < numberOfIterations; ++i)
    {
        if(i%2 == 0)
        {
            images.emplace_back(img2);
        }
        else if (i % 3 == 0)
        {
            images.emplace_back(img3);
        }
        else
        {
            images.emplace_back(img);
        }
    }
    
    //cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    //cv::imshow("img",img);
    //cv::waitKey(0);
    printf("Starting inference\n");
    fflush(stdout);
    //First iteration takes longer.
    auto t1 = Clock::now();
    succ = engine.inference(img, 1, inferenceTime);
    auto t2 = Clock::now();
    double totalTime = std::chrono::duration_cast<chrono::milliseconds>(t2-t1).count();
    printf("Time of first: %f\n", totalTime);
    if(!succ)
    {
        throw runtime_error("Could not run inference");
    }
    t1 = Clock::now();
    inferenceTime.clear();
    for (int i = 0; i < numberOfIterations; ++i)
    {
        engine.inference(images[i], 1, inferenceTime);
    }
    t2 = Clock::now();
    totalTime = std::chrono::duration_cast<chrono::milliseconds>(t2-t1).count();
    images.clear();

    printf("size %ld\n", inferenceTime.size());
    float totalInferenceTime = 0;
    for (size_t i = 0; i < numberOfIterations; ++i)
    {
        totalInferenceTime += inferenceTime[i];
    }
    

    cout << "Average inference time is: " << totalInferenceTime/numberOfIterations << "ms" <<endl;
    cout << "Average time per inference procedure on "<< numberOfIterations <<" was " << totalTime / numberOfIterations << "ms" << endl;
    return 0;
}