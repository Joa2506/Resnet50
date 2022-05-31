#include <getopt.h>
#include <opencv2/opencv.hpp>
#include "Engine.hpp"
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

int main(int argc, char **argv)
{
    Configurations config;
    
    //Set configurations
  
    int numberOfIterations = 1000;
    //Precision
    int i;
    while((i = getopt(argc, argv, "p:d:s:h:n:")) != -1)
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
                fflush(stdout);
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
                printf("%d\n", atoi(optarg));
                numberOfIterations = atoi(optarg);
                break;
        }
    }

            
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
    const std::string InputImage = "images/leopard.jpg";
    const std::string InputImage2 = "images/cheetah.jpg";
    const std::string InputImage3 = "images/jaguar.jpg";
    


    auto img = cv::imread(InputImage);
    auto img2 = cv::imread(InputImage2);
    auto img3 = cv::imread(InputImage3);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::cvtColor(img2, img2, cv::COLOR_BGR2RGB);
    cv::cvtColor(img3, img3, cv::COLOR_BGR2RGB);
    for (size_t i = 0; i < numberOfIterations; ++i)
    {
        if(i % 2 == 0)
        {
            images.emplace_back(img2);
        }
        else if(i % 3)
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
    succ = engine.inference(img, 1);
    auto t2 = Clock::now();
    double totalTime = std::chrono::duration_cast<chrono::milliseconds>(t2-t1).count();
    printf("Time of first: %fms\n", totalTime);
    if(!succ)
    {
        throw runtime_error("Could not run inference");
    }
    t1 = Clock::now();
    for (int i = 0; i < numberOfIterations; ++i)
    {
        engine.inference(images[i], 1);
    }
    t2 = Clock::now();
    totalTime = std::chrono::duration_cast<chrono::milliseconds>(t2-t1).count();
    images.clear();

    cout << "Success! Average time per inference on "<< numberOfIterations <<" was " << totalTime / numberOfIterations << "ms" << endl;
    engine.writeToListsPower();
    engine.calculate_averages(totalTime/numberOfIterations, numberOfIterations);
    return 0;
}