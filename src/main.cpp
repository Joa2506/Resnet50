#include <getopt.h>
#include <opencv2/opencv.hpp>
#include "Engine.hpp"

int main(int argc, char **argv)
{
    Configurations config;
    
    //Set configurations
  
    //Precision
    int i;
    while((i = getopt(argc, argv, "p:d:s:h")) != -1)
    {
        switch (i)
        {
            case 'p':
                printf("Precision\n");
                //printf("%s\n", optarg);
                //fflush(stdout);
                set_precision(optarg, config);
                printf("fp16: %d    |   ", config.FP16);
                printf("INT8: %d\n", config.INT8);
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
        }
    }

        
        //DLA

        //Workspace size
    
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
    //const std::string InputImage = "images/leopard.jpg";
    //const std::string InputImage = "images/cheetah.jpg";
    const std::string InputImage = "images/jaguar2.jpg";
    
    auto img = cv::imread(InputImage);
    //cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    //cv::imshow("img",img);
    //cv::waitKey(0);
    printf("Starting inference\n");
    fflush(stdout);
    succ = engine.inference(img, 1);
    if(!succ)
    {
        throw runtime_error("Could not run inference");
    }
    if(succ)
    {
        cout << "Inference worked man!" << endl;
    }
    printf("Success\n");
    fflush(stdout);
    return 0;
}