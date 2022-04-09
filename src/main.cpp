#include "Engine.hpp"
#include <opencv2/opencv.hpp>

int main()
{
    Configurations config;
    
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