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
    printf("Success\n");
    fflush(stdout);
    return 0;
}