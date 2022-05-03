#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory.h>
#include <vector>

struct Configurations {
    //Using 16 point floats for inference
    bool FP16 = false;
    //Using int8
    bool INT8 = false;
    //Batch size for optimization
    std::vector<int32_t> optBatchSize;
    // Maximum allowed batch size
    int32_t maxBatchSize = 1;
    //Max GPU memory allowed for the model.
    long int maxWorkspaceSize = 400000000;//
    //GPU device index number, might be useful for more Tegras in the future
    int deviceIndex = 0;
    // DLA
    int dlaCore = -1;

};

void print_help();
void set_precision(char* optargs, Configurations &config);
void set_dla(int cores, Configurations &config);