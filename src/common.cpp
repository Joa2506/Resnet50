#include "common.hpp"

//Prints some simple guidelines
void print_help()
{
    printf("Help meny\n");
    printf("Precision:      You can add different precision using -p <PRECISION> | You can choose fp16, fp32 and int8. Remember fp32 is default\n");
    printf("DLA:            You can set the number of DLA cores up to 2 by using -d <DLA_CORES>\n");
    printf("Workspace size: You can set the max workspace size using -s <SIZE>  | warning: Be aware of your GPUs limitation\n");
    exit(2);
}

//Sets the precision added by the user. Make sure only on presicion can be available at a time
void set_precision(char* optargs, Configurations &config)
{
    int ret;
    char fp16[5] = "fp16";
    char fp32[5] = "fp32";
    char int8[5] = "int8";

    if(strncmp(optargs, fp16, 4) == 0)
    {
        config.FP16 = true;
    }
    else if(strncmp(optargs, fp32, 4) == 0)
    {
        //default
        config.FP16 = false;
        config.INT8 = false;
    }
    else if(strncmp(optargs, int8, 4) == 0)
    {
        config.INT8 = true;
        config.FP16 = false;

    }
    else
    {
        printf("Not a valid precision number. Check by running with -h flag\n");
        exit(2);
    }
}
//Sets the number of dla cores to be run by the engine
void set_dla(int cores, Configurations &config)
{
    if(cores == 1)
    {
        config.dlaCore = 1;
    }
    else if(cores == 0)
    {
        config.dlaCore = 0;
    }
    else
    {
        printf("DLA has to be from 0 to 1\n");
        exit(2);
    }
}