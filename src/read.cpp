#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <iterator>
using namespace std;

int main()
{
    vector<int> list;     
    string name = "/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power1_input";
    string powerFile = "power.txt";
    string line;

    
    ofstream outputfile(powerFile);
    std::ostream_iterator<int> output_iterator(outputfile, "\n");
  
  
        for(int i = 0; i < 10; ++i){
            ifstream file(name);
            getline(file, line);
            list.emplace_back(stoi(line));
            printf("%d\n", list[i]);
            file.close();
        }


    std::copy(list.begin(), list.end(), output_iterator);
    
    
    outputfile.close();
    

}