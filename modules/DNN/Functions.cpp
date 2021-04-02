#include "Functions.hpp"
#include <iostream>
#include <fstream>


float applySigmoidFunction(float val) {
    float sigmoid = 1 / (1 + exp(-val));
    float value = (int)(sigmoid * 10000 + .5);
    float result = (float)value / 10000;
    return result;
}

int *readDigitData(std::string file, int dataSize) {
    int *trainDataDigit = new int[dataSize];
    std::string fileDigitData = file;

    std::ifstream inFile;
    inFile.open(fileDigitData.c_str());

    if (inFile.is_open()) {
        for (int i = 0; i < dataSize; i++) {
            inFile >> trainDataDigit[i];
        }
        inFile.close();
    }
    return trainDataDigit;
}


int **readBinData(std::string file, int dataSize) {
    const int inputLayer = 784;
    int **trainDataBin = new int*[dataSize];

    std::string fileBinData = file;
    std::ifstream inFile;
    inFile.open(fileBinData.c_str());

    if (inFile.is_open()) {
        for(int i = 0 ; i < dataSize; i++) {
            trainDataBin[i] = new int[inputLayer];
            
            for (int j = 0; j < inputLayer; j++) {
                inFile >> trainDataBin[i][j];
            }
        }
        inFile.close();
    }
    return trainDataBin;
}


int **resultPredictions(int outputLayer) {
    int **result = new int*[outputLayer];
    
    for(int i = 0; i < 10; i++) {
        result[i] = new int[outputLayer];
        
        for(int j = 0; j < 10; j++) {
            result[i][j] = (i == j) ?  1 :  0;
        }
    }
    return result;
}


//dgm::dnn::CNeuron* dotProd(int hiddenLayer, int inputLayer, dgm::dnn::CNeuron A[], dgm::dnn::CNeuron B[]) {
//    for(int i=0 ; i < hiddenLayer; i++) {
//        double val = 0;
//        for(int j=0; j < inputLayer ; j++) {
//           val += A[j].getWeight(i) * A[j].getNodeValue();
//        }
//        float value = applySigmoidFunction(val);
//        B[i].setNodeValue(value);
//    }
//    return B;
//}
