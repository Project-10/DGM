#pragma once
//#pragma comment(lib, "Ws2_32.lib")
//#include <string>
#include "types.h"
#include <random>
#include <fstream>

namespace DirectGraphicalModels {
    namespace dnn
    {
        class CNeuron
        {
        public:
            //static const int SIZE = 60;
            
            CNeuron(int size){
                SIZE = size;
            }
            ~CNeuron();
            
            void setNodeValue(double thisValue) {
                m_value = thisValue;
            }

            double getNodeValue() const {
              return m_value;
            }
            
            void generateWeights(int size){
                unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
                srand(seed);

                for (int i= 0; i < size; i++) {
                    double f = (double)rand() / RAND_MAX;
                    double var = -0.5 + f * ((0.5) - (-0.5));
                    m_weight[i] = var;
                }
            }
            
            void setWeight(int index, double x){
                m_weight[index] = x;
            }

            double getWeight(int i) const {
                return m_weight[i];
            }
            
            int getSize() {
                return SIZE;
            }
            
        private:
            int SIZE;
            double m_value;
            double m_weight[100];
        };
    }
}

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





















//DO NOT USE WHATS BELOW

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
