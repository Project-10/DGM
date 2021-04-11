#include "Neuron.h"
#include "DGM/random.h"
#include "macroses.h"

namespace DirectGraphicalModels { namespace dnn
{
    // Constructor
    CNeuron::CNeuron(size_t size, float value) : m_value(value) {
        m_vWeights.resize(size);
    }

    void CNeuron::generateRandomWeights(void)
    {
        for (float &weight : m_vWeights)
            weight = random::U<float>(-1, 1);
    }

    void CNeuron::setWeight(size_t index, float weight)
    {
        DGM_ASSERT(index < m_vWeights.size());
        m_vWeights[index] = weight;
    }

    float CNeuron::getWeight(size_t index) const
    {
        DGM_ASSERT(index < m_vWeights.size());
        return m_vWeights[index];
    }
}}


float applySigmoidFunction(float val)
{
    float sigmoid = 1 / (1 + exp(-val));
    return sigmoid;
}

int *readDigitData(std::string file, int dataSize)
{
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


int **readImgData(std::string file, int dataSize)
{
    const int inputLayer = 784;
    static int **trainDataBin = new int*[dataSize];
    
    for(int m = 0; m < dataSize; m++)
    {
        trainDataBin[m] = new int[inputLayer];
        std::string number = std::to_string(m);
        std::string path = file + number + ".png";
        std::string image_path = samples::findFile(path);

        Mat img = imread(image_path, 0);
        
        int l=0;
        for(int i = 0; i < img.rows; i++) {
            for(int j = 0; j < img.cols; j++) {
                int value = abs((int)img.at<uchar>(i,j) - 255);
                trainDataBin[m][l] = value;
                l++;
            }
        }
    }
    return trainDataBin;
}


int **resultPredictions(int outputLayer)
{
    int **result = new int*[outputLayer];
    
    for(int i = 0; i < outputLayer; i++) {
        result[i] = new int[outputLayer];
        for(int j = 0; j < outputLayer; j++) {
            result[i][j] = (i == j) ?  1 :  0;
        }
    }
    return result;
}

//int **readBinData(std::string file, int dataSize) {
//    const int inputLayer = 784;
//    int **trainDataBin = new int*[dataSize];
//
//    std::string fileBinData = file;
//    std::ifstream inFile;
//    inFile.open(fileBinData.c_str());
//
//    if (inFile.is_open()) {
//        for(int i = 0 ; i < dataSize; i++) {
//            trainDataBin[i] = new int[inputLayer];
//
//            for (int j = 0; j < inputLayer; j++) {
//                inFile >> trainDataBin[i][j];
//            }
//        }
//        inFile.close();
//    }
//    return trainDataBin;
//}
