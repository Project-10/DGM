#include "DNN.h"
#include "DGM.h"
#include "VIS.h"
#include "DGM/timer.h"
#include <fstream>

namespace dgm = DirectGraphicalModels;

float sigmoidFunction_derivative(float x)
{
    float s = 1.0f / (1.0f + expf(-x));
    return s * (1 - s);
}
/**
 * Reads the digits numerical value in a decimal notation
 *
 * @param file to read, and the number of digits to read
 * @return an array of digit labels
 */
std::vector<byte> readGroundTruth(const std::string& fileName)
{
    std::vector<byte> res;
    std::ifstream inFile;
    inFile.open(fileName.c_str());
    
    if (inFile.is_open()) {
        int val;
        while (!inFile.eof()) {
            inFile >> val;
            res.push_back(static_cast<byte>(val));
        }
        inFile.close();
    }
    return res;
}

int main()
{
    const byte      nStates                  = 10; // 10 digits (number of output nodes)
    const word      nFeatures                = 28 * 28; // every pixel of 28x28 digit image
    const size_t    numNeuronsHiddenLayer    = 100;
    const size_t    numTrainSamples          = 3000;
    const size_t    numTestSamples           = 2000;
    const size_t    epochs                   = 3;

#ifdef WIN32
    const std::string dataPath = "../../data/digits/";
#else
    const std::string dataPath = "../../../data/digits/";
#endif

    dgm::dnn::CNeuronLayerMat layerInput(nFeatures, numNeuronsHiddenLayer);
    dgm::dnn::CNeuronLayerMat layerHidden(numNeuronsHiddenLayer, nStates);
    dgm::dnn::CNeuronLayerMat layerOutput(nStates, 0);

    layerInput.generateRandomWeights();
    layerHidden.generateRandomWeights();

    Mat fv;

    // ==================== TRAINING DIGITS ====================
    dgm::Timer::start("Training...");
    auto    trainGT = readGroundTruth(dataPath + "train_gt.txt");

    for(size_t x = 0; x < epochs; x++){
        for(size_t s = 0; s < numTrainSamples; s++) {
            std::stringstream ss;
            ss << dataPath << "train/digit_" << std::setfill('0') << std::setw(4) << s << ".png";
            std::string fileName = samples::findFile(ss.str());
            Mat img = imread(fileName, 0);
            img = img.reshape(1, img.cols * img.rows);
            img.convertTo(fv, CV_32FC1, 1.0 / 255);
            fv = Scalar(1.0f) - fv;

            layerInput.setValues(fv);

            layerHidden.dotProd(layerInput);
            layerHidden.applySigmoid();

            layerOutput.dotProd(layerHidden);
            Mat outputValues = layerOutput.getValues();
            layerOutput.applySigmoid();

            Mat resultErrorRate(nStates, 1, CV_32FC1);

            for(int i = 0; i < resultErrorRate.rows; i++) {
                resultErrorRate.at<float>(i, 0) = (trainGT[s] == i) ? 1 : 0;
                resultErrorRate.at<float>(i, 0) = (resultErrorRate.at<float>(i, 0) - outputValues.at<float>(i,0)) * sigmoidFunction_derivative(outputValues.at<float>(i,0));
            }
            dgm::dnn::CNeuronLayerMat::backPropagate(layerInput, layerHidden, layerOutput, resultErrorRate, 0.1f);
        }
    }
    dgm::Timer::stop();

    // ==================== TESTING DIGITS ====================
    dgm::CCMat confMat(nStates);
    dgm::Timer::start("Testing...");
    auto     testGT = readGroundTruth(dataPath + "test_gt.txt");
    for(size_t s = 0; s < numTestSamples; s++) {
        std::stringstream ss;
        ss << dataPath << "test/digit_" << std::setfill('0') << std::setw(4) << s << ".png";
        std::string fileName = samples::findFile(ss.str());
        Mat img = imread(fileName, 0);
        img = img.reshape(1, img.cols * img.rows);
        img.convertTo(fv, CV_32FC1, 1.0 / 255);
        fv = Scalar(1.0f) - fv;

        layerInput.setValues(fv);

        layerHidden.dotProd(layerInput);
        layerHidden.applySigmoid();

        layerOutput.dotProd(layerHidden);
        layerOutput.applySigmoid();

        std::vector<double>pot = layerOutput.getValues();

        auto maxAccuracy = max_element(std::begin(pot), std::end(pot));
        int number = std::distance(pot.begin(), maxAccuracy);

        confMat.estimate(number, testGT[s]);

        //printf("prediction [%d] for digit %d with %.3f%s at position %zu \n", number, testDataDigit[z], maxAccuracy, "%", z);
    }
    dgm::Timer::stop();
    printf("Accuracy = %.2f%%\n", confMat.getAccuracy());

    // Confusion matrix
    dgm::vis::CMarker marker;
    Mat cMat    = confMat.getConfusionMatrix();
    Mat cMatImg = marker.drawConfusionMatrix(cMat, dgm::vis::MARK_BW);
    imshow("Confusion Matrix", cMatImg);

    waitKey();

    return 0;
}