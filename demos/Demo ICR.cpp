#include "DNN.h"
namespace dgm = DirectGraphicalModels;
using namespace std::chrono;
//
//void dotProd(std::vector<dgm::dnn::ptr_neuron_t>& vpLayerA, std::vector<dgm::dnn::ptr_neuron_t>& vpLayerB);

int main() {
    const size_t     numNeuronsInputLayer   = 784;
    const size_t     numNeuronsHiddenLayer  = 60;
    const size_t     numNeuronsOutputLayer  = 10;
    const size_t                  dataSize  = 4000;

    std::vector<dgm::dnn::ptr_neuron_t> vpInputLayer;
    std::vector<dgm::dnn::ptr_neuron_t> vpHiddenLayer;
    std::vector<dgm::dnn::ptr_neuron_t> vpOutputLayer;

    for (size_t i = 0; i < numNeuronsInputLayer; i++)
        vpInputLayer.push_back( std::make_shared<dgm::dnn::CNeuron>(numNeuronsHiddenLayer, 0) );

    for (size_t i = 0; i < numNeuronsHiddenLayer; i++)
        vpHiddenLayer.push_back( std::make_shared<dgm::dnn::CNeuron>(numNeuronsOutputLayer) );

    for (size_t i = 0; i < numNeuronsOutputLayer; i++)
        vpOutputLayer.push_back( std::make_shared<dgm::dnn::CNeuron>(0) );
    
    
    
    
    
    
    

//    int **trainDataBin   = readImgData("../../../train_images_4000/digit_", dataSize);
//    int *trainDataDigit  = readDigitData("../../../a_digit.txt", dataSize);
//    int **resultsArray   = resultPredictions(numNeuronsOutputLayer);
//
//    for (size_t i = 0; i < vpHiddenLayer.size(); i++)
//        vpHiddenLayer[i]->generateRandomWeights();
//
//    for(size_t i = 0; i < vpInputLayer.size(); i++)
//        vpInputLayer[i]->generateRandomWeights();
//
//    auto startTraining = high_resolution_clock::now();
//    for(int k = 0; k < dataSize; k++) {
//            for(size_t i = 0; i < vpInputLayer.size(); i++) {
//                float val = (float)trainDataBin[k][i]/255;
//                vpInputLayer[i]->setNodeValue(val);
//            }
//
//            dotProd(vpInputLayer, vpHiddenLayer);
//            dotProd(vpHiddenLayer, vpOutputLayer);
//
//            double *resultErrorRate = new double[numNeuronsOutputLayer];
//            for(size_t i=0 ; i < vpOutputLayer.size(); i++) {
//                resultErrorRate[i] = resultsArray[trainDataDigit[k]][i] - vpOutputLayer[i]->getNodeValue();
//            }
//
//        // ==================== BACKPROPAGATION ====================
//            float (*DeltaWjk)[numNeuronsOutputLayer]  = new float[numNeuronsHiddenLayer][numNeuronsOutputLayer];
//            float (*DeltaWik)[numNeuronsHiddenLayer]  = new float[numNeuronsInputLayer][numNeuronsHiddenLayer];
//            float *DeltaJ                             = new float[numNeuronsHiddenLayer];
//            float learningRate                        = 0.1;
//
//            for(size_t i = 0; i < vpHiddenLayer.size(); i++) {
//                double nodeVal = 0;
//                for(size_t j = 0; j < vpOutputLayer.size(); j++) {
//                    nodeVal += vpHiddenLayer[i]->getWeight(j) * resultErrorRate[j];
//                    DeltaWjk[i][j] = learningRate * resultErrorRate[j] * vpHiddenLayer[i]->getNodeValue();
//                }
//                float sigmoid = applySigmoidFunction(vpHiddenLayer[i]->getNodeValue());
//                DeltaJ[i] = nodeVal * sigmoid * (1-sigmoid);
//            }
//
//            for(size_t i = 0; i < vpInputLayer.size(); i++) {
//                for(size_t j = 0; j < vpHiddenLayer.size(); j++) {
//                    DeltaWik[i][j] = learningRate * DeltaJ[j] * vpInputLayer[i]->getNodeValue();
//                    float oldWeight = vpInputLayer[i]->getWeight(j);
//                    vpInputLayer[i]->setWeight(j, oldWeight + DeltaWik[i][j]);
//                }
//            }
//
//            for(size_t i = 0; i < vpHiddenLayer.size(); i++) {
//                for(size_t j = 0; j < vpOutputLayer.size(); j++) {
//                    float oldWeight = vpHiddenLayer[i]->getWeight(j);
//                    vpHiddenLayer[i]->setWeight(j, oldWeight + DeltaWjk[i][j]);
//                }
//            }
//    }
//    auto stopTraining = high_resolution_clock::now();
//
////     ==================== TEST DIGITS ====================
//    int testDataSize    = 2000;
//    int correct         = 0;
//    int uncorrect       = 0;
//    int *testDataDigit  = readDigitData("../../../b_digit.txt", testDataSize);
//    int **testDataBin   = readImgData("../../../test_images_2000/digit_", testDataSize);
//
//     auto startTesting = high_resolution_clock::now();
//     for(size_t z = 0; z < testDataSize; z++) {
//         for(size_t i = 0; i < vpInputLayer.size(); i++) {
//             float val = (float)testDataBin[z][i]/255;
//             vpInputLayer[i]->setNodeValue(val);
//         }
//
//         dotProd(vpInputLayer, vpHiddenLayer);
//         dotProd(vpHiddenLayer, vpOutputLayer);
//
//         double *allPredictionsforDigits = new double[numNeuronsOutputLayer];
//         for(size_t i=0 ; i < vpOutputLayer.size(); i++) {
//             allPredictionsforDigits[i] = vpOutputLayer[i]->getNodeValue();
//         }
//
//         float maxAccuracy = 0;
//         int   number;
//         for(size_t i=0 ; i < vpOutputLayer.size(); i++) {
//             if(allPredictionsforDigits[i] >= maxAccuracy) {
//                 maxAccuracy = allPredictionsforDigits[i];
//                 number = i;
//             }
//         }
//
//         std::cout<<"prediction "<<"["<<number<<"] for digit " <<testDataDigit[z] <<" with "<<maxAccuracy<<"% at position: "<<z<<std::endl;
//         number == testDataDigit[z] ? correct++ : uncorrect++;
//     }
//     auto stopTesting    = high_resolution_clock::now();
//
//     auto durationTest   = duration_cast<milliseconds>(stopTesting - startTesting);
//     auto durationTrain  = duration_cast<milliseconds>(stopTraining - startTraining);
//
//     std::cout << "Time taken to train data: "<< durationTrain.count() << " miliseconds" << std::endl;
//     std::cout << "Time taken to test data: " << durationTest.count()  << " miliseconds" << std::endl;
//     std::cout << "poz: " << correct << std::endl << "neg: " << uncorrect << std::endl;
//     std::cout << "average: " << (float)correct/(correct+uncorrect)*100 << "%" << std::endl;
    return 0;
}

//void dotProd(std::vector<dgm::dnn::ptr_neuron_t>& vpLayerA, std::vector<dgm::dnn::ptr_neuron_t>& vpLayerB) {
//    for(size_t i = 0 ; i < vpLayerB.size(); i++) {
//        float value = 0;
//        for(const auto& a : vpLayerA)
//            value += a->getWeight(i) * a->getNodeValue();
//
//        value = applySigmoidFunction(value);
//        vpLayerB[i]->setNodeValue(value);
//    }
//}


//    ==== READ IMAGE DATA FROM PIXELS ====
    
//    int **trainDataBin = readImgData(200);
//    static int testDataBin[2000][784];
//
//    for(int m=0; m<25; m++) {
//        int num = m;
//        std::string number = std::to_string(num);
//        std::string path = "/Users/diond/Desktop/train_images_4000/digit_" + number + ".png";
//        std::string image_path = samples::findFile(path);
//
//        Mat img = imread(image_path, 0);
//
//        if(img.empty()) {
//            std::cout << "Could not read the image: " << image_path << std::endl;
//            return 1;
//        }

//        int l=0;
//        for(int i = 0; i < 28; i++)
//        {
//            for(int j = 0; j < 28; j++)
//            {
//                int value = abs((int)img.at<uchar>(i,j) - 255);
//                testDataBin[m][l] = value;
//                l++;
//            }
//        }
//    }
    
    
//    ==== WRITE IMAGE DATA TO PIXELS ====

// int testDataSize    = 2000;
// int *testDataDigit  = readDigitData("../../../train_digit.txt", testDataSize);
// int **testDataBin   = readBinData("/Users/diond/Desktop/b_data.txt", testDataSize);

//for(int z=0; z<testDataSize; z++){
//    Mat img(Size(28,28), CV_8UC3, Scalar(255,255,255));
//    int num = z;
//    std::string number = std::to_string(num);
//    std::string path = "/Users/diond/Desktop/test_images_2000/digit_" + number + ".png";
//
//        int arr2[28][28];
//        int l=0;
//        for(int i=0; i<28; i++){
//            for(int j=0; j<28; j++){
//                arr2[i][j]=testDataBin[z][l];
//                l++;
//            }
//        }
//        cv::Vec3b color = img.at<Vec3b>(Point(0,0));
//        for(int i = 0; i < 28; i++)
//        {
//            for(int j = 0; j < 28; j++)
//            {
//                Vec3b bgrPixel = img.at<Vec3b>(j, i);
//                //img.at<Vec3b>(i+100, j+20) = 255;
//                if(arr2[i][j] == 0){
//                    color[0]=255;
//                    color[1]=255;
//                    color[2]=255;
//                }
//                else {
//                    color[0]= 255 - arr2[i][j];
//                    color[1]= 255 - arr2[i][j];
//                    color[2]= 255 - arr2[i][j];
//                }
//                img.at<Vec3b>(Point(j,i)) = color;
//            }
//        }
//        imwrite(path, img);
//    }
