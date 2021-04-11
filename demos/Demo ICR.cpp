#include "DNN.h"
namespace dgm = DirectGraphicalModels;
using namespace std::chrono;

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

//    int *trainDataDigit  = readDigitData("../../../test_digits.txt", 2000);
//    int **trainDataBin   = readBinData("../../../test_data.txt", 2000);
//    int **trainDataBin   = readBinData("../../../a_data.txt", dataSize);

//    int **trainDataBin   = readImgData("../../../train_images_4000/digit_", dataSize);
//    int *trainDataDigit  = readDigitData("../../../a_digit.txt", dataSize);
    
    int **trainDataBin   = readImgData("/Users/diond/Desktop/zzz/zzz", dataSize);
    int *trainDataDigit  = readDigitData("/Users/diond/Desktop/zzz_bin.txt", dataSize);
    int **resultsArray   = resultPredictions(numNeuronsOutputLayer);

    for (size_t i = 0; i < vpHiddenLayer.size(); i++)
        vpHiddenLayer[i]->generateRandomWeights();

    for(size_t i = 0; i < vpInputLayer.size(); i++)
        vpInputLayer[i]->generateRandomWeights();


    auto startTraining = high_resolution_clock::now();

    for(int k = 0; k < dataSize; k++) {
            for(size_t i = 0; i < vpInputLayer.size(); i++) {
                float val = (float)trainDataBin[k][i]/255;
                vpInputLayer[i]->setNodeValue(val);
            }

            //inputWeightMatrix dotProduct inputMatrix
            //[36 x 784] [784 x 1] --> Hidden layerMatrix [36, 1]
            for(size_t i=0 ; i < vpHiddenLayer.size(); i++) {
                double val = 0;
                for(size_t j=0; j < vpInputLayer.size(); j++) {
                   val += vpInputLayer[j]->getWeight(i) * vpInputLayer[j]->getNodeValue();
                }
                float value = applySigmoidFunction(val);
                vpHiddenLayer[i]->setNodeValue(value);
            }

            // HiddenOutputMatrix dotProduct Hidden node
            //  [10 x 36] [36 x 1] --> Output layerMatrix [10, 1]
            for(size_t i=0 ; i < vpOutputLayer.size(); i++) {
                double val = 0;
                for(size_t j = 0; j < vpHiddenLayer.size() ; j++) {
                   val += vpHiddenLayer[j]->getWeight(i) * vpHiddenLayer[j]->getNodeValue();
                }
                float value = applySigmoidFunction(val);
                vpOutputLayer[i]->setNodeValue(value);
            }

            double *resultErrorRate = new double[numNeuronsOutputLayer];
            for(size_t i=0 ; i < vpOutputLayer.size(); i++) {
                resultErrorRate[i] = resultsArray[trainDataDigit[k]][i] - vpOutputLayer[i]->getNodeValue();
            }

        // ==================== BACKPROPAGATION ====================
            float (*DeltaWjk)[numNeuronsOutputLayer]  = new float[numNeuronsHiddenLayer][numNeuronsOutputLayer];
            float (*DeltaWik)[numNeuronsHiddenLayer]  = new float[numNeuronsInputLayer][numNeuronsHiddenLayer];
            float *DeltaIn_j                          = new float[numNeuronsHiddenLayer];
            float *DeltaJ                             = new float[numNeuronsHiddenLayer];
            float learningRate                        = 0.1;

            //updates weights between [hiddenLayer][outputLayer]
            for(size_t i = 0; i < vpHiddenLayer.size(); i++) {
                double val = 0;
                for(size_t j = 0; j < vpOutputLayer.size(); j++) {
                    val += vpHiddenLayer[i]->getWeight(j) * resultErrorRate[j];
                    DeltaWjk[i][j] = learningRate * resultErrorRate[j] * vpHiddenLayer[i]->getNodeValue();
                }
                DeltaIn_j[i] = val;
            }

            //still hiddenlayer nodes
            for(size_t i = 0; i < vpHiddenLayer.size(); i++) {
                float sigmoid = 1 / (1 + exp(vpHiddenLayer[i]->getNodeValue()));
                float inverse = 1 - sigmoid;
                DeltaJ[i] = DeltaIn_j[i] * sigmoid * inverse;
            }

            //updates weights between [inputLayer][hiddenLayer]
            for(size_t i = 0; i < vpInputLayer.size(); i++) {
                for(int j = 0; j < vpHiddenLayer.size(); j++) {
                    DeltaWik[i][j] = learningRate * DeltaJ[j] * vpInputLayer[i]->getNodeValue();
                }
            }

            for(size_t i = 0; i < vpInputLayer.size(); i++) {
                for(size_t j = 0; j < vpHiddenLayer.size(); j++) {
                    float oldWeight = vpInputLayer[i]->getWeight(j);
                    vpInputLayer[i]->setWeight(j, oldWeight + DeltaWik[i][j]);
                }
            }

            for(size_t i = 0; i < vpHiddenLayer.size(); i++) {
                for(size_t j = 0; j < vpOutputLayer.size(); j++) {
                    float oldWeight = vpHiddenLayer[i]->getWeight(j);
                    vpHiddenLayer[i]->setWeight(j, oldWeight + DeltaWjk[i][j]);
                }
            }
    }
    auto stopTraining = high_resolution_clock::now();


//     ==================== TEST DIGITS ====================
    int testDataSize    = 2000;
    int *testDataDigit  = readDigitData("../../../b_digit.txt", testDataSize);
    int **testDataBin   = readImgData("../../../test_images_2000/digit_", testDataSize);
    
//     int *testDataDigit  = readDigitData("../../../train_digit.txt", testDataSize);
//     int **testDataBin   = readBinData("../../../train_data.txt", testDataSize);

     int correct      = 0;
     int uncorrect    = 0;

     auto startTesting = high_resolution_clock::now();
     for(size_t z = 0; z < testDataSize; z++) {
         for(size_t i = 0; i < vpInputLayer.size(); i++) {
             float val = (float)testDataBin[z][i]/255;
             vpInputLayer[i]->setNodeValue(val);
         }

         for(size_t i=0 ; i < vpHiddenLayer.size(); i++) {
             double val = 0;
             for(size_t j=0; j < vpInputLayer.size(); j++) {
                val += vpInputLayer[j]->getWeight(i) * vpInputLayer[j]->getNodeValue();
             }
             float value = applySigmoidFunction(val);
             vpHiddenLayer[i]->setNodeValue( value );
         }

         for(size_t i=0 ; i < vpOutputLayer.size(); i++) {
             double val = 0;
             for(size_t j = 0; j < vpHiddenLayer.size(); j++) {
                val += vpHiddenLayer[j]->getWeight(i) * vpHiddenLayer[j]->getNodeValue();
             }
             float value = applySigmoidFunction(val);
             vpOutputLayer[i]->setNodeValue(  value );
         }

         double *allPredictionsforDigits = new double[numNeuronsOutputLayer];
         for(size_t i=0 ; i < vpOutputLayer.size(); i++) {
             allPredictionsforDigits[i] = vpOutputLayer[i]->getNodeValue();
         }

         float  maxAccuracy = 0;
         int number;
         for(size_t i=0 ; i < vpOutputLayer.size(); i++) {
             if(allPredictionsforDigits[i] >= maxAccuracy) {
                 maxAccuracy = allPredictionsforDigits[i];
                 number = i;
             }
         }

         if (number == testDataDigit[z]) {
             correct ++;
             //std::cout<<"prediction "<<"["<<number<<"] for digit " <<testDataDigit[z] <<" with "<<maxAccuracy<<"% at position: "<<z<<std::endl;
         } else{
             uncorrect++;
             //std::cout<<"prediction "<<"["<<number<<"] for digit " <<testDataDigit[z] <<" with "<<maxAccuracy<<"% at position: "<<z<<"  [x]"<<std::endl;
         }
     }
     auto stopTesting    = high_resolution_clock::now();

     auto durationTest   = duration_cast<milliseconds>(stopTesting - startTesting);
     auto durationTrain  = duration_cast<milliseconds>(stopTraining - startTraining);

     std::cout << "Time taken to train data: "<< durationTrain.count() << " miliseconds" << std::endl;
     std::cout << "Time taken to test data: " << durationTest.count()  << " miliseconds" << std::endl;
     std::cout << "poz: " << correct << std::endl << "neg: " << uncorrect << std::endl;
     std::cout << "average: " << (float)correct/(correct+uncorrect)*100 << "%" << std::endl;

    

    
    
    

    
    
    
//    int **trainDataBin = readImgData(200);
//    static int testDataBin[2000][784];
//
//    for(int m=0; m<25; m++)
//    {
//        int num = m;
//        std::string number = std::to_string(num);
//
//        std::string path = "/Users/diond/Desktop/train_images_4000/digit_" + number + ".png";
//        std::string image_path = samples::findFile(path);
//
//        Mat img = imread(image_path, 0);
//
//        if(img.empty()) {
//            std::cout << "Could not read the image: " << image_path << std::endl;
//            return 1;
//        }
//
//
//        int l=0;
//        for(int i = 0; i < 28; i++)
//        {
//            for(int j = 0; j < 28; j++)
//            {
//                int value = abs((int)img.at<uchar>(i,j) - 255);
//                testDataBin[m][l] = value;
////                std::cout<<  std::setw(2)<< testDataBin[m][l] << " ";
//                l++;
//            }
////            std::cout<<"\n";
//        }
//
//    }
    
    
    
//    for(int m=0; m<3; m++){
//        int k=0;
//        for(int i = 0; i < 28; i++)
//        {
//            for(int j = 0; j < 28; j++)
//            {
//                std::cout<< trainDataBin[m][k] << " ";
//                k++;
//            }
//            std::cout<<"\n";
//        }
//    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

// int testDataSize    = 2000;
//// int *testDataDigit  = readDigitData("../../../train_digit.txt", testDataSize);
// int **testDataBin   = readBinData("/Users/diond/Desktop/b_data.txt", testDataSize);
//
//
//for(int z=0; z<testDataSize; z++){
//
//    Mat img(Size(28,28), CV_8UC3, Scalar(255,255,255));
//    int num = z;
//
//    std::string number = std::to_string(num);
//
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


    return 0;
}















































//#include "DNN.h"
//#include <fstream>
//namespace dgm = DirectGraphicalModels;
//using namespace std::chrono;

//int main() {
//    const int inputLayer   = 784;
//    const int hiddenLayer  = 60;
//    const int outputLayer  = 10;
//    float     rangeMin     = -0.5;
//    float     rangeMax     = 0.5;
//
//    dgm::dnn::CNeuron *myNeuron        = new dgm::dnn::CNeuron[inputLayer];
//    dgm::dnn::CNeuron *myHiddenNeuron  = new dgm::dnn::CNeuron[hiddenLayer];
//    dgm::dnn::CNeuron *myOutputNeuron  = new dgm::dnn::CNeuron[outputLayer];
//
//    int *trainDataDigit  = readDigitData("../../../test_digits.txt", 2000);
//    int **trainDataBin   = readBinData("../../../test_data.txt", 2000);
//    int **resultsArray   = resultPredictions(outputLayer);
//
//// ==================== Training Data ====================
//    auto startTraining = high_resolution_clock::now();
//
//    for(int i = 0; i < inputLayer; i++) {
//        myNeuron[i].generateWeights(hiddenLayer);
//    }
//    for(int i = 0; i < hiddenLayer; i++) {
//        myHiddenNeuron[i].generateWeights(outputLayer);
//    }
//
//        for(int k = 0; k < 2000; k++) {
//            //std::cout<<"Training for digit: "<< trainDataDigit[k]<<std::endl;
//            for(int i = 0; i < inputLayer; i++) {
//                float val = (float)trainDataBin[k][i]/255;
//                float value = (int)(val * 1000 + .5);
//                myNeuron[i].setNodeValue( (float)value / 1000 );
//            }
//
//            //inputWeightMatrix dotProduct inputMatrix
//            //[36 x 784] [784 x 1] --> Hidden layerMatrix [36, 1]
//            for(int i=0 ; i < hiddenLayer; i++) {
//                double val = 0;
//                for(int j=0; j < inputLayer ; j++) {
//                   val += myNeuron[j].getWeight(i) * myNeuron[j].getNodeValue();
//                }
//                float sigmoid = 1 / (1 + exp(-val));
//                float value = (int)(sigmoid * 1000 + .5);
//                myHiddenNeuron[i].setNodeValue((float)value / 1000);
//            }
//
//            // HiddenOutputMatrix dotProduct Hidden node
//            //  [10 x 36] [36 x 1] --> Output layerMatrix [10, 1]
//            for(int i=0 ; i < outputLayer; i++) {
//                double val = 0;
//                for(int j = 0; j < hiddenLayer ; j++) {
//                   val += myHiddenNeuron[j].getWeight(i) * myHiddenNeuron[j].getNodeValue();
//                }
//                float sigmoid = 1 / (1 + exp(-val));
//                float value = (int)(sigmoid * 1000 + .5);
//                myOutputNeuron[i].setNodeValue((float)value / 1000);
//            }
//
//
//            double *resultErrorRate = new double[outputLayer];
//            for(int i=0 ; i < outputLayer; i++) {
//                resultErrorRate[i] = resultsArray[trainDataDigit[k]][i] - myOutputNeuron[i].getNodeValue();
//            }
//
//        // ==================== BACKPROPAGATION ====================
//            float (*DeltaWjk)[outputLayer]  = new float[hiddenLayer][outputLayer];
//            float (*DeltaVjk)[hiddenLayer]  = new float[inputLayer][hiddenLayer];
//            float *DeltaIn_j                = new float[hiddenLayer];
//            float *DeltaJ                   = new float[hiddenLayer];
//            float learningRate              = 0.1;
//
//            //updates weights between [hiddenLayer][outputLayer]
//            for(int i = 0; i < hiddenLayer; i++) {
//                double val = 0;
//                for(int j = 0; j < outputLayer; j++) {
//                    val += myHiddenNeuron[i].getWeight(j) * resultErrorRate[j];
//                    DeltaWjk[i][j] = learningRate * resultErrorRate[j] * myHiddenNeuron[i].getNodeValue();
//                }
//                DeltaIn_j[i] = val;
//            }
//
//            //still hiddenlayer nodes
//            for(int i = 0; i < hiddenLayer; i++) {
//                float sigmoid = 1 / (1 + exp(myHiddenNeuron[i].getNodeValue()));
//                float inverse = 1 - sigmoid;
//                DeltaJ[i] = DeltaIn_j[i] * sigmoid * inverse;
//            }
//
//            //updates weights between [inputLayer][hiddenLayer]
//            for(int i = 0; i < inputLayer; i++) {
//                for(int j = 0; j < hiddenLayer; j++) {
//                    DeltaVjk[i][j] = learningRate * DeltaJ[j] * myNeuron[i].getNodeValue();
//                }
//            }
//
//        // ==================== UPDATE WEIGHTS ====================
//            for(int i = 0; i < inputLayer; i++) {
//                for(int j = 0; j < hiddenLayer; j++) {
//                    float oldWeight = myNeuron[i].getWeight(j);
//                    myNeuron[i].setWeight(j, oldWeight + DeltaVjk[i][j]);
//                }
//            }
////
//            for(int i = 0; i < hiddenLayer; i++) {
//                for(int j = 0; j < outputLayer; j++) {
//                    float oldWeight = myHiddenNeuron[i].getWeight(j);
//                    myHiddenNeuron[i].setWeight(j, oldWeight + DeltaWjk[i][j]);
//                }
//            }
//        }
//        auto stopTraining = high_resolution_clock::now();
////
//
//
//
//    // ==================== TEST DIGITS ====================
//        int testDataSize    = 2000;
//        int *testDataDigit  = readDigitData("../../../train_digit.txt", testDataSize);
//        int **testDataBin   = readBinData("../../../train_data.txt", testDataSize);
//        int correct         = 0;
//        int uncorrect       = 0;
//
//        auto startTesting = high_resolution_clock::now();
//        for(int z = 0; z < testDataSize; z++) {
//            for(int i = 0; i < inputLayer; i++) {
//                float val = (float)testDataBin[z][i]/255;
//                float value = (int)(val * 1000 + .5);
//                myNeuron[i].setNodeValue( (float)value / 1000 );
//            }
//
//            for(int i=0 ; i < hiddenLayer; i++) {
//                double val = 0;
//                for(int j=0; j < inputLayer ; j++) {
//                   val += myNeuron[j].getWeight(i) * myNeuron[j].getNodeValue();
//                }
//                float sigmoid = 1 / (1 + exp(-val));
//                float value = (int)(sigmoid * 1000 + .5);
//
//                myHiddenNeuron[i].setNodeValue( (float)value / 1000 );
//            }
//
//            for(int i=0 ; i < outputLayer; i++) {
//                double val = 0;
//                for(int j = 0; j < hiddenLayer ; j++) {
//                   val += myHiddenNeuron[j].getWeight(i) * myHiddenNeuron[j].getNodeValue();
//                }
//                float sigmoid = 1 / (1 + exp(-val));
//                float value = (int)(sigmoid * 1000 + .5);
//                myOutputNeuron[i].setNodeValue(  (float)value / 1000 );
//            }
//
//
//            double *allPredictionsforDigits = new double[outputLayer];
//            for(int i=0 ; i < outputLayer; i++) {
//                allPredictionsforDigits[i] = myOutputNeuron[i].getNodeValue();
//            }
//
//            float maxAccuracy = 0;
//            int number;
//            for(int i=0 ; i < outputLayer; i++){
//                if(allPredictionsforDigits[i] >= maxAccuracy){
//                    maxAccuracy = allPredictionsforDigits[i];
//                    number = i;
//                }
//            }
//            std::cout<<"prediction "<<"["<<number<<"] for digit " <<testDataDigit[z] <<" with "<<maxAccuracy<<"% at position: "<<z<<"\n";
//            (number == testDataDigit[z]) ? correct++ : uncorrect++;
//        }
//
//        auto stopTesting = high_resolution_clock::now();
//        auto durationTest = duration_cast<milliseconds>(stopTesting - startTesting);
//        auto durationTrain = duration_cast<milliseconds>(stopTraining - startTraining);
//
//        std::cout << "Time taken to train data: "<< durationTrain.count() << " miliseconds" << std::endl;
//        std::cout << "Time taken to test data: "<< durationTest.count()<< " miliseconds" << std::endl;
//
//        std::cout << "poz: " << correct << std::endl << "neg: " << uncorrect << std::endl;
//        std::cout << "average: " << (float)correct/(correct+uncorrect)*100 << "%" << std::endl;
//
 





    
//    double valueMatrix[28][28];
//    int z = 0;
//    for(int i = 0; i < 28; i++){
//        for(int j = 0; j < 28; j++){
//            valueMatrix[i][j] = trainDataBin[1666][z];
//            z++;
//        }
//    }
//    for(int i = 0; i < 28; i++){
//        for(int j = 0; j < 28; j++){
//            valueMatrix[i][j] > 0 ? std::cout<<"X " : std::cout<<". ";
//        }
//        std::cout<<std::endl;
//    }
  
    
    
    
//    //std::string image_path = samples::findFile("/Users/diond/Desktop/Pics/a.jpg");
//    std::string image_path = samples::findFile("/Users/diond/Desktop/foto 3.png");
//    Mat img = imread(image_path, IMREAD_COLOR);
//
//    for(int z=0; z<1935; z++){
//        int arr2[28][28];
//        int l=0;
//        for(int i=0; i<28; i++){
//            for(int j=0; j<28; j++){
//                arr2[i][j]=trainDataBin[z][l];
//                l++;
//            }
//        }

//        if(img.empty()) {
//            std::cout << "Could not read the image: " << image_path << std::endl;
//            return 1;
//        }

//        cv::Vec3b color = img.at<Vec3b>(Point(0,0));
//        //std::cout<<color; --> result [255, 249, 250]
//
//        for(int i = 0; i < 28; i++)
//        {
//            for(int j = 0; j < 28; j++)
//            {
//                Vec3b bgrPixel = img.at<Vec3b>(j, i);
//                //img.at<Vec3b>(i+100, j+20) = 255;
//                if(arr2[i][j] == 0){
//                    color[0]=255;
//                    color[1]=255;
//                    color[2] =255;
//                }
//                else {
//                    color[0]= 255 - arr2[i][j];
//                    color[1]= 255 - arr2[i][j];
//                    color[2]= 255 - arr2[i][j];
//                }
//
//                // set a pixel back to the image
//
//                float c =  z/45;
//                int baba = ceil(c);
//                img.at<Vec3b>(Point(j+(z*28),i+(c*28))) = color;
//            }
//        }
//        imwrite("/Users/diond/Desktop/foto 3.png", img);
//    }



//	return 0;
//}
