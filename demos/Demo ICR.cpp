#include "DGM.h"
#include "DNN.h"

#include <iostream>
#include <fstream>
#include <string>
//using namespace std;
namespace dgm = DirectGraphicalModels;

float applySigmoidFunction(float val){
    float sigmoid = 1 / (1 + exp(-val));
    float value = (int)(sigmoid * 10000 + .5);
    float result = (float)value / 10000;
    return result;
}



int main() {
//	dgm::dnn::CNeuron neuron;
        
// ==================== Read the MNIST data ====================

    
    static int trainDataBin[2000][784];
    static int trainDataDigit[2000];
    
    std::string fileBinData = "../../../bin_data.txt";
    std::string fileDigitData = "../../../digit_data.txt";
    
    std::ifstream inFile, inFile2;
    inFile.open(fileBinData.c_str());
    inFile2.open(fileDigitData.c_str());

    if (inFile2.is_open()) {
        for (int i = 0; i < 2000; i++) {
            inFile2 >> trainDataDigit[i];
        }
        inFile2.close();
    }
    
    if (inFile.is_open()) {
        for(int i = 0 ; i < 2000; i++) {
            for (int j = 0; j < 784; j++) {
                inFile >> trainDataBin[i][j];
            }
        }
        inFile.close();
    }


// ==================== Visualize Digit ====================

//    double valueMatrix[28][28];
//    int z = 0;
//    for(int i = 0; i < 28; i++){
//        for(int j = 0; j < 28; j++){
//            valueMatrix[i][j] = trainDataBin[0][z];
//            z++;
//        }
//    }
//
//    for(int i = 0; i < 28; i++){
//        for(int j = 0; j < 28; j++){
//            valueMatrix[i][j] > 0 ? std::cout<<"X " : std::cout<<". ";
//        }
//        std::cout<<std::endl;
//    }


// ==================== matrix to compare the output in the end ====================

    int result[10][10];
    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 10; j++) {
            (i == j) ? result[i][j] = 1 : result[i][j] = 0;
        }
    }

    
// ==================== Create Neurons ====================

    const byte nStates = 10;
    
    int inputLayer = 784;
    int hiddenLayer = 60;
    int outputLayer = nStates; //10
    
    //[-0.1 to 0.1] 88%
    float rangeMin = -0.5;
    float rangeMax = 0.5;
    
    dgm::dnn::CNeuron myNeuron[inputLayer];
    dgm::dnn::CNeuron myHiddenNeuron[hiddenLayer];
    dgm::dnn::CNeuron myOutputNeuron[outputLayer];
 
    
// ==================== Set Initial Weights ====================

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    srand(seed);
 
        for(int i = 0; i < inputLayer; i++) {
            for(int j = 0; j < hiddenLayer; j++) {
                double f = (double)rand() / RAND_MAX;
                double var = rangeMin + f * ((rangeMax) - (rangeMin));
                myNeuron[i].setWeight(j, var);
            }
        }
        for(int i = 0; i < hiddenLayer; i++) {
            for(int j = 0; j < outputLayer; j++) {
                double f = (double)rand() / RAND_MAX;
                double var = rangeMin + f * ((rangeMax) - (rangeMin));
                myHiddenNeuron[i].setWeight(j, var);
            }
        }
    
//    for(int i = 0; i < inputLayer; i++) {
//        myNeuron[i].generateWeights();
//    }
//    for(int i = 0; i < hiddenLayer; i++) {
//        myHiddenNeuron[i].generateWeights();
//    }
    
//    for(int i = 0; i < hiddenLayer; i++) {
//        for(int j=0; j< hiddenLayer; j++) {
//            std::cout<<myNeuron[i].getWeight(j)<< " ";
//        }
//        std::cout<<std::endl;
//    }


    
// ==================== Start Training Data ====================

for(int k = 0; k < 2000; k++) {
    std::cout<<"Training for digit: "<< trainDataDigit[k]<<std::endl;

    for(int i = 0; i < inputLayer; i++) {
        float val = (float)trainDataBin[k][i]/255;
        float value = (int)(val * 1000 + .5);
        myNeuron[i].setNodeValue( (float)value / 1000 ); 
    }
    

    //inputWeightMatrix dotProduct inputMatrix
    //[36 x 784] [784 x 1] --> Hidden layerMatrix [36, 1]
    for(int i=0 ; i < hiddenLayer; i++) {
        double val = 0;
        for(int j=0; j < inputLayer ; j++) {
           val += myNeuron[j].getWeight(i) * myNeuron[j].getNodeValue();
        }
        float value = applySigmoidFunction(val);
        myHiddenNeuron[i].setNodeValue(value);
    }
    

    
    // HiddenOutputMatrix dotProduct Hidden node
    //  [10 x 36] [36 x 1] --> Output layerMatrix [10, 1]
    for(int i=0 ; i < outputLayer; i++) {
        double val = 0;
        for(int j = 0; j < hiddenLayer ; j++) {
           val += myHiddenNeuron[j].getWeight(i) * myHiddenNeuron[j].getNodeValue();
        }
        float value = applySigmoidFunction(val);
        myOutputNeuron[i].setNodeValue(value);
    }


    
    double resultErrorRate[10];
    
    for(int i=0 ; i < outputLayer; i++) {
        resultErrorRate[i] = result[trainDataDigit[k]][i] - myOutputNeuron[i].getNodeValue();
    }
    
    
// ==================== BACKPROPAGATION ====================
    
    float alpha = 0.1; //learning rate
    
    //updates weights between [hiddenLayer][outputLayer]
    float DeltaWjk[hiddenLayer][outputLayer];
    for(int i = 0; i < hiddenLayer; i++) {
        for(int j = 0; j < outputLayer; j++) {
            DeltaWjk[i][j] = alpha * resultErrorRate[j] * myHiddenNeuron[i].getNodeValue();
            //do also for bias
        }
    }

    //updates the values in the hiddenLayer
    float DeltaIn_j[hiddenLayer];
    for(int i = 0; i < hiddenLayer; i++) {
        double val = 0;
        for(int j = 0; j < outputLayer; j++) {
            val += myHiddenNeuron[i].getWeight(j) * resultErrorRate[j];
        }
        DeltaIn_j[i] = val;
    }

    
    //still hiddenlayer nodes
    float DeltaJ[hiddenLayer];
    for(int i = 0; i < hiddenLayer; i++) {
        //derivative of sigmoid ... f'(hiddenNode value)
        float sigmoid = 1 / (1 + exp(myHiddenNeuron[i].getNodeValue()));
        float inverse = 1 - sigmoid;
        DeltaJ[i] = DeltaIn_j[i] * sigmoid * inverse;
    }

    
    //updates weights between [inputLayer][hiddenLayer]
    float DeltaVjk[inputLayer][hiddenLayer];
    for(int i = 0; i < inputLayer; i++) {
        for(int j = 0; j < hiddenLayer; j++) {
            DeltaVjk[i][j] = alpha * DeltaJ[j] * myNeuron[i].getNodeValue();
            //do also for bias
        }
    }

    

// ==================== UPDATE WEIGHTS ====================

    for(int i = 0; i < inputLayer; i++) {
        for(int j = 0; j < hiddenLayer; j++) {
            float oldWeight = myNeuron[i].getWeight(j);
            myNeuron[i].setWeight(j, oldWeight + DeltaVjk[i][j]);
            //also for bias
        }
    }
    
    for(int i = 0; i < hiddenLayer; i++) {
        for(int j = 0; j < outputLayer; j++) {
            float oldWeight = myHiddenNeuron[i].getWeight(j);
            myHiddenNeuron[i].setWeight(j, oldWeight + DeltaWjk[i][j]);
            //also biases
        }
    }
}
    
    
// ==================== TEST DIGITS ====================

    int testDataSize = 2000;
    
    int testDataBin[testDataSize][784];
    int testDataDigit[testDataSize];

    //this gives 92% accuracy with 10000
    //std::string testfileBinData = "../../../test_bin.txt";
    //std::string testfileDigitData = "../../../test_digit.txt";

    std::string testfileBinData = "../../../train_data.txt";
    std::string testfileDigitData = "../../../train_digit.txt";
    
    std::ifstream testinFile, testinFile2;
    testinFile.open(testfileBinData.c_str());
    testinFile2.open(testfileDigitData.c_str());

    if (testinFile2.is_open()) {
        for (int i = 0; i < testDataSize; i++) {
            testinFile2 >> testDataDigit[i];
        }
        testinFile2.close();
    }

    if (testinFile.is_open()) {
        for(int i = 0 ; i < testDataSize; i++) {
            for (int j = 0; j < 784; j++) {
                testinFile >> testDataBin[i][j];
            }
        }
        testinFile.close();
    }

    std::cout<<"\n";

    int poz = 0;
    int neg = 0;

    //int numrat[10]={0,0,0,0,0,0,0,0,0,0};


for(int z = 0; z < testDataSize; z++) {
        for(int i = 0; i < inputLayer; i++) {
            float val = (float)testDataBin[z][i]/255;
            float value = (int)(val * 1000 + .5);
            myNeuron[i].setNodeValue( (float)value / 1000 );
        }

        for(int i=0 ; i < hiddenLayer; i++) {
            double val = 0;
            for(int j=0; j < inputLayer ; j++) {
               val += myNeuron[j].getWeight(i) * myNeuron[j].getNodeValue();
            }
            float value = applySigmoidFunction( val);
            myHiddenNeuron[i].setNodeValue(value);
        }
    
        for(int i=0 ; i < outputLayer; i++) {
            double val = 0;
            for(int j = 0; j < hiddenLayer ; j++) {
               val += myHiddenNeuron[j].getWeight(i) * myHiddenNeuron[j].getNodeValue();
            }
            float value = applySigmoidFunction(val);
            myOutputNeuron[i].setNodeValue(value);
        }

        double allPredictionsforDigits[10];

        for(int i=0 ; i < outputLayer; i++) {
            allPredictionsforDigits[i] = myOutputNeuron[i].getNodeValue();
        }


        float max = 0;
        float secondMax = 0;
        int vlera2;
        int vlera;
        for(int i=0 ; i < outputLayer; i++){
            if(allPredictionsforDigits[i] >= max){
                max = allPredictionsforDigits[i];
                vlera = i;
            }
        }

    
    if(vlera == testDataDigit[z]) {
        std::cout<<"prediction "<<"["<<vlera<<"] for " << testDataDigit[z]<<" with "<<max<<"% at position: "<<z<<std::endl;
        poz++;
    }
                
    else{
        std::cout<<"prediction "<<"["<<vlera<<"] for " << testDataDigit[z]<<" with "<<max<<"% at position: "<<z<<std::setw(5)<<"[x]"<<std::endl;
        neg++;
        //numrat[testDataDigit[z]] += 1;
    }
}

    std::cout << "poz: " << poz << std::endl;
    std::cout << "nez: " << neg << std::endl;
    std::cout <<"average: " << (float)poz/(poz+neg)*100 << "%" << std::endl;
    
    
//    for(int i=0; i<10; i++) {
//        cout<< "wrong ["<<i<<"] : "<< numrat[i]<<endl;
//    }





    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
//      Create the transposed Weight Matrix [36 x 784]
//        double weightMatrix [hiddenLayer][inputLayer];
//        for(int i = 0; i < hiddenLayer; i++) {
//            for(int j = 0; j < inputLayer;  j++) {
//                weightMatrix[i][j] = myNeuron[j].getWeight(i);
//                cout<<weightMatrix[i][j]<< " ";
//            }
//            cout<<"\n";
//        }

    
    
    
    //testdatabin[5] = 9 -> 4
    //[19] = 7 -> 4
    //[47] = 7 -> 2
    //[64] = 8 -> 3
    //[278] = 5-> 6
    //[441] = 9 -> 7
    //[784] 6-> 5
    
//    int arr2[28][28];
//    int l=0;
//
//        for(int i=0; i<28; i++){
//            for(int j=0; j<28; j++){
//                arr2[i][j]=testDataBin[784][l];
//                l++;
//            }
//        }
//
//    for(int i = 0; i < 28; i++){
//        for(int j = 0; j < 28; j++){
//            arr2[i][j] > 0 ? cout<<"X " : cout<<". ";
//        }
//        cout<<endl;
//    }
    
    
 
    
    
    
    
    
    
    


//    //std::string image_path = samples::findFile("/Users/diond/Desktop/Pics/a.jpg");
//    std::string image_path = samples::findFile("/Users/diond/Desktop/foto1.png");
//    Mat img = imread(image_path, IMREAD_COLOR);
    
    
//
//for(int z=0; z<1000; z++){
//
//    int arr2[28][28];
//    int l=0;
//
//        for(int i=0; i<28; i++){
//            for(int j=0; j<28; j++){
//                arr2[i][j]=testDataBin[z][l];
//                l++;
//            }
//        }
//
//    if(img.empty())
//    {
//        std::cout << "Could not read the image: " << image_path << std::endl;
//        return 1;
//    }
//
//    cv::Vec3b color = img.at<Vec3b>(Point(0,0));
//    //std::cout<<color; --> result [255, 249, 250]
//
//    for(int i = 0; i < 28; i++)
//    {
//        for(int j = 0; j < 28; j++)
//        {
//            Vec3b bgrPixel = img.at<Vec3b>(j, i);
//            //img.at<Vec3b>(i+100, j+20) = 255;
//            if(arr2[i][j] == 0){
//                color[0]=255;
//                color[1]=255;
//                color[2] =255;
//            }
//            else {
//                color[0]= 255 - arr2[i][j];
//                color[1]= 255 - arr2[i][j];
//                color[2]= 255 - arr2[i][j];
//            }
//
//            // set a pixel back to the image
//
//            float c =  z/45;
//            int baba = ceil(c);
//            img.at<Vec3b>(Point(j+(z*28),i+(c*28))) = color;
//        }
//    }
//
//    imwrite("/Users/diond/Desktop/foto1.png", img);
//}

    
    

	return 0;
}
