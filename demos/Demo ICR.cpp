#include "DGM.h"
#include "DNN.h"

#include <iostream>
#include <fstream>
#include <string>
using namespace std;
namespace dgm = DirectGraphicalModels;


int main() {
    
//	dgm::dnn::CNeuron neuron;
    
//@ Read the binary and its corresponding digit from files
/*xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx*/
    
    static int trainDataBin[2000][784];
    static int trainDataDigit[2000];
    
    string fileBinData = "../../../bin_data.txt";
    string fileDigitData = "../../../digit_data.txt";
    
    ifstream inFile, inFile2;
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
/*xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx*/


//@ Create a matrix so you can visually see the number
/*xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx*/
    
//    srand (time(NULL));
//    int v1 = rand() % 1000;
    
    double valueMatrix[28][28];
    int z = 0;
    for(int i = 0; i < 28; i++){
        for(int j = 0; j < 28; j++){
            valueMatrix[i][j] = trainDataBin[0][z];
            z++;
        }
    }

    for(int i = 0; i < 28; i++){
        for(int j = 0; j < 28; j++){
            valueMatrix[i][j] > 0 ? cout<<"X " : cout<<". ";
        }
        cout<<endl;
    }
/*xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx*/


    
//@ Create the matrix to compare the results in the end of each iteration
/*xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx*/
    int result[10][10];
    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 10; j++) {
            (i == j) ? result[i][j] = 1 : result[i][j] = 0;
        }
    }
/*xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx*/


    
    int inputLayer = 784;
    int hiddenLayer = 60; //76
    int outputLayer = 10;
    
    dgm::dnn::CNeuron myNeuron[inputLayer];
    dgm::dnn::CNeuron myHiddenNeuron[hiddenLayer];
    dgm::dnn::CNeuron myOutputNeuron[outputLayer];
 
    
//@ Set initial random weights
/*xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx*/
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    srand(seed);
 
    for(int i = 0; i < inputLayer; i++) {
        for(int j = 0; j < hiddenLayer; j++) {
            double f = (double)rand() / RAND_MAX;
            double var = -0.5 + f * ((0.5) - (-0.5));
            myNeuron[i].setWeight(j, var);
        }
    }
    
    for(int i = 0; i < hiddenLayer; i++) {
        for(int j = 0; j < outputLayer; j++) {
            double f = (double)rand() / RAND_MAX;
            double var = -0.5 + f * ((0.5) - (-0.5));
            myHiddenNeuron[i].setWeight(j, var);
        }
    }
/*xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx*/


    
for(int k = 0; k < 2000; k++) {
    
    cout<<"Training for digit: "<< trainDataDigit[k]<<endl;

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
        float sigmoid = 1 / (1 + exp(-val));
        float value = (int)(sigmoid * 1000 + .5);
        myHiddenNeuron[i].setNodeValue((float)value / 1000);
    }
    


    // HiddenOutputMatrix dotProduct Hidden node
    //  [10 x 36] [36 x 1] --> Output layerMatrix [10, 1]
    for(int i=0 ; i < outputLayer; i++) {
        double val = 0;
        for(int j = 0; j < hiddenLayer ; j++) {
           val += myHiddenNeuron[j].getWeight(i) * myHiddenNeuron[j].getNodeValue();
        }
        float sigmoid = 1 / (1 + exp(-val));
        float value = (int)(sigmoid * 1000 + .5);
        myOutputNeuron[i].setNodeValue((float)value / 1000);
    }


    
    double resultErrorRate[10];
    
    for(int i=0 ; i < outputLayer; i++) {
        resultErrorRate[i] = result[trainDataDigit[k]][i] - myOutputNeuron[i].getNodeValue();
//        if (i == trainDataDigit[k]) {
//            cout<<"--> ["<<i<<"] "<<setw(6)<< myOutputNeuron[i].getNodeValue()<<" ("<< resultErrorRate[i]<< ")\n";
//        }
//        else {
//            cout<<"["<<i<<"] "<<setw(6)<< myOutputNeuron[i].getNodeValue()<<" ("<< resultErrorRate[i]<< ")\n";
//        }
    }
    
    
    // BACKPROPAGATION

    //i bon update weights
    float DeltaWjk[hiddenLayer][outputLayer];
    float alpha = 0.1; //change alpha

    for(int i = 0; i < hiddenLayer; i++) {
        for(int j = 0; j < outputLayer; j++) {
            DeltaWjk[i][j] = alpha * resultErrorRate[j] * myHiddenNeuron[i].getNodeValue();
            //do also for bias
        }
    }


    //update Weights (1)
    float DeltaIn_j[hiddenLayer];
    for(int i = 0; i < hiddenLayer; i++) {
        double val = 0;
        for(int j = 0; j < outputLayer; j++) {
            val += myHiddenNeuron[i].getWeight(j) * resultErrorRate[j];
        }
        DeltaIn_j[i] = val;
    }

    
    float DeltaJ[hiddenLayer];
    for(int i = 0; i < hiddenLayer; i++) {
        //derivative of sigmoid ... f'(hiddenNode value)
        float sigmoid = 1 / (1 + exp(myHiddenNeuron[i].getNodeValue()));
        float inverse = 1 - sigmoid;
        DeltaJ[i] = DeltaIn_j[i] * sigmoid * inverse;
    }

    
    //this updates the weights (2)
    float DeltaVjk[inputLayer][hiddenLayer];
    for(int i = 0; i < inputLayer; i++) {
        for(int j = 0; j < hiddenLayer; j++) {
            DeltaVjk[i][j] = alpha * DeltaJ[j] * myNeuron[i].getNodeValue();
            //do also for bias
        }
    }

    
    //UPDATE WEIGHTS AND BIASES
    
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
    
    
    
    
    
    
    
    
    
    
    /*xxxxxxxxxxxxxx        TEST DATA         xxxxxxxxxxxxxxxx*/
    
    
    int testDataBin[2000][784];
    int testDataDigit[2000];

    //string testfileBinData = "../../../test_bin.txt";
    //string testfileDigitData = "../../../test_digit.txt";

    string testfileBinData = "../../../train_data.txt";
    string testfileDigitData = "../../../train_digit.txt";

    ifstream testinFile, testinFile2;
    testinFile.open(testfileBinData.c_str());
    testinFile2.open(testfileDigitData.c_str());

    if (testinFile2.is_open()) {
        for (int i = 0; i < 2000; i++) {
            testinFile2 >> testDataDigit[i];
        }
        testinFile2.close();
    }

    if (testinFile.is_open()) {
        for(int i = 0 ; i < 2000; i++) {
            for (int j = 0; j < 784; j++) {
                testinFile >> testDataBin[i][j];
            }
        }
        testinFile.close();
    }

    cout<<"\n";

    int poz = 0;
    int neg = 0;


//    int numrat[10]={0,0,0,0,0,0,0,0,0,0};


for(int z = 0; z < 2000; z++)
{

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
            float sigmoid = 1 / (1 + exp(-val));
            float value = (int)(sigmoid * 1000 + .5);

            myHiddenNeuron[i].setNodeValue( (float)value / 1000 );
        }
    
        for(int i=0 ; i < outputLayer; i++) {
            double val = 0;
            for(int j = 0; j < hiddenLayer ; j++) {
               val += myHiddenNeuron[j].getWeight(i) * myHiddenNeuron[j].getNodeValue();
            }
            float sigmoid = 1 / (1 + exp(-val));
            float value = (int)(sigmoid * 1000 + .5);
            myOutputNeuron[i].setNodeValue(  (float)value / 1000 );
        }


        double resultErrorRate[10];

        for(int i=0 ; i < outputLayer; i++) {
            resultErrorRate[i] = myOutputNeuron[i].getNodeValue();
        }


        float max = 0;
        int vlera;
        for(int i=0 ; i < outputLayer; i++){
            if(resultErrorRate[i] >= max){
                max = resultErrorRate[i];
                vlera = i;
            }
        }

//    cout<<"prediction "<<"["<<vlera<<"] for " << testDataDigit[z]<<" with %: "<<max<<" at position: "<<z<<endl;
//        vlera == testDataDigit[z] ? poz++ : neg++;
    
    
    
    if(vlera == testDataDigit[z]) {
        cout<<"prediction "<<"["<<vlera<<"] for " << testDataDigit[z]<<" at "<<max<<"% at position: "<<z<<endl;
        poz++;
    }
    else {
//        numrat[testDataDigit[z]] += 1;
        
        cout<<"prediction "<<"["<<vlera<<"] for " << testDataDigit[z]<<" at "<<max<<"% at position: "<<z<<setw(5)<<"[x]"<<endl;
        neg++;
    }
}

    cout << "poz: " << poz << endl;
    cout << "nez: " << neg << endl;
    cout <<"average: " << (float)poz/(poz+neg)*100 << "%" <<endl;
    
    
//    for(int i=0; i<10; i++) {
//        cout<< "wrong ["<<i<<"] : "<< numrat[i]<<endl;
//    }






    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    //  Create the transposed Weight Matrix [36 x 784]
    //    double weightMatrix [hiddenLayer][inputLayer];
    //    for(int i = 0; i < hiddenLayer; i++) {
    //        for(int j = 0; j < inputLayer;  j++) {
    //            weightMatrix[i][j] = myNeuron[j].getWeight(i);
    //            cout<<weightMatrix[i][j]<< " ";
    //        }
    //        cout<<"\n";
    //    }

    
    
    
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
    
    
 
    
    
    
    
    
    
    
//
//
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
