#include "DGM.h"
#include "DNN.h"
#include <fstream>
namespace dgm = DirectGraphicalModels;

dgm::dnn::CNeuron* dotProd(int hiddenLayer, int inputLayer, dgm::dnn::CNeuron A[], dgm::dnn::CNeuron B[]);

int main() {
    const int inputLayer = 784;
    const int hiddenLayer = 60;
    const int outputLayer = 10;
    float rangeMin = -0.5;
    float rangeMax = 0.5;
    
    dgm::dnn::CNeuron *myNeuron = new dgm::dnn::CNeuron[inputLayer];
    dgm::dnn::CNeuron *myHiddenNeuron =  new dgm::dnn::CNeuron[hiddenLayer];
    dgm::dnn::CNeuron *myOutputNeuron = new dgm::dnn::CNeuron[outputLayer];
    
    int *trainDataDigit = readDigitData("../../../test_digits.txt", 2000);
    int **trainDataBin = readBinData("../../../test_data.txt", 2000);
    int **resultsArray = resultPredictions(outputLayer);
    
    for(int i = 0; i < inputLayer; i++) {
        myNeuron[i].generateWeights();
    }
    for(int i = 0; i < hiddenLayer; i++) {
        myHiddenNeuron[i].generateWeights();
    }
    
// ==================== Training Data ====================
    for(int k = 0; k < 2000; k++) {
        std::cout<<"Training for digit: "<< trainDataDigit[k]<<std::endl;
        
        for(int i = 0; i < inputLayer; i++) {
            float val = (float)trainDataBin[k][i]/255;
            float value = (int)(val * 1000 + .5);
            myNeuron[i].setNodeValue( (float)value / 1000 );
        }
        
        myHiddenNeuron = dotProd(hiddenLayer, inputLayer, myNeuron, myHiddenNeuron);
        myOutputNeuron = dotProd(outputLayer, hiddenLayer, myHiddenNeuron, myOutputNeuron);

        double *resultErrorRate = new double[outputLayer];        
        for(int i=0 ; i < outputLayer; i++) {
            resultErrorRate[i] = resultsArray[trainDataDigit[k]][i] - myOutputNeuron[i].getNodeValue();
        }
        
    // ==================== BACKPROPAGATION ====================
        float learningRate = 0.1;
        float (*DeltaWjk)[outputLayer] = new float[hiddenLayer][outputLayer];
        float *DeltaIn_j = new float[hiddenLayer];
        float *DeltaJ = new float[hiddenLayer];
        float (*DeltaVjk)[hiddenLayer] = new float[inputLayer][hiddenLayer];
        
        //updates weights between [hiddenLayer][outputLayer]
        for(int i = 0; i < hiddenLayer; i++) {
            double val = 0;
            for(int j = 0; j < outputLayer; j++) {
                val += myHiddenNeuron[i].getWeight(j) * resultErrorRate[j];
                DeltaWjk[i][j] = learningRate * resultErrorRate[j] * myHiddenNeuron[i].getNodeValue();
            }
            DeltaIn_j[i] = val;
        }

        //still hiddenlayer nodes
        for(int i = 0; i < hiddenLayer; i++) {
            float sigmoid = 1 / (1 + exp(myHiddenNeuron[i].getNodeValue()));
            float inverse = 1 - sigmoid;
            DeltaJ[i] = DeltaIn_j[i] * sigmoid * inverse;
        }
        
        //updates weights between [inputLayer][hiddenLayer]
        for(int i = 0; i < inputLayer; i++) {
            for(int j = 0; j < hiddenLayer; j++) {
                DeltaVjk[i][j] = learningRate * DeltaJ[j] * myNeuron[i].getNodeValue();
            }
        }

    // ==================== UPDATE WEIGHTS ====================
        for(int i = 0; i < inputLayer; i++) {
            for(int j = 0; j < hiddenLayer; j++) {
                float oldWeight = myNeuron[i].getWeight(j);
                myNeuron[i].setWeight(j, oldWeight + DeltaVjk[i][j]);
            }
        }
        
        for(int i = 0; i < hiddenLayer; i++) {
            for(int j = 0; j < outputLayer; j++) {
                float oldWeight = myHiddenNeuron[i].getWeight(j);
                myHiddenNeuron[i].setWeight(j, oldWeight + DeltaWjk[i][j]);
            }
        }
    }
    
// ==================== TEST DIGITS ====================

    int testDataSize = 2000;
    int *testDataDigit = readDigitData("../../../train_digit.txt", testDataSize);
    int **testDataBin = readBinData("../../../train_data.txt", testDataSize);
    int correct = 0;
    int uncorrect = 0;

    for(int z = 0; z < testDataSize; z++) {
        for(int i = 0; i < inputLayer; i++) {
            float val = (float)testDataBin[z][i]/255;
            float value = (int)(val * 1000 + .5);
            myNeuron[i].setNodeValue( (float)value / 1000 );
        }

        myHiddenNeuron = dotProd(hiddenLayer, inputLayer, myNeuron, myHiddenNeuron);
        myOutputNeuron = dotProd(outputLayer, hiddenLayer, myHiddenNeuron, myOutputNeuron);

        double *allPredictionsforDigits = new double[outputLayer];
        for(int i=0 ; i < outputLayer; i++) {
            allPredictionsforDigits[i] = myOutputNeuron[i].getNodeValue();
        }

        float maxAccuracy = 0;
        int number;
        for(int i=0 ; i < outputLayer; i++){
            if(allPredictionsforDigits[i] >= maxAccuracy){
                maxAccuracy = allPredictionsforDigits[i];
                number = i;
            }
        }

        std::cout<<"prediction "<<"["<<number<<"] for digit " <<testDataDigit[z]
        <<" with "<<maxAccuracy<<"% at position: "<<z<<std::setw(5);
        if (number == testDataDigit[z]) {
            correct++;
            std::cout<<std::endl;
        }
        else {
            uncorrect++;
            std::cout<<"[x]"<<std::endl;
        }
    }
    
    std::cout << "poz: " << correct << std::endl << "neg: " << uncorrect << std::endl;
    std::cout <<"average: " << (float)correct/(correct+uncorrect)*100 << "%" << std::endl;
	return 0;
}


dgm::dnn::CNeuron* dotProd(int hiddenLayer, int inputLayer, dgm::dnn::CNeuron A[], dgm::dnn::CNeuron B[]) {
    for(int i=0 ; i < hiddenLayer; i++) {
        double val = 0;
        for(int j=0; j < inputLayer ; j++) {
           val += A[j].getWeight(i) * A[j].getNodeValue();
        }
        float value = applySigmoidFunction(val);
        B[i].setNodeValue(value);
    }
    return B;
}
