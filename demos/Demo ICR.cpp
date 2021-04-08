#include "DNN.h"
#include <fstream>
namespace dgm = DirectGraphicalModels;
using namespace std::chrono;

/**
 * @brief Applies the Sigmoid Activation function
 * @param val value at each node
 * @return a number between 0 and 1.
 */
double applySigmoidFunction(double val)
{
	double sigmoid = 1 / (1 + exp(-val));
	
	//TODO: why this is needed ?
	double value = (int)(sigmoid * 10000 + .5);
	double result = value / 10000;
	
	return result;
}

//TODO: First: I suggest to apply this method for the neurons in layer B invidiually, i.e. move the first loop from the method body to the user code
//TODO: Second: It makes sense to move this method to the CNeuron class, since it will update its value
//TODO: Third: Once this method in the CNeuron class, it makes sense to move tha activation function there as well. It is posible to use std::function for that
void dotProd(std::vector<dgm::dnn::ptr_neuron_t>& vpLayerA, std::vector<dgm::dnn::ptr_neuron_t>& vpLayerB)
{
	for(size_t i = 0 ; i < vpLayerB.size(); i++) {
		
		double value = 0;
		for(const auto& a : vpLayerA)
			value += a->getWeight(i) * a->getNodeValue();
	
		value = applySigmoidFunction(value);
		
		vpLayerB[i]->setNodeValue(value);
	}
	
}

int main() {
    const size_t 	numNeuronsInputLayer   = 784;
    const size_t 	numNeuronsHiddenLayer  = 60;
    const size_t 	numNeuronsOutputLayer  = 10;

    std::vector<dgm::dnn::ptr_neuron_t> vpInputLayer;
    std::vector<dgm::dnn::ptr_neuron_t> vpHiddenLayer;
    std::vector<dgm::dnn::ptr_neuron_t> vpOutputLayer;
    
	for (size_t i = 0; i < numNeuronsInputLayer; i++) {
		double value = 1; // TODO: read the correct input values from the digit images
		vpInputLayer.push_back( std::make_shared<dgm::dnn::CNeuron>(numNeuronsHiddenLayer, value) );
	}
	
	for (size_t i = 0; i < numNeuronsHiddenLayer; i++)
		vpHiddenLayer.push_back( std::make_shared<dgm::dnn::CNeuron>(numNeuronsOutputLayer) );
	
	for (size_t i = 0; i < numNeuronsOutputLayer; i++)
		vpOutputLayer.push_back( std::make_shared<dgm::dnn::CNeuron>(0) );
	

	// Example 1
	for(size_t i = 0; i < vpInputLayer.size(); i++) {
		double value = i * 5 - 3;
		vpInputLayer[i]->setNodeValue( value / 1000 );
		vpInputLayer[i]->generateRandomWeights();
	}
	
	// Example 2
	for (size_t i = 0; i < vpInputLayer.size(); i++)
		vpInputLayer[i]->generateRandomWeights();
	
	// Example 3
	for (auto neuron: vpInputLayer) {
		//numNeuronsHiddenLayer shpuld be equal to neuron->getSize()
		//printf("number of weights: %d\n", neuron->getSize());
		for (size_t i = 0; i < neuron->getSize(); i++)
			printf("%.2f ", neuron->getWeight(i));
		printf("\n");
	}
	

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
//            myHiddenNeuron = dotProd(hiddenLayer, inputLayer, myNeuron, myHiddenNeuron);
//            myOutputNeuron = dotProd(outputLayer, hiddenLayer, myHiddenNeuron, myOutputNeuron);
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
//            myHiddenNeuron = dotProd(hiddenLayer, inputLayer, myNeuron, myHiddenNeuron);
//            myOutputNeuron = dotProd(outputLayer, hiddenLayer, myHiddenNeuron, myOutputNeuron);
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
//    //        std::cout<<"prediction "<<"["<<number<<"] for digit " <<testDataDigit[z] <<" with "<<maxAccuracy<<"% at position: "<<z<<std::setw(5);
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
	return 0;
}
