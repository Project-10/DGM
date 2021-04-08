#pragma once
#include "types.h"

namespace DirectGraphicalModels {
    namespace dnn
    {
        //TODO: Please add comments to this class
		class CNeuron
        {
        public:
			DllExport CNeuron(void) = delete;
			/**
			 * @brief Constructor
			 * @param size
			 * @param value
			 */
			DllExport CNeuron(size_t size, double value = 0);
			DllExport CNeuron(const CNeuron&) = delete;
			DllExport ~CNeuron(void) = default;
            
			DllExport bool		operator=(const CNeuron&) = delete;
			
			DllExport void 		generateRandomWeights(void);
			
			// Accessors
			DllExport void		setNodeValue(double value) { m_value = value; }
			DllExport double	getNodeValue(void) const { return m_value; }
			DllExport void 		setWeight(size_t index, double weight);
			DllExport double 	getWeight(size_t index) const;
			DllExport size_t 	getSize(void) const { return m_vWeights.size(); }
            
			
        private:
			double				m_value;		///<
            std::vector<double>	m_vWeights;		///<
        };
	
		using ptr_neuron_t = std::shared_ptr<CNeuron>;
    }
}


//
// * Reads the digits numerical value in a decimal notation
// *
// * @param file to read, and the number of digits to read
// * @return an array of digits
// */
//int *readDigitData(std::string file, int dataSize);
//
///**
// * Reads the digits pixel values (784 values for a digit)
// *
// * @param file to read, and the number of digits to read
// * @return a 2D array of digits with their pixel values 2D[dataSize][784]
// */
//int **readBinData(std::string file, int dataSize);
//
////@returns an array with the correct firing output nodes for each digit (0-9)
///**
// * Gives the correct firing output nodes for each digit so we can find the errorRate*
// * Example when comparing the output nodes in the end, the correct prediction for
// * number 2 lets say is : {0,0,1,0,0,0,0,0,0,0,0}
// *
// * @param the output digits value (10)
// * @return an array of numbers filled with 0's for input 'x' excepted with a single 1 in the index[x].
// */
//int **resultPredictions(int outputDigits);




//int *readDigitData(std::string file, int dataSize) {
//    int *trainDataDigit = new int[dataSize];
//    std::string fileDigitData = file;
//
//    std::ifstream inFile;
//    inFile.open(fileDigitData.c_str());
//
//    if (inFile.is_open()) {
//        for (int i = 0; i < dataSize; i++) {
//            inFile >> trainDataDigit[i];
//        }
//        inFile.close();
//    }
//    return trainDataDigit;
//}
//
//
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
//
//
//int **resultPredictions(int outputLayer) {
//    int **result = new int*[outputLayer];
//
//    for(int i = 0; i < 10; i++) {
//        result[i] = new int[outputLayer];
//
//        for(int j = 0; j < 10; j++) {
//            result[i][j] = (i == j) ?  1 :  0;
//        }
//    }
//    return result;
//}





















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
