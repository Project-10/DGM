#pragma once
#include<string>

//#include "DGM.h"
//#include "DNN.h"
//namespace dgm = DirectGraphicalModels;

/**
 * Applies the Sigmoid Activation function
 *
 * @param the value at each node
 * @return a number between 0 and 1.
 */
float applySigmoidFunction(float val);

/**
 * Reads the digits numerical value in a decimal notation
 *
 * @param file to read, and the number of digits to read
 * @return an array of digits
 */
int *readDigitData(std::string file, int dataSize);

/**
 * Reads the digits pixel values (784 values for a digit)
 *
 * @param file to read, and the number of digits to read
 * @return a 2D array of digits with their pixel values 2D[dataSize][784]
 */
int **readBinData(std::string file, int dataSize);

//@returns an array with the correct firing output nodes for each digit (0-9)
/**
 * Gives the correct firing output nodes for each digit so we can find the errorRate*
 * Example when comparing the output nodes in the end, the correct prediction for
 * number 2 lets say is : {0,0,1,0,0,0,0,0,0,0,0}
 *
 * @param the output digits value (10)
 * @return an array of numbers filled with 0's for input 'x' excepted with a single 1 in the index[x].
 */
int **resultPredictions(int outputDigits);


//dgm::dnn::CNeuron* dotProd(int hiddenLayer, int inputLayer, dgm::dnn::CNeuron A[], dgm::dnn::CNeuron B[]);
