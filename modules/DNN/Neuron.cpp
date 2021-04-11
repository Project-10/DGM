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
 * @return an array of digit labels
 */
int *readDigitData(std::string file, int dataSize);

/**
 * Reads the image pixel value
 *
 * @param file to read, and the number of digits to read
 * @return an array of pixel values for each image
 */
int **readImgData(std::string file, int dataSize);

/**
 * Gives the correct firing output nodes for each digit so we can find the errorRate*
 * Example when comparing the output nodes in the end, the correct prediction for
 * number 2 lets say is : {0,0,1,0,0,0,0,0,0,0,0}
 *
 * @param the output digits value (10)
 * @return an array of numbers filled with 0's for input 'x' excepted with a single 1 in the index[x].
 */
int **resultPredictions(int outputLayer);

/**
 * Reads the digits pixel value in a decimal notation
 *
 * @param file to read, and the number of digits to read
 * @return an array of digits
 */
//int **readBinData(std::string file, int dataSize);
