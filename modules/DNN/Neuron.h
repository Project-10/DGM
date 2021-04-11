#pragma once
#include "types.h"
#include <fstream>

namespace DirectGraphicalModels {
    namespace dnn
    {
        class CNeuron
        {
        public:
            DllExport CNeuron(void) = delete;
            /**
             * @brief Constructor
             * @param size
             * @param value
             */
            DllExport CNeuron(size_t size, float value = 0);
            DllExport CNeuron(const CNeuron&) = delete;
            DllExport ~CNeuron(void) = default;
            
            DllExport bool        operator=(const CNeuron&) = delete;
            
            DllExport void         generateRandomWeights(void);
            
            // Accessors
            DllExport void        setNodeValue(float value) { m_value = value; }
            DllExport float       getNodeValue(void) const { return m_value; }
            DllExport void        setWeight(size_t index, float weight);
            DllExport float       getWeight(size_t index) const;
            DllExport size_t      getSize(void) const { return m_vWeights.size(); }
            
        private:
            float                 m_value;
            std::vector<float>    m_vWeights;
        };
    
        using ptr_neuron_t = std::shared_ptr<CNeuron>;
    }
}

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
