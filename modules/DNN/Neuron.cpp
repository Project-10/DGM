#include "Neuron.h"
#include "DGM/random.h"
#include "macroses.h"

namespace DirectGraphicalModels { namespace dnn
{
    // Constructor
    CNeuron::CNeuron(size_t numConnections) {
        m_vWeights.resize(numConnections);
    }

    void CNeuron::generateRandomWeights(void)
    {
        for (float &weight : m_vWeights)
            weight = random::U<float>(-0.5, 0.5);
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
