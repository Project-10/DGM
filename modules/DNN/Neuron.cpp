#include "Neuron.h"
#include "DGM/random.h"
#include "macroses.h"

namespace DirectGraphicalModels { namespace dnn
{
	// Constructor
	CNeuron::CNeuron(size_t size, double value) : m_value(value) {
		m_vWeights.resize(size);
	}

	void CNeuron::generateRandomWeights(void)
	{
		for (double &weight : m_vWeights)
			weight = random::U<double>(-0.5, 0.5);
	}

	void CNeuron::setWeight(size_t index, double weight)
	{
		DGM_ASSERT(index < m_vWeights.size());
		m_vWeights[index] = weight;
	}

	double CNeuron::getWeight(size_t index) const
	{
		DGM_ASSERT(index < m_vWeights.size());
		return m_vWeights[index];
	}
	
	
} }
