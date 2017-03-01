#include "SamplesAccumulator.h"
#include "random.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
	void CSamplesAccumulator::reset(void)
	{
		for (Mat &acc : m_vSamplesAcc) acc.release();
		std::fill(m_vNumInputSamples.begin(), m_vNumInputSamples.end(), 0);
	}

	void CSamplesAccumulator::addSample(const Mat &featureVector, byte state)
	{
		// Assertions:
		DGM_ASSERT_MSG(state < m_vSamplesAcc.size(), "The groundtruth value %d is out of range %zu", state, m_vSamplesAcc.size());

		if (m_vSamplesAcc[state].rows < m_maxSamples) {
			m_vSamplesAcc[state].push_back(featureVector.t());
		}
		else {
			int k = random::u(0, m_vNumInputSamples[state]);
			if (k < m_maxSamples)
				m_vSamplesAcc[state].row(k) = featureVector.t();
		}
		m_vNumInputSamples[state]++;
	}

	int	CSamplesAccumulator::getNumSamples(byte state) const
	{
		DGM_ASSERT_MSG(state < m_vSamplesAcc.size(), "The groundtruth value %d is out of range %zu", state, m_vSamplesAcc.size());
		return m_vSamplesAcc[state].rows;
	}

	int CSamplesAccumulator::getNumInputSamples(byte state) const
	{
		DGM_ASSERT_MSG(state < m_vNumInputSamples.size(), "The groundtruth value %d is out of range %zu", state, m_vNumInputSamples.size());
		return m_vNumInputSamples[state];
	}

	void CSamplesAccumulator::release(byte state)
	{
		m_vNumInputSamples[state] = 0;
		m_vSamplesAcc[state].release();
	}

}