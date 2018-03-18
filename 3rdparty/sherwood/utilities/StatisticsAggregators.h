#pragma once

#include <math.h>

#include <limits>
#include <vector>

#include "../Sherwood.h"

#include "DataPointCollection.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
	/// @todo Document this class
	class HistogramAggregator 
	{
	public:
		HistogramAggregator(void) : m_nStates(0), m_sampleCount(0) {}
		HistogramAggregator(unsigned char nStates);
		~HistogramAggregator(void) {};

		void				Clear(void);
		void				Aggregate(const IDataPointCollection& data, size_t index);
		void				Aggregate(const HistogramAggregator& aggregator);
		HistogramAggregator	DeepClone(void) const;

		unsigned long		SampleCount(void) const {return m_sampleCount;}
		float				GetProbability(int classIndex) const {return (float)(m_pBins[classIndex]) / m_sampleCount;}
		double				Entropy(void) const;
		double				Entropy(unsigned char state) const;
		unsigned char		FindTallestBinIndex(void) const;


	private:
		unsigned long	m_pBins[256];
		unsigned char	m_nStates;
		unsigned long 	m_sampleCount;
	};

} } }
