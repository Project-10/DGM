#include "StatisticsAggregators.h"

#include <iostream>

#include "DataPointCollection.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
HistogramAggregator::HistogramAggregator(unsigned char nStates) : m_nStates(nStates), m_sampleCount(0) 
{
	memset(m_pBins, 0, m_nStates * sizeof(unsigned long));
}

void HistogramAggregator::Clear()
{

	memset(m_pBins, 0, m_nStates * sizeof(unsigned long));
	m_sampleCount = 0;
}

void HistogramAggregator::Aggregate(const IDataPointCollection& data, size_t index)
{
	const DataPointCollection& concreteData = (const DataPointCollection&) data;
	m_pBins[concreteData.GetIntegerLabel((int)index)]++;
	m_sampleCount++;
}

void HistogramAggregator::Aggregate(const HistogramAggregator& aggregator)
{
	assert(aggregator.m_nStates == m_nStates);

	for (unsigned char s = 0; s < m_nStates; s++) m_pBins[s] += aggregator.m_pBins[s];

	m_sampleCount += aggregator.m_sampleCount;
} 

HistogramAggregator HistogramAggregator::DeepClone() const
{
	HistogramAggregator res(m_nStates);

	for (unsigned char s = 0; s < m_nStates; s++) res.m_pBins[s] = m_pBins[s];
	res.m_sampleCount = m_sampleCount;

	return res;
}

double HistogramAggregator::Entropy(void) const
{
	double res = 0.0;

	// Assertion
	if (m_sampleCount == 0) return res;

	for (unsigned char s = 0; s < m_nStates; s++) {
		double p = (double) m_pBins[s] / (double) m_sampleCount;
		if (p > 0) res -= p * log(p) / log(2.0);
	} // b

	return res;
}

double HistogramAggregator::Entropy(unsigned char state) const
{
	double res = 0.0;

	// Assertion
	if (m_sampleCount == 0) return res;

	double p = (double) m_pBins[state] / (double) m_sampleCount;
	if (p > 0) res -= p * log(p) / log(2.0);

	return res;
}

unsigned char HistogramAggregator::FindTallestBinIndex() const
{
	unsigned char	res		 = 0;
	unsigned int	maxCount= m_pBins[res];

	for (unsigned char s = 1; s < m_nStates; s++) 
		if (m_pBins[s] > maxCount) {
			maxCount = m_pBins[s];
			res = s;
		} // if

	return res;
}

}}}