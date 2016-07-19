#pragma once

#include "..\Sherwood.h"

#include "FeatureResponseFunctions.h"
#include "StatisticsAggregators.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood {

// =========================== ClassificationTrainingContext Class ===========================
	/// @todo Document this class
	class ClassificationTrainingContext : public ITrainingContext<LinearFeatureResponse, HistogramAggregator> 
	{
	public:
		ClassificationTrainingContext(unsigned char nStates, unsigned short nFeatures) : m_nStates(nStates), m_nFeatures(nFeatures) {}
		~ClassificationTrainingContext(void) {}


	private:
		LinearFeatureResponse	GetRandomFeature(Random& random)	{ return LinearFeatureResponse::CreateRandom(m_nFeatures, random); }
		HistogramAggregator		GetStatisticsAggregator(void)		{ return HistogramAggregator(m_nStates); }
		double					ComputeInformationGain(const HistogramAggregator& allStatistics, const HistogramAggregator& leftStatistics, const HistogramAggregator& rightStatistics);
		bool					ShouldTerminate(const HistogramAggregator& parent, const HistogramAggregator& leftChild, const HistogramAggregator& rightChild, double gain) {return gain < 0.01;}


	private:
		unsigned char	m_nStates;
		unsigned short	m_nFeatures;
	};
}}}