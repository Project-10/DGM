#include "TrainingContexts.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
double	ClassificationTrainingContext::ComputeInformationGain(const HistogramAggregator& allStatistics, const HistogramAggregator& leftStatistics, const HistogramAggregator& rightStatistics) {
	double			entropyBefore	= allStatistics.Entropy();
	unsigned int	nTotalSamples	= leftStatistics.SampleCount() + rightStatistics.SampleCount();
	if (nTotalSamples <= 1) return 0.0;
	double			entropyAfter	= (leftStatistics.SampleCount() * leftStatistics.Entropy() + rightStatistics.SampleCount() * rightStatistics.Entropy()) / nTotalSamples;
	return entropyBefore - entropyAfter;
}
}}}