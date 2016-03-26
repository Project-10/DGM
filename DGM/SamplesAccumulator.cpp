#include "SamplesAccumulator.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
// Constructor
CSamplesAccumulator::CSamplesAccumulator(void) : m_nSamples(0)
{ }

// Destructor
CSamplesAccumulator::~CSamplesAccumulator(void)
{
	for (auto c = m_vClusters.begin(); c != m_vClusters.end(); c++)	c->release();
	m_vClusters.clear();
}

void CSamplesAccumulator::addSample(const Mat &sample)
{
	if (m_nSamples == 0) m_sampleLength = sample.rows;
	
	// Assertions
	DGM_ASSERT_MSG(sample.rows == m_sampleLength, "The input sample size mismatch");
	DGM_ASSERT_MSG(sample.type() == CV_64FC1, "The input sample type missmatch");

	// nSamples = c * CLUSTER_SIZE + i;
	size_t nSamples	= m_nSamples++;						// number of overall samples
	size_t c		= nSamples / CLUSTER_SIZE;			// destination cluster index
	size_t i		= nSamples - c * CLUSTER_SIZE;		// sample index in the destination cluster 

	// If not enough clusters - create a new one
	if (m_vClusters.size() <= c) {
		Mat cluster(CLUSTER_SIZE, m_sampleLength, CV_64FC1);
		m_vClusters.push_back(cluster);
	}

	// Adding the sample to the cluster
	double * pSample = m_vClusters.at(c).ptr<double>(static_cast<int>(i));
	for (int f = 0; f < m_sampleLength; f++) pSample[f] = sample.at<double>(f, 0);
}
	
Mat CSamplesAccumulator::getSample(size_t idx) const
{
	// Assertions
	DGM_ASSERT_MSG(m_nSamples > 0, "No samples are in accumulator");
	
	Mat res(m_sampleLength, 1, CV_64FC1);
	
	// idx = c * CLUSTER_SIZE + i;
	size_t c	= idx / CLUSTER_SIZE;		// cluster index
	size_t i	= idx - c * CLUSTER_SIZE;	// sample index in cluster iCluster	
	
	const double * pSample = m_vClusters.at(c).ptr<double>(static_cast<int>(i));
	for (int f = 0; f < m_sampleLength; f++) res.at<double>(f, 0) = pSample[f];

	return res;
}

void CSamplesAccumulator::reset(void)
{
	for (auto c = m_vClusters.begin(); c != m_vClusters.end(); c++)	c->release();
	m_vClusters.clear();
	m_nSamples = 0;
}

}