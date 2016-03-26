#include "TrainEdgePrior.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
// Constructor
CTrainEdgePrior::CTrainEdgePrior(byte nStates, byte nFeatures, ePotPenalApproach penApproach, ePotNormApproach normApproach) 
	: CTrainEdgePottsCS(nStates, nFeatures, penApproach)
	, CPriorEdge(nStates, normApproach)
	, CBaseRandomModel(nStates)
	, m_prior(Mat())
{}

// Destructor
CTrainEdgePrior::~CTrainEdgePrior(void) 
{
	if (!m_prior.empty()) m_prior.release();
}

// Resets <Prior>  matrix
void CTrainEdgePrior::reset(void) 
{
	CPriorEdge::reset();							// resetting the prior histogram matrix
	if (!m_prior.empty()) m_prior.release();		// resetting the prior
}	

void CTrainEdgePrior::addFeatureVecs(const Mat &, byte gt1, const Mat &, byte gt2)
{
	CPriorEdge::addEdgeGroundTruth(gt1, gt2);
}

void CTrainEdgePrior::train(void)
{
	loadPriorMatrix();
}

void CTrainEdgePrior::saveFile(FILE *pFile) const 
{
	CPriorEdge::saveFile(pFile);
}

void CTrainEdgePrior::loadFile(FILE *pFile)
{
	CPriorEdge::loadFile(pFile);
	loadPriorMatrix();				
}

Mat	CTrainEdgePrior::calculateEdgePotentials(const Mat &featureVector1, const Mat &featureVector2, float *params, size_t params_len) const
{
	Mat res = CTrainEdgePottsCS::calculateEdgePotentials(featureVector1, featureVector2, params, params_len);
	multiply(res, m_prior, res);
	return res;
}

inline void CTrainEdgePrior::loadPriorMatrix(void)
{
	if (!m_prior.empty()) m_prior.release();
	m_prior = getPrior();	
}

}