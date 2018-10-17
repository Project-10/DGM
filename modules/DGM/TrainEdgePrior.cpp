#include "TrainEdgePrior.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
// Constructor
CTrainEdgePrior::CTrainEdgePrior(byte nStates, word nFeatures, ePotPenalApproach penApproach, ePotNormApproach normApproach) 
	: CBaseRandomModel(nStates)
    , CTrainEdgePottsCS(nStates, nFeatures, penApproach)
    , CPriorEdge(nStates, normApproach)
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

void CTrainEdgePrior::train(bool)
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

Mat	CTrainEdgePrior::calculateEdgePotentials(const Mat &featureVector1, const Mat &featureVector2, const vec_float_t &vParams) const
{
	Mat res = CTrainEdgePottsCS::calculateEdgePotentials(featureVector1, featureVector2, vParams);
	multiply(res, m_prior, res);
	return res;
}

inline void CTrainEdgePrior::loadPriorMatrix(void)
{
	if (!m_prior.empty()) m_prior.release();
	m_prior = getPrior();	
}

}
