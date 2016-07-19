#include "TrainNodeNaiveBayes.h"
#include "PDFHistogram.h"
#include "PDFGaussian.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
// Constructor
CTrainNodeNaiveBayes::CTrainNodeNaiveBayes(byte nStates, word nFeatures) 
	: CTrainNode(nStates, nFeatures)
	, CPriorNode(nStates)
	, CBaseRandomModel(nStates)
	, m_prior(Mat())
{
	m_pPDF = new CPDF**[m_nStates];
	for (byte s = 0; s < m_nStates; s++) {
		m_pPDF[s] = new CPDF*[m_nFeatures];
		for (word f = 0; f < m_nFeatures; f++)
			m_pPDF[s][f] = new CPDFHistogram();
//			m_pPDF[s][f] = new CPDFGaussian();
	} // s


#ifdef DEBUG_MODE	// --- Debug ---
	word len = static_cast<word>(nStates);
	m_H2d = new histogram2D_byte[len];
	for (word i = 0; i < len; i++) {
		memset(m_H2d[i].data, 0, 256*256*sizeof(long));
		m_H2d[i].n = 0;
	}
#endif				// --- ----- ---
}

// Destructor
CTrainNodeNaiveBayes::~CTrainNodeNaiveBayes(void) 
{
	if (!m_prior.empty()) m_prior.release();
	for (byte s = 0; s < m_nStates; s++) {
		for (word f = 0; f < m_nFeatures; f++)
			delete m_pPDF[s][f];
		delete m_pPDF[s];
	} // s
	delete m_pPDF;
	
#ifdef DEBUG_MODE	// --- Debug ---
	delete [] m_H2d;
#endif				// --- ----- ---
}

void CTrainNodeNaiveBayes::reset(void) 
{
	CPriorNode::reset();							// resetting the prior histogram vector
	if (!m_prior.empty()) m_prior.release();		// resetting the prior

	for (byte s = 0; s < m_nStates; s++)
		for (word f = 0; f < m_nFeatures; f++)
			m_pPDF[s][f]->reset();
	
#ifdef DEBUG_MODE	// --- Debug ---
	word len = static_cast<word>(m_nStates);
	for (word i = 0; i < len; i++) {
		memset(m_H2d[i].data, 0, 256*256*sizeof(long));
		m_H2d[i].n = 0;
	}
#endif				// --- ----- ---
}

void CTrainNodeNaiveBayes::addFeatureVec(const Mat &featureVector, byte gt) 
{
	// Assertions
	DGM_ASSERT_MSG(gt < m_nStates, "The groundtruth value %d is out of range %d", gt, m_nStates);
	DGM_ASSERT_MSG(featureVector.type() == CV_8UC1, "The feature vector has incorrect type");
	
	addNodeGroundTruth(gt);

	for (word f = 0; f < m_nFeatures; f++) {
//		byte feature = featureVector.ptr<byte>(f)[0];
		byte feature = featureVector.at<byte>(f, 0);
		m_pPDF[gt][f]->addPoint(feature);
	}
	
#ifdef DEBUG_MODE	// --- Debug ---
	byte x = featureVector.at<byte>(0, 0);
	byte y = featureVector.at<byte>(1, 0);

	m_H2d[gt].data[x][y]++;
	m_H2d[gt].n++;
#endif				// --- ----- ---
}

void CTrainNodeNaiveBayes::train(void)
{
	calculatePrior(); 
}

void CTrainNodeNaiveBayes::smooth(int nIt)
{
	if (typeid(*** m_pPDF) != typeid(CPDFHistogram)) return;
	for (byte s = 0; s < m_nStates; s++)
		for (word f = 0; f < m_nFeatures; f++)
			dynamic_cast<CPDFHistogram *>(m_pPDF[s][f])->smooth(nIt);
}

void CTrainNodeNaiveBayes::saveFile(FILE *pFile) const 
{
	CPriorNode::saveFile(pFile);

	for (byte s = 0; s < m_nStates; s++)
		for (word f = 0; f < m_nFeatures; f++)
			m_pPDF[s][f]->saveFile(pFile);

} 

void CTrainNodeNaiveBayes::loadFile(FILE *pFile) 
{
	CPriorNode::loadFile(pFile);
	calculatePrior();		// loads m_prior from the CPriorNode class

	for (byte s = 0; s < m_nStates; s++)
		for (word f = 0; f < m_nFeatures; f++)
			m_pPDF[s][f]->loadFile(pFile);
} 

void CTrainNodeNaiveBayes::calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const
{
	m_prior.copyTo(potential);
	for (byte s = 0; s < m_nStates; s++) {			// state
		float	* pPot	= potential.ptr<float>(s);
		byte	* pMask	= mask.ptr<byte>(s);
		for (word f = 0; f < m_nFeatures; f++) {		// feature
			byte feature = featureVector.ptr<byte>(f)[0];
			if (m_pPDF[s][f]->isEstimated()) 
				pPot[0] *= m_pPDF[s][f]->getDensity(feature);	
			else  {
				pPot[0] = 0; 
				pMask[0] = 0;
			}
		} // f
	} // s
}

inline void CTrainNodeNaiveBayes::calculatePrior(void)
{
	if (!m_prior.empty()) m_prior.release();
	m_prior = getPrior(FLT_MAX);
}

}