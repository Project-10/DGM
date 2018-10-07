#include "TrainNodeNaiveBayes.h"
#include "PDFHistogram.h"
#include "PDFHistogram2D.h"
#include "PDFGaussian.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
	// Constructor
	CTrainNodeBayes::CTrainNodeBayes(byte nStates, word nFeatures)
		: CBaseRandomModel(nStates)
		, CTrainNode(nStates, nFeatures)
		, CPriorNode(nStates)
		, m_prior(Mat())
	{
		m_pPDF = new IPDF**[m_nStates];
		for (byte s = 0; s < m_nStates; s++) {
			m_pPDF[s] = new IPDF*[m_nFeatures];
			for (word f = 0; f < m_nFeatures; f++)
				m_pPDF[s][f] = new CPDFHistogram();
	//			m_pPDF[s][f] = new CPDFGaussian();
		} // s

		if (m_nFeatures == 2) {
			m_pPDF2D = new IPDF*[m_nStates];
			for (byte s = 0; s < m_nStates; s++)
				m_pPDF2D[s] = new CPDFHistogram2D();
		} else m_pPDF2D = NULL;
	}

	// Destructor
	CTrainNodeBayes::~CTrainNodeBayes(void)
	{
		if (!m_prior.empty()) m_prior.release();
		for (byte s = 0; s < m_nStates; s++) {
			for (word f = 0; f < m_nFeatures; f++)
				delete m_pPDF[s][f];
			delete m_pPDF[s];
		} // s
		delete m_pPDF;
		
		if (m_pPDF2D) {
			for (byte s = 0; s < m_nStates; s++)
				delete[] m_pPDF2D[s];
			delete m_pPDF2D;
		}
	}

	void CTrainNodeBayes::reset(void)
	{
		CPriorNode::reset();							// resetting the prior histogram vector
		if (!m_prior.empty()) m_prior.release();		// resetting the prior

		for (byte s = 0; s < m_nStates; s++)
			for (word f = 0; f < m_nFeatures; f++)
				m_pPDF[s][f]->reset();
		
		if (m_pPDF2D)
			for (byte s = 0; s < m_nStates; s++)
				m_pPDF2D[s]->reset();
	}

	void CTrainNodeBayes::addFeatureVec(const Mat &featureVector, byte gt)
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
		
		if (m_pPDF2D) {
			byte x = featureVector.at<byte>(0, 0);
			byte y = featureVector.at<byte>(1, 0);
			m_pPDF2D[gt]->addPoint(Scalar(x, y));
		}
	}

	void CTrainNodeBayes::train(bool)
	{
		m_prior = getPrior(FLT_MAX);
	}

	void CTrainNodeBayes::smooth(int nIt)
	{
		if (typeid(*** m_pPDF) != typeid(CPDFHistogram)) return;
		for (byte s = 0; s < m_nStates; s++)
			for (word f = 0; f < m_nFeatures; f++)
				dynamic_cast<CPDFHistogram *>(m_pPDF[s][f])->smooth(nIt);
		if (m_pPDF2D)
			for (byte s = 0; s < m_nStates; s++)
				dynamic_cast<CPDFHistogram2D *>(m_pPDF2D[s])->smooth(nIt);
	}

	void CTrainNodeBayes::saveFile(FILE *pFile) const
	{
		CPriorNode::saveFile(pFile);

		for (byte s = 0; s < m_nStates; s++)
			for (word f = 0; f < m_nFeatures; f++)
				m_pPDF[s][f]->saveFile(pFile);
		if (m_pPDF2D)
			for (byte s = 0; s < m_nStates; s++)
				m_pPDF2D[s]->saveFile(pFile);
	} 

	void CTrainNodeBayes::loadFile(FILE *pFile)
	{
		CPriorNode::loadFile(pFile);
		m_prior = getPrior(FLT_MAX);		// loads m_prior from the CPriorNode class

		for (byte s = 0; s < m_nStates; s++)
			for (word f = 0; f < m_nFeatures; f++)
				m_pPDF[s][f]->loadFile(pFile);
		if (m_pPDF2D)
			for (byte s = 0; s < m_nStates; s++)
				m_pPDF2D[s]->loadFile(pFile);
	} 

	void CTrainNodeBayes::calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const
	{
		m_prior.copyTo(potential);
		for (byte s = 0; s < m_nStates; s++) {				// state
			float	* pPot	= potential.ptr<float>(s);
			byte	* pMask	= mask.ptr<byte>(s);
			for (word f = 0; f < m_nFeatures; f++) {		// feature
				byte feature = featureVector.ptr<byte>(f)[0];
				if (m_pPDF[s][f]->isEstimated()) 
					pPot[0] *= static_cast<float>(m_pPDF[s][f]->getDensity(feature));	
				else  {
					pPot[0] = 0; 
					pMask[0] = 0;
				}
			} // f
		} // s
	}
}
