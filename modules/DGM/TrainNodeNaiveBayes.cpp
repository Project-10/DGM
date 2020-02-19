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
		for (int i = 0; i < m_nStates * getNumFeatures(); i++)
			m_vPDF.push_back(std::make_shared<CPDFHistogram>());
			//m_vPDF.push_back(std::make_shared<CPDFGaussian>());

		if (getNumFeatures() == 2)
			for (byte s = 0; s < m_nStates; s++)
				m_vPDF2D.push_back(std::make_shared<CPDFHistogram2D>());
	}

	void CTrainNodeBayes::reset(void)
	{
		CPriorNode::reset();							// resetting the prior histogram vector
		if (!m_prior.empty()) m_prior.release();		// resetting the prior

		for (auto& pdf : m_vPDF)
			pdf->reset();
		
		m_vPDF2D.clear();
	}

	void CTrainNodeBayes::addFeatureVec(const Mat &featureVector, byte gt)
	{
		// Assertions
		DGM_ASSERT_MSG(gt < m_nStates, "The groundtruth value %d is out of range %d", gt, m_nStates);
		DGM_ASSERT_MSG(featureVector.type() == CV_8UC1, "The feature vector has incorrect type");
		
		addNodeGroundTruth(gt);

		for (word f = 0; f < getNumFeatures(); f++) {
			byte feature = featureVector.at<byte>(f, 0);
			m_vPDF[f * m_nStates + gt]->addPoint(feature);
		}
		
		if (!m_vPDF2D.empty()) {
			byte x = featureVector.at<byte>(0, 0);
			byte y = featureVector.at<byte>(1, 0);
			m_vPDF2D[gt]->addPoint(Scalar(x, y));
		}
	}

	void CTrainNodeBayes::train(bool)
	{
		m_prior = getPrior(FLT_MAX);
	}

	void CTrainNodeBayes::smooth(int nIt)
	{
		if (typeid(m_vPDF[0].get()) != typeid(CPDFHistogram)) return;
		for (auto &pdf: m_vPDF)
			dynamic_cast<CPDFHistogram *>(pdf.get())->smooth(nIt);
		for(auto &pdf: m_vPDF2D)
			dynamic_cast<CPDFHistogram2D *>(pdf.get())->smooth(nIt);
	}

	void CTrainNodeBayes::saveFile(FILE *pFile) const
	{
		CPriorNode::saveFile(pFile);

		for (auto& pdf : m_vPDF)
			pdf->saveFile(pFile);
		for (auto &pdf: m_vPDF2D)
			pdf->saveFile(pFile);
	} 

	void CTrainNodeBayes::loadFile(FILE *pFile)
	{
		CPriorNode::loadFile(pFile);
		m_prior = getPrior(FLT_MAX);		// loads m_prior from the CPriorNode class

		for (auto& pdf : m_vPDF)
			pdf->loadFile(pFile);
		for (auto &pdf: m_vPDF2D)
			pdf->loadFile(pFile);
	} 

	void CTrainNodeBayes::calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const
	{
		m_prior.copyTo(potential);
		for (byte s = 0; s < m_nStates; s++) {				// state
			float	* pPot	= potential.ptr<float>(s);
			byte	* pMask	= mask.ptr<byte>(s);
			for (word f = 0; f < getNumFeatures(); f++) {		// feature
				byte feature = featureVector.ptr<byte>(f)[0];
				if (m_vPDF[f * m_nStates + s]->isEstimated()) 
					pPot[0] *= static_cast<float>(m_vPDF[f * m_nStates + s]->getDensity(feature));
				else  {
					pPot[0] = 0; 
					pMask[0] = 0;
				}
			} // f
		} // s
	}
}
