#include "Prior.h"

namespace DirectGraphicalModels
{
// Constructor
CPrior::CPrior(byte nStates, RandomModelType type) : CBaseRandomModel(nStates), m_type(type)
{
	const int size[] = { m_nStates, m_nStates, m_nStates};
	m_histogramPrior = Mat(type, size, CV_32SC1);
	m_histogramPrior.setTo(0);
}

// Destructor
CPrior::~CPrior(void) 
{
	m_histogramPrior.release();
}

void CPrior::reset(void)
{
	m_histogramPrior.setTo(0);
}

Mat CPrior::getPrior(float weight) const
{
	if (sum(m_histogramPrior)[0] < 1)									// if addXXXGroundTruth() was not called
		return Mat(m_nStates, m_nStates, CV_32FC1, Scalar(1.0f));		// return uniform distribution	
	
	Mat res = calculatePrior();
	if (weight != 1.0f)  res.convertTo(res, res.type(), weight);
	return res;
}

void CPrior::saveFile(FILE *pFile) const 
{
	switch(m_type) {
		case RM_UNARY:		//---------------------------------------------------------------------------------
			for (register byte y = 0; y < m_nStates; y++)
				fwrite(&m_histogramPrior.at<int>(y, 0), sizeof(int), 1, pFile);
			break;
		case RM_PAIRWISE:	//---------------------------------------------------------------------------------
			for (register byte y = 0; y < m_nStates; y++)
				for (register byte x = 0; x < m_nStates; x++)
					fwrite(&m_histogramPrior.at<int>(y, x), sizeof(int), 1, pFile);
			break;
		case RM_TRIPLET:	//---------------------------------------------------------------------------------
			for (register byte z = 0; z < m_nStates; z++)	
				for (register byte y = 0; y < m_nStates; y++)
					for (register byte x = 0; x < m_nStates; x++)
						fwrite(&m_histogramPrior.at<int>(z, y, x), sizeof(int), 1, pFile);
			break;
	}
} 

void CPrior::loadFile(FILE *pFile) 
{
	switch(m_type) {
		case RM_UNARY:		//---------------------------------------------------------------------------------
			for (register byte y = 0; y < m_nStates; y++)
				fread(&m_histogramPrior.at<int>(y, 0), sizeof(int), 1, pFile);
			break;
		case RM_PAIRWISE:	//---------------------------------------------------------------------------------
			for (register byte y = 0; y < m_nStates; y++)
				for (register byte x = 0; x < m_nStates; x++)
					fread(&m_histogramPrior.at<int>(y, x), sizeof(int), 1, pFile);
			break;
		case RM_TRIPLET:	//---------------------------------------------------------------------------------
			for (register byte z = 0; z < m_nStates; z++)	
				for (register byte y = 0; y < m_nStates; y++)
					for (register byte x = 0; x < m_nStates; x++)
						fread(&m_histogramPrior.at<int>(z, y, x), sizeof(int), 1, pFile);
			break;
	}
}


}