#include "PriorEdge.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
void CPriorEdge::addEdgeGroundTruth(byte gt1, byte gt2)
{
	DGM_ASSERT(gt1 < m_nStates);
	DGM_ASSERT(gt2 < m_nStates);
	m_histogramPrior.at<int>(gt2, gt1)++;	
}

Mat CPriorEdge::calculatePrior(void) const
{
	byte x;
	Mat H;
	Mat res(m_nStates, m_nStates, CV_32FC1);

	m_histogramPrior.convertTo(H, CV_32FC1);

	switch (m_normApp) {
		case eP_APP_NORM_STANDARD: {	// standard approach
			double Sum = sum(m_histogramPrior)[0];
			m_histogramPrior.convertTo(res, res.type(), 1.0 / Sum);
			break;
		}
		case eP_APP_NORM_SYMMETRIC: {	// symetric approach
			Mat tmp;
			Mat Hd(H.size(), H.type()); Hd.setTo(0);	// Hd - diagonal normalization matrix
			for (x = 0; x < m_nStates; x++)
				if (H.at<float>(x, x) >= 1.0f)	Hd.at<float>(x, x) = 1.0f / sqrtf(H.at<float>(x, x));
				
			gemm(H, Hd, 1.0, Mat(), 0.0, tmp);			// tmp = Prior * Hd;
			gemm(Hd, tmp, 1.0, Mat(), 0.0, res);		// res = Hd * (Prior * Hd);
			tmp.release();
			Hd.release();
			break;
		}
		case eP_APP_NORM_ASYMMETRIC: {	// assymetric approach
			for (byte y = 0; y < m_nStates; y++) {
				const float	*pH		= H.ptr<float>(y);
				float		*pRes	= res.ptr<float>(y);
				float		 max	= 1.0f;
				for (x = 0; x < m_nStates; x++) if (max < pH[x]) max = pH[x];
				for (x = 0; x < m_nStates; x++) pRes[x] = pH[x] / max;			
			} // y
			break;
		}
	}
	return res;
}

}