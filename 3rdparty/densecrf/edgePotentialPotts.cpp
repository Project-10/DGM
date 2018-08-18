#include "edgePotentialPotts.h"
#include "permutohedral.h"
#include <numeric>

// Constructor
CEdgePotentialPotts::CEdgePotentialPotts(const float *pFeatures, word nFeatures, size_t nNodes, float w, const SemiMetricFunction *pFunction, bool per_pixel_normalization)
	: CEdgePotential()
	, m_nNodes(nNodes)
	, m_w(w)
	, m_pLattice(std::make_unique<CPermutohedral>())
    , m_pFunction(pFunction)
{
	m_pLattice->init(pFeatures, nFeatures, nNodes);

    m_norm = Mat(nNodes, 1, CV_32FC1, Scalar(1));

	// Compute the normalization factor
	m_pLattice->compute(m_norm, m_norm, 0, 0, 0, 0);
	if (per_pixel_normalization)
        for (int n = 0; n < m_norm.rows; n++)
			m_norm.at<float>(n, 0) = 1.f / (m_norm.at<float>(n, 0) + FLT_EPSILON);
	else {
        float mean_norm = static_cast<float>(sum(m_norm)[0]);
		mean_norm = m_norm.rows / mean_norm;
        m_norm.setTo(mean_norm);
	}
}

void CEdgePotentialPotts::apply(Mat &out, const Mat &in, Mat &temp) const
{
	m_pLattice->compute(temp, in, 0, 0, 0, 0);

    if (m_pFunction) { // ------------------------- With the SemiMetric function -------------------------
        // To the metric transform
        float * tmp2 = new float[out.cols];
        for (size_t n = 0; n < m_nNodes; n++) {
            float *pOut = out.ptr<float>(n);
            float *pTemp = temp.ptr<float>(n);
            m_pFunction->apply(tmp2, pTemp);
            
			for (int s = 0; s < out.cols; s++)	// states
                pOut[s] -= m_w * m_norm.at<float>(n, 0) * tmp2[s];
        }
        delete[] tmp2;
    } else {            // ------------------------- Standard -------------------------
		for (int n = 0; n < out.rows; n++) {	// nodes
			float *pOut = out.ptr<float>(n);
			float *pTemp = temp.ptr<float>(n);
			for (int s = 0; s < out.cols; s++)	// states
				pOut[s] += m_w * m_norm.at<float>(n, 0) * pTemp[s];
		}
    }
}
