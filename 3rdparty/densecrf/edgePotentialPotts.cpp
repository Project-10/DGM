#include "edgePotentialPotts.h"
#include "permutohedral.h"
#include <numeric>
#include "types.h"

// Constructor
CEdgePotentialPotts::CEdgePotentialPotts(const Mat &features, float w, const std::function<void(const vec_float_t &src, vec_float_t &dst)> &SemiMetricFunction, bool per_pixel_normalization)
	: CEdgePotential()
	, m_w(w)
	, m_pLattice(std::make_unique<CPermutohedral>())
    , m_function(SemiMetricFunction)
{
	m_pLattice->init(features);

    m_norm = Mat(features.rows, 1, CV_32FC1, Scalar(1));

	// Compute the normalization factor
	m_pLattice->compute(m_norm, m_norm);
	if (per_pixel_normalization)
        for (int n = 0; n < m_norm.rows; n++)
			m_norm.at<float>(n, 0) = 1.0f / (m_norm.at<float>(n, 0) + FLT_EPSILON);
	else {
        float mean_norm = static_cast<float>(sum(m_norm)[0]);
		mean_norm = m_norm.rows / mean_norm;
        m_norm.setTo(mean_norm);
	}
}

void CEdgePotentialPotts::apply(const Mat &src, Mat &dst, Mat &temp) const
{
	// TODO: temp might be empty
	m_pLattice->compute(src, temp);

    if (m_function) { // ------------------------- With the SemiMetric function -------------------------
        // To the metric transform
        vec_float_t tmp2(src.cols);
        for (int n = 0; n < src.rows; n++) {
            float *pDst = dst.ptr<float>(n);
            float *pTemp = temp.ptr<float>(n);
            m_function(vec_float_t(pTemp, pTemp + src.cols), tmp2);
            
			for (int s = 0; s < src.cols; s++)	// states
                pDst[s] -= m_w * m_norm.at<float>(n, 0) * tmp2[s];
        }
    } else {            // ------------------------- Standard -------------------------
		for (int n = 0; n < src.rows; n++) {	// nodes
			float *pDst = dst.ptr<float>(n);
			float *pTemp = temp.ptr<float>(n);
			
            for (int s = 0; s < src.cols; s++)	// states
				pDst[s] += m_w * m_norm.at<float>(n, 0) * pTemp[s];
		}
    }
}
