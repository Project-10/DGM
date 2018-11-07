#include "edgePotentialPotts.h"
#include "permutohedral.h"
#include <numeric>
#include "types.h"

// Constructor
CEdgePotentialPotts::CEdgePotentialPotts(const Mat &features, float weight, const std::function<void(const Mat &src, Mat &dst)> &SemiMetricFunction, bool per_pixel_normalization)
	: CEdgePotential()
	, m_weight(weight)
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
	m_pLattice->compute(src, temp);

	for (int n = 0; n < src.rows; n++) {	// nodes
		if (m_function) m_function(temp.row(n), lvalue_cast(temp.row(n)));		// With the SemiMetric function

		float *pDst = dst.ptr<float>(n);
		float *pTemp = temp.ptr<float>(n);
			
        for (int s = 0; s < src.cols; s++)	// states
			pDst[s] += m_weight * m_norm.at<float>(n, 0) * pTemp[s];
	}
}
