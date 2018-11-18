#include "edgePotentialPotts.h"
#include <numeric>
#include "types.h"

// Constructor
CEdgePotentialPotts::CEdgePotentialPotts(const Mat &features, float weight, const std::function<void(const Mat &src, Mat &dst)> &SemiMetricFunction, bool per_pixel_normalization)
	: CEdgePotential()
	, m_pLattice(std::make_unique<CPermutohedral>())
    , m_weight(weight)
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

void CEdgePotentialPotts::apply(const Mat &pots, Mat &dst) const
{
	m_pLattice->compute(pots, dst);

#ifdef ENABLE_PPL
	concurrency::parallel_for(0, dst.rows, [&](int n) {
#else
	for (int n = 0; n < dst.rows; n++) {	// nodes
#endif
		if (m_function) m_function(dst.row(n), lvalue_cast(dst.row(n)));		// With the SemiMetric function

		float*	pDst = dst.ptr<float>(n);
		float	k = m_weight * m_norm.at<float>(n, 0);
		for (int s = 0; s < dst.cols; s++) pDst[s] *= k;
	}
#ifdef ENABLE_PPL
	);
#endif
	exp(dst, dst);
}
