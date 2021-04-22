#include "EdgeModelPotts.h"
#include "permutohedral/permutohedral.h"

namespace DirectGraphicalModels {
	// Constructor
	CEdgeModelPotts::CEdgeModelPotts(const Mat& features, float weight, const std::function<void(const Mat& src, Mat& dst)>& semiMetricFunction, bool perPixelNormalization)
		: IEdgeModel()
		, m_pLattice(new CPermutohedral())
		, m_weight(weight)
		, m_norm(features.rows, 1, CV_32FC1, Scalar(1))
		, m_function(semiMetricFunction)
	{
		m_pLattice->init(features);

		// Compute the normalization factor
		m_pLattice->compute(m_norm, m_norm);
		
		if (perPixelNormalization)
			for (int n = 0; n < m_norm.rows; n++)
				m_norm.at<float>(n, 0) = 1.0f / (m_norm.at<float>(n, 0) + FLT_EPSILON);
		else {
			float mean_norm = static_cast<float>(sum(m_norm)[0]);
			mean_norm = m_norm.rows / mean_norm;
			m_norm.setTo(mean_norm);
		}
	}

	// Destructor
	CEdgeModelPotts::~CEdgeModelPotts(void)
	{
 		delete m_pLattice;
	}

	// dst = e^(w * norm * f(Lattice.compute(src)))
	void CEdgeModelPotts::apply(const Mat &src, Mat &dst) const
	{
		m_pLattice->compute(src, dst);				// dst = Lattice.compute(src)

#ifdef ENABLE_PDP
		parallel_for_(Range(0, dst.rows), [&](const Range& range) {
#else
		const Range range(0, dst.rows); 
#endif
		for (int n = range.start; n < range.end; n++) {	// nodes
			if (m_function) m_function(dst.row(n), lvalue_cast(dst.row(n)));		// With the SemiMetric function

			// dst.row(n) *= m_weight * m_norm.at<float>(n, 0);
			// Using expressive notation for sake of efficiency
			float* pDst = dst.ptr<float>(n);
			float	k = m_weight * m_norm.at<float>(n, 0);
			for (int s = 0; s < dst.cols; s++) pDst[s] *= k;
		}
#ifdef ENABLE_PDP
		});
#endif
		exp(dst, dst);
	}

}
