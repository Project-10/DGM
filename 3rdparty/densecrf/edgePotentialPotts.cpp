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
	m_pLattice->compute(m_norm, m_norm, 1);
	if (per_pixel_normalization)
        for (int n = 0; n < m_norm.rows; n++)
			m_norm.at<float>(n, 0) = 1.f / (m_norm.at<float>(n, 0) + FLT_EPSILON);
	else {
        float mean_norm = static_cast<float>(sum(m_norm)[0]);
		mean_norm = m_norm.rows / mean_norm;
        m_norm.setTo(mean_norm);
	}
}

// TODO: perhaps value_size is not needed anymore
void CEdgePotentialPotts::apply(Mat &out, const Mat &in, Mat &temp, int value_size) const
{
	m_pLattice->compute(temp, in, value_size);

    if (m_pFunction) { // ------------------------- With the SemiMetric function -------------------------
        // To the metric transform
        float * tmp2 = new float[value_size];
        for (size_t n = 0; n < m_nNodes; n++) {
            float *pOut = out.ptr<float>(n);
            float *pTemp = temp.ptr<float>(n);
            m_pFunction->apply(tmp2, pTemp, value_size);
            
            for (int j = 0; j < value_size; j++)
                pOut[j] -= m_w * m_norm.at<float>(n, 0) * tmp2[j];
        }
        delete[] tmp2;
    } else {            // ------------------------- Standard -------------------------
        // TODO:optimize
        size_t k = 0;
        for (int n = 0; n < m_norm.rows; n++)
            for (int j = 0; j < value_size; j++) {
                reinterpret_cast<float *>(out.data)[k] += m_w * m_norm.at<float>(n, 0) * reinterpret_cast<float *>(temp.data)[k];
                k++;
            } // j
    }
}
