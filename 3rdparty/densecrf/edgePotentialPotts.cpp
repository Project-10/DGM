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

	m_vNorm.resize(nNodes);
	std::fill(m_vNorm.begin(), m_vNorm.end(), 1);

	// Compute the normalization factor
	m_pLattice->compute(m_vNorm, m_vNorm, 1);
	if (per_pixel_normalization)
		for (float &norm : m_vNorm)
			norm = 1.f / (norm + FLT_EPSILON);
	else {
		float mean_norm = std::accumulate(m_vNorm.begin(), m_vNorm.end(), 0.0f);
		mean_norm = m_vNorm.size() / mean_norm;
		std::fill(m_vNorm.begin(), m_vNorm.end(), mean_norm);
	}
}

void CEdgePotentialPotts::apply(vec_float_t &out_values, const vec_float_t &in_values, vec_float_t &tmp, int value_size) const
{
	m_pLattice->compute(tmp, in_values, value_size);

    if (m_pFunction) { // ------------------------- With the SemiMetric function -------------------------
        // To the metric transform
        float * tmp2 = new float[value_size];
        for (size_t i = 0; i < m_nNodes; i++) {
            float * out = out_values.data() + i * value_size;
            float * t1 = tmp.data() + i * value_size;
            m_pFunction->apply(tmp2, t1, value_size);
            
            for (int j = 0; j < value_size; j++)
                out[j] -= m_w * m_vNorm[i] * tmp2[j];
        }
        delete[] tmp2;
    } else {            // ------------------------- Standard -------------------------
        size_t k = 0;
        for (const float &norm : m_vNorm)
            for (int j = 0; j < value_size; j++) {
                out_values[k] += m_w * norm * tmp[k];
                k++;
            } // j
    }
}
