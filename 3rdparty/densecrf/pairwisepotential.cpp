#include "pairwisepotential.h"
#include "permutohedral.h"

// Constructor
CPottsPotential::CPottsPotential(const float *pFeatures, word nFeatures, int nNodes, float w, bool per_pixel_normalization)
    : CPairwisePotential()
    , m_nNodes(nNodes)
    , m_w(w)
    , m_pLattice(std::make_unique<CPermutohedral>())
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


void CPottsPotential::apply(vec_float_t &out_values, const vec_float_t &in_values, vec_float_t &tmp, int value_size) const
{
    m_pLattice->compute(tmp, in_values, value_size);
    
    size_t k = 0;
    for (const float &norm : m_vNorm)
        for (int j = 0; j < value_size; j++) {
            out_values[k] += m_w * norm * tmp[k];
            k++;
        }
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Constructor
CBPPottsPotential::CBPPottsPotential(const float* features1, const float* features2, int D, int N1, int N2, float w, bool per_pixel_normalization)
    : CPairwisePotential()
    , m_w(w)
    , m_pLattice(std::make_unique<CPermutohedral>())
    , m_N1(N1)
    , m_N2(N2)
{
    float * features = new float[(m_N1 + m_N2) * D];
    memset(features, 0, (m_N1 + m_N2) * D * sizeof(float));
    memcpy(features, features1, m_N1 * D * sizeof(float));
    memcpy(features + m_N1 * D, features2, m_N2 * D * sizeof(float));
    m_pLattice->init(features, D, m_N1 + m_N2);
    delete[] features;
    
    m_vNorm.resize(m_N2);
    std::fill(m_vNorm.begin(), m_vNorm.end(), 0);
    
    vec_float_t tmp(m_N1);
    std::fill(tmp.begin(), tmp.end(), 1);
    
    // Compute the normalization factor
    m_pLattice->compute(m_vNorm, tmp, 1, 0, m_N1, m_N1, m_N2);
    if (per_pixel_normalization) {
        for (float &norm : m_vNorm)
            norm = 1.0f / (norm + FLT_EPSILON);
    }
    else {
        float mean_norm = std::accumulate(m_vNorm.begin(), m_vNorm.end(), 0.0f);
        mean_norm = m_vNorm.size() / mean_norm;
        // use a per pixel normalization
        for (float &norm : m_vNorm)
            norm = mean_norm;
    }
}

void CBPPottsPotential::apply(vec_float_t &out_values, const vec_float_t &in_values, vec_float_t &tmp, int value_size) const
{
    m_pLattice->compute(tmp, in_values, value_size, 0, m_N1, m_N1, m_N2);
    
    int k = 0;
    for (int i = 0; i < m_N2; i++)
        for (int j = 0; j < value_size; j++) {
            out_values[k] += m_w * m_vNorm[i] * tmp[k];
            k++;
        }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CBPSemiMetricPotential::apply(vec_float_t &out_values, const vec_float_t &in_values, vec_float_t &tmp, int value_size) const
{
    m_pLattice->compute(tmp, in_values, value_size, 0, m_N1, m_N1, m_N2);
    
    // To the metric transform
    float * tmp2 = new float[value_size];
    for (int i = 0; i < m_N2; i++) {
        float * out = out_values.data() + i * value_size;
        float * t1 = tmp.data() + i * value_size;
        m_pFunction->apply(tmp2, t1, value_size);
        for (int j = 0; j<value_size; j++)
            out[j] -= m_w * m_vNorm[i] * tmp2[j];
    }
    delete[] tmp2;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSemiMetricPotential::apply(vec_float_t &out_values, const vec_float_t &in_values, vec_float_t  &tmp, int value_size) const
{
    m_pLattice->compute(tmp, in_values, value_size);
    
    // To the metric transform
    float * tmp2 = new float[value_size];
    for (size_t i = 0; i < m_nNodes; i++) {
        float * out = out_values.data() + i * value_size;
        float * t1 = tmp.data() + i * value_size;
        m_pFunction->apply(tmp2, t1, value_size);
        for (int j = 0; j<value_size; j++)
            out[j] -= m_w* m_vNorm[i] * tmp2[j];
    }
    delete[] tmp2;
}


