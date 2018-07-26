#pragma once

#include "permutohedral.h"
#include <numeric>


class PairwisePotential {
public:
    virtual void apply(vec_float_t &out_values, const vec_float_t &in_values, vec_float_t &tmp, int value_size) const = 0;
};

class SemiMetricFunction {
public:
    // For two probabilities apply the semi metric transform: v_i = sum_j mu_ij u_j
    virtual void apply(float *out_values, const float *in_values, int value_size ) const = 0;
};

class BPPottsPotential : public PairwisePotential {

public:
    BPPottsPotential(const float* features1, const float* features2, int D, int N1, int N2, float w, bool per_pixel_normalization = true) : N1_(N1), N2_(N2), w_(w) {
        float * features = new float[(N1_ + N2_)*D];
        memset(features, 0, (N1_ + N2_)*D * sizeof(float));
        memcpy(features, features1, N1_ * D * sizeof(float));
        memcpy(features + N1_ * D, features2, N2_ * D * sizeof(float));
        lattice_.init(features, D, N1_ + N2_);
        delete[] features;
        
		m_vNorm.resize(N2_);
		std::fill(m_vNorm.begin(), m_vNorm.end(), 0);
       
		vec_float_t tmp(N1_);
		std::fill(tmp.begin(), tmp.end(), 1);
        
		// Compute the normalization factor
        lattice_.compute(m_vNorm, tmp, 1, 0, N1_, N1_, N2_);
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
    
	virtual ~BPPottsPotential(void) {}

    virtual void apply(vec_float_t &out_values, const vec_float_t &in_values, vec_float_t &tmp, int value_size) const 
	{
        lattice_.compute(tmp, in_values, value_size, 0, N1_, N1_, N2_);
		
		int k = 0;
		for (int i = 0; i < N2_; i++)
			for (int j = 0; j < value_size; j++) {
				out_values[k] += w_ * m_vNorm[i] * tmp[k];
				k++;
			}
    }

protected:
	CPermutohedral lattice_;
	BPPottsPotential(const BPPottsPotential&o) {}
	int N1_, N2_;
	float w_;


protected:
	vec_float_t m_vNorm;
};

class BPSemiMetricPotential : public BPPottsPotential {
protected:
    const SemiMetricFunction * function_;

public:
    void apply(vec_float_t &out_values, const vec_float_t &in_values, vec_float_t &tmp, int value_size) const
    {
        lattice_.compute(tmp, in_values, value_size, 0, N1_, N1_, N2_);
        
        // To the metric transform
        float * tmp2 = new float[value_size];
        for (int i = 0; i < N2_; i++) {
            float * out = out_values.data() + i * value_size;
            float * t1 = tmp.data() + i * value_size; 
            function_->apply(tmp2, t1, value_size);
            for (int j = 0; j<value_size; j++)
                out[j] -= w_ * m_vNorm[i] * tmp2[j];
        }
        delete[] tmp2;
    }
    
    BPSemiMetricPotential(const float* features1, const float* features2, int D, int N1, int N2, float w, const SemiMetricFunction* function, bool per_pixel_normalization = true) :BPPottsPotential(features1, features2, D, N1, N2, w, per_pixel_normalization), function_(function) {
    }
};

class PottsPotential : public PairwisePotential
{
public:
	PottsPotential(const float *features, int D, int nNodes, float w, bool per_pixel_normalization = true) : m_nNodes(nNodes), w_(w)
	{
		lattice_.init(features, D, nNodes);
		
		m_vNorm.resize(nNodes);
		std::fill(m_vNorm.begin(), m_vNorm.end(), 1);
	
		// Compute the normalization factor
		lattice_.compute(m_vNorm, m_vNorm, 1);
		if (per_pixel_normalization) 
			for (float &norm : m_vNorm) 
				norm = 1.f / (norm + FLT_EPSILON);
		else {
			float mean_norm = std::accumulate(m_vNorm.begin(), m_vNorm.end(), 0.0f);
			mean_norm = m_vNorm.size() / mean_norm;
			// use a per pixel normalization
			for (float &norm : m_vNorm)
				norm = mean_norm;
		}
	}

	virtual ~PottsPotential(void) {}

	void apply(vec_float_t &out_values, const vec_float_t &in_values, vec_float_t &tmp, int value_size) const
	{
		lattice_.compute(tmp, in_values, value_size);
		
		size_t k = 0;
		for (const float &norm : m_vNorm)
			for (int j = 0; j < value_size; j++, k++)
				out_values[k] += w_ * norm * tmp[k];
	}


protected:
	CPermutohedral lattice_;
	float w_;

protected:
	size_t		m_nNodes;

protected:
	vec_float_t m_vNorm;

};

class SemiMetricPotential : public PottsPotential
{
public:
	void apply(vec_float_t &out_values, const vec_float_t &in_values, vec_float_t  &tmp, int value_size) const {
		lattice_.compute(tmp, in_values, value_size);

		// To the metric transform
		float * tmp2 = new float[value_size];
		for (size_t i = 0; i < m_nNodes; i++) {
			float * out = out_values.data() + i * value_size;
			float * t1 = tmp.data() + i * value_size;
			function_->apply(tmp2, t1, value_size);
			for (int j = 0; j<value_size; j++)
				out[j] -= w_ * m_vNorm[i] * tmp2[j];
		}
		delete[] tmp2;
	}
	
	SemiMetricPotential(const float* features, int D, int N, float w, const SemiMetricFunction* function, bool per_pixel_normalization = true) :PottsPotential(features, D, N, w, per_pixel_normalization), function_(function) {
	}


protected:
	const SemiMetricFunction * function_;
};
