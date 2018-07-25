#pragma once

#include "permutohedral.h"

class PairwisePotential {
public:
    virtual void apply(float *out_values, const float *in_values, float *tmp, int value_size) const = 0;
};


class SemiMetricFunction {
public:
    // For two probabilities apply the semi metric transform: v_i = sum_j mu_ij u_j
    virtual void apply(float *out_values, const float *in_values, int value_size ) const = 0;
};


class BPPottsPotential : public PairwisePotential {
protected:
    CPermutohedral lattice_;
    BPPottsPotential(const BPPottsPotential&o) {}
    int N1_, N2_;
    float w_;
    float *norm_;
public:
    ~BPPottsPotential() {
        if (norm_) delete[] norm_;
    }
    
    BPPottsPotential(const float* features1, const float* features2, int D, int N1, int N2, float w, bool per_pixel_normalization = true) : N1_(N1), N2_(N2), w_(w) {
        float * features = new float[(N1_ + N2_)*D];
        memset(features, 0, (N1_ + N2_)*D * sizeof(float));
        memcpy(features, features1, N1_ * D * sizeof(float));
        memcpy(features + N1_ * D, features2, N2_ * D * sizeof(float));
        lattice_.init(features, D, N1_ + N2_);
        delete[] features;
        
        norm_ = new float[N2_];
        memset(norm_, 0, N2_ * sizeof(float));
        float *tmp = new float[N1_];
        for (int i = 0; i < N1_; i++) tmp[i] = 1;
        // Compute the normalization factor
        lattice_.compute(norm_, tmp, 1, 0, N1_, N1_, N2_);
        if (per_pixel_normalization) {
            // use a per pixel normalization
            for (int i = 0; i<N2_; i++)
                norm_[i] = 1.f / (norm_[i] + 1e-20f);
        }
        else {
            float mean_norm = 0;
            for (int i = 0; i<N2_; i++)
                mean_norm += norm_[i];
            mean_norm = N2_ / mean_norm;
            // use a per pixel normalization
            for (int i = 0; i<N2_; i++)
                norm_[i] = mean_norm;
        }
        delete[] tmp;
    }
    
    virtual void apply(float * out_values, const float * in_values, float * tmp, int value_size) const {
        lattice_.compute(tmp, in_values, value_size, 0, N1_, N1_, N2_);
        for (int i = 0, k = 0; i<N2_; i++)
            for (int j = 0; j<value_size; j++, k++)
                out_values[k] += w_ * norm_[i] * tmp[k];
    }
};

class BPSemiMetricPotential : public BPPottsPotential {
protected:
    const SemiMetricFunction * function_;
public:
    void apply(float* out_values, const float* in_values, float* tmp, int value_size) const
    {
        lattice_.compute(tmp, in_values, value_size, 0, N1_, N1_, N2_);
        
        // To the metric transform
        float * tmp2 = new float[value_size];
        for (int i = 0; i<N2_; i++) {
            float * out = out_values + i * value_size;
            float * t1 = tmp + i * value_size; ;
            function_->apply(tmp2, t1, value_size);
            for (int j = 0; j<value_size; j++)
                out[j] -= w_ * norm_[i] * tmp2[j];
        }
        delete[] tmp2;
    }
    
    BPSemiMetricPotential(const float* features1, const float* features2, int D, int N1, int N2, float w, const SemiMetricFunction* function, bool per_pixel_normalization = true) :BPPottsPotential(features1, features2, D, N1, N2, w, per_pixel_normalization), function_(function) {
    }
};


