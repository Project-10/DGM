#pragma once

#include "types.h"
#include <numeric>

class CPermutohedral;

// ********************** CPairwisePotential **********************
class CPairwisePotential
{
public:
    CPairwisePotential(void) {}
    virtual ~CPairwisePotential(void) {}
    virtual void apply(vec_float_t &out_values, const vec_float_t &in_values, vec_float_t &tmp, int value_size) const = 0;


private:
    // Copy semantics are disabled
    CPairwisePotential(const CPairwisePotential &rhs) {}
    const CPairwisePotential & operator= (const CPairwisePotential & rhs) { return *this; }
};


// ********************** CPottsPotential **********************
class CPottsPotential : public CPairwisePotential
{
public:
    CPottsPotential(const float *pFeatures, word nFeatures, int nNodes, float w, bool per_pixel_normalization = true);
    virtual ~CPottsPotential(void) {}
    
    void apply(vec_float_t &out_values, const vec_float_t &in_values, vec_float_t &tmp, int value_size) const;
    
    
    // TODO: try private
    // TODO: try without members
protected:
    size_t                          m_nNodes;
    float                           m_w;
    std::unique_ptr<CPermutohedral> m_pLattice;
    vec_float_t                     m_vNorm;
};


// ********************** BPPottsPotential **********************
class CBPPottsPotential : public CPairwisePotential
{
public:
    CBPPottsPotential(const float* features1, const float* features2, int D, int N1, int N2, float w, bool per_pixel_normalization = true);
	virtual ~CBPPottsPotential(void) {}

    virtual void apply(vec_float_t &out_values, const vec_float_t &in_values, vec_float_t &tmp, int value_size) const;


protected:
    float                           m_w;
    std::unique_ptr<CPermutohedral> m_pLattice;
    int                             m_N1;
    int                             m_N2;
	vec_float_t                     m_vNorm;
};


class SemiMetricFunction {
public:
    // For two probabilities apply the semi metric transform: v_i = sum_j mu_ij u_j
    virtual void apply(float *out_values, const float *in_values, int value_size ) const = 0;
};


// ********************** CBPPottsPotential **********************
class CBPSemiMetricPotential : public CBPPottsPotential
{
public:
    CBPSemiMetricPotential(const float *features1, const float *features2, int D, int N1, int N2, float w, const SemiMetricFunction* function, bool per_pixel_normalization = true)
        : CBPPottsPotential(features1, features2, D, N1, N2, w, per_pixel_normalization)
        , m_pFunction(function)
    { }
    virtual ~CBPSemiMetricPotential(void) {}
    
    void apply(vec_float_t &out_values, const vec_float_t &in_values, vec_float_t &tmp, int value_size) const;

    
protected:
    const SemiMetricFunction * m_pFunction;
};


// ********************** CSemiMetricPotential **********************
class CSemiMetricPotential : public CPottsPotential
{
public:
    CSemiMetricPotential(const float *features, int D, int N, float w, const SemiMetricFunction *function, bool per_pixel_normalization = true)
        : CPottsPotential(features, D, N, w, per_pixel_normalization)
        , m_pFunction(function)
    { }
    virtual ~CSemiMetricPotential(void) {}
    
    void apply(vec_float_t &out_values, const vec_float_t &in_values, vec_float_t &tmp, int value_size) const;
	

protected:
	const SemiMetricFunction * m_pFunction;
};
