#pragma once

#include "gtest/gtest.h"
#include "types.h"
#include "DGM.h"

using namespace DirectGraphicalModels;



class CTestParamEstimation : public ::testing::Test {
public:
	CTestParamEstimation(void)
		: m_vInitParams(nParams)
		, m_vInitDeltas(nParams)
		, m_vSolution(nParams)
	{}
	~CTestParamEstimation(void) = default;


protected:	
	void	testParamEstimation(CParamEstimation& paramEstimator);
	float	objectiveFunction(const vec_float_t& vParams);

private:
	vec_float_t m_vInitParams;
	vec_float_t m_vInitDeltas;
	vec_float_t	m_vSolution;


protected:	// Test configuration
	const static size_t nParams = 16;
};
