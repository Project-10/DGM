#pragma once

#include "gtest/gtest.h"
#include "types.h"
#include "DGM.h"

using namespace DirectGraphicalModels;

class CTestInference : public ::testing::Test {
public:
	CTestInference(void);
	~CTestInference(void) = default;
	

protected:
	vec_float_t						m_vPotExact;


protected:
	void	testInferer(CInfer &inferer);


protected:
	const static byte	m_nStates	= 2;
	const static size_t	m_nNodes	= 8;
};