#pragma once

#include "gtest/gtest.h"
#include "types.h"
#include "DGM.h"

using namespace DirectGraphicalModels;

class CTestInference : public ::testing::Test {
public:
	CTestInference(void);
	~CTestInference(void) {}
	

protected:
	std::unique_ptr<CGraphPairwise> m_pGraph;
	vec_float_t						m_vPotExact;


protected:
	void	fillGraph(void);
	void	testInferer(CInfer &inferer);


private:
	const byte		m_nStates	= 2;
	const size_t	m_nNodes	= 8;
};