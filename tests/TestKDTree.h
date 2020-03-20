#pragma once

#include "gtest/gtest.h"
#include "types.h"
#include "DGM.h"

using namespace DirectGraphicalModels;

const int	nSamples	= 10000;
const int	nFeatures	= 16;
const int	nTests		= 100;

class CTestKDTree : public ::testing::Test {
public:
	CTestKDTree(void) = default;
	~CTestKDTree(void) = default;

	void fill_tree(CKDTree& tree);
	Mat  find_nearestNeighbor_bruteForce(const Mat& key);

private:
	Mat m_keys		= Mat();
	Mat m_values	= Mat();
};