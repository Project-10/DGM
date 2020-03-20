#include "TestKDTree.h"
#include "DGM/random.h"

void CTestKDTree::fill_tree(CKDTree& tree) {

	// Keys
	if (!m_keys.empty()) m_keys.release();

	Mat key(1, nFeatures, CV_8UC1);
	for (int s = 0; s < nSamples; s++) {
		for (int f = 0; f < nFeatures; f++)
			key.at<byte>(0, f) = 5 * random::u(0, 51);
		m_keys.push_back(key);
	}

	// Values
	m_values = Mat(nSamples, 1, CV_8UC1);

	tree.build(m_keys.clone(), m_values);
}

Mat  CTestKDTree::find_nearestNeighbor_bruteForce(const Mat& key)
{
	float min_dist = std::numeric_limits<float>::infinity();
	Mat res;
	const byte* pKey = key.ptr<byte>(0);
	for (int s = 0; s < m_keys.rows; s++) {
		float dist = 0;
		const byte* pKeys = m_keys.ptr<byte>(s);
		for (int f = 0; f < nFeatures; f++) {
			float diff = static_cast<float>(pKeys[f]) - static_cast<float>(pKey[f]);
			dist += diff * diff;
		}
		dist = sqrtf(dist);
		if (dist < min_dist) {
			min_dist = dist;
			res = m_keys.row(s);
		}
	}
	return res;
}

TEST_F(CTestKDTree, findNearestNeighbor)
{
	CKDTree tree;
	
	fill_tree(tree);
	
	// Test Key cantainer
	Mat key(1, nFeatures, CV_8UC1);

	for (int i = 0; i < nTests; i++) {
		// Fill the Test Key
		for (int f = 0; f < nFeatures; f++)
			key.at<byte>(0, f) = 5 * random::u(0, 51) + 1;

		Mat bf_key = find_nearestNeighbor_bruteForce(key);	
		Mat nn_key = tree.findNearestNeighbor(key)->getKey();

		// There might be multiple points with the same distance to the test key. So we compare the distances
		float bf_dist = 0;
		float nn_dist = 0;
		for (int f = 0; f < nFeatures; f++) {
			float key_f = static_cast<float>(key.at<byte>(0, f));
			float bf_f = static_cast<float>(bf_key.at<byte>(0, f));
			float nn_f = static_cast<float>(nn_key.at<byte>(0, f));
			bf_dist += (bf_f - key_f) * (bf_f - key_f);
			nn_dist += (nn_f - key_f) * (nn_f - key_f);
		}
		ASSERT_FLOAT_EQ(bf_dist, nn_dist);
	}
}