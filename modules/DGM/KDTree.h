// K-Dimensional Tree class interface
// Copied from http://codereview.stackexchange.com/questions/110225/k-d-tree-implementation-in-c11
#pragma once

#include "types.h"

#include "KDNode.h"

namespace DirectGraphicalModels
{
	class CKDTree
	{
	public:
		/**
		*/
		DllExport CKDTree(void) : m_root(nullptr) {}
		/**
		*/
		DllExport CKDTree(Mat &keys, Mat &values) : m_root(nullptr) { build(keys, values); }	
		DllExport CKDTree(const CKDTree &) = delete;
		DllExport ~CKDTree(void) {}

		DllExport bool											operator=(const CKDTree)  = delete;

		/**
		*/
		DllExport void											reset(void) { m_root.reset(); }
		/**
		*/
		DllExport void											save(const std::string &fileName) const;
		/**
		* @brief Builds a k-d tree on \b keys with corresponding \b values
		* @param keys The tree keys: k-d points: Mat(size: nKeys x k; type: CV_8UC1)
		* @param values The values for every key: Mat(size: nKeys x 1; type: CV_8UC1)
		*/
		DllExport void											build(Mat &keys, Mat &values);
		/**
		* 
		*/
		DllExport std::shared_ptr<const CKDNode>				findNearestNeighbor(const Mat &key) const { return findNearestNeighbors(key, 1).front(); }
		/**
		*
		*/
		DllExport std::vector<std::shared_ptr<const CKDNode>>	findNearestNeighbors(const Mat &key, size_t max) const;
;
	
		/**
		* @brief Returns pointer to the root of the tree
		* @returns The pointer to the root of the tree
		*/
		DllExport std::shared_ptr<const CKDNode>				getRoot(void) const { return m_root; }

	private:
		std::shared_ptr<CKDNode>								buildTree(Mat &data, pair_mat_t &boundingBox);
		std::shared_ptr<const CKDNode>							findNearestNode(const Mat &key) const;


	private:
		std::shared_ptr<CKDNode>	m_root;
	};
}