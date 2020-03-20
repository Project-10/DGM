// K-Dimensional Tree class interface
// Written by Sergey Kosov in 2017 for Project X
// Inspired by http://codereview.stackexchange.com/questions/110225/k-d-tree-implementation-in-c11
#pragma once

#include "types.h"
#include "KDNode.h"

namespace DirectGraphicalModels
{
	// ================================ k-D Tree Class ================================
	/**
	* @brief Class implementing k-D Tree data structure
	* @details This class implementats a non-uniform <a href="https://en.wikipedia.org/wiki/K-d_tree" target="blank">k-D Tree</a> data structure.
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CKDTree
	{
	public:
		/**
		* @brief Default constructor
		*/
		DllExport CKDTree(void) = default;
		/**
		* @brief Constructor
		* @param keys The tree keys: k-d points: Mat(size: nKeys x k; type: CV_8UC1)
		* @param values The values for every key: Mat(size: nKeys x 1; type: CV_8UC1)
		*/
		DllExport CKDTree(Mat &keys, Mat &values) { build(keys, values); }	
		DllExport CKDTree(const CKDTree&) = delete;
		DllExport ~CKDTree(void) = default;

		DllExport bool											operator=(const CKDTree&) = delete;

		/**
		* @brief Resets the tree
		*/
		DllExport void											reset(void) { m_root.reset(); }
		/**
		* @brief Saves the tree into a file
		* @param fileName The output file name
		*/
		DllExport void											save(const std::string &fileName) const;
		/**
		* @brief Loads a tree from the file
		* @param fileName The output file name
		*/
		DllExport void											load(const std::string &fileName);
		/**
		* @brief Builds a k-d tree on \b keys with corresponding \b values
		* @param keys The tree keys: k-d points: Mat(size: nKeys x k; type: CV_8UC1)
		* > The \b keys matrix is modified by this function
		* @param values The values for every key: Mat(size: nKeys x 1; type: CV_8UC1)
		*/
		DllExport void											build(Mat &keys, Mat &values);
		/**
		* @brief Finds the nearest neighbor to the \b key
		* @param key The search key: k-d point: Mat(size: 1 x k; type: CV_8UC1)
		* @returns The %Node (tree leaf) with the key, which is the most close to the argument \b key
		*/
		DllExport std::shared_ptr<const CKDNode>				findNearestNeighbor(const Mat &key) const { return findNearestNeighbors(key, 1).front(); }
		/**
		* @brief Finds up to \b maxNeighbors nearest neighbors to the \b key
		* @param key The search key: k-d point: Mat(size: 1 x k; type: CV_8UC1)
		* @param maxNeighbors maximum number of neighbor nodes to find
		* @returns The array of %Nodes (tree leaves) with the keys, which are the most close to the argument \b key
		*/
		DllExport std::vector<std::shared_ptr<const CKDNode>>	findNearestNeighbors(const Mat &key, size_t maxNeighbors) const;
		/**
		* @brief Returns pointer to the root of the tree
		* @returns The pointer to the root of the tree
		*/
		DllExport std::shared_ptr<const CKDNode>				getRoot(void) const { return m_root; }


	private:
		std::shared_ptr<CKDNode>								loadTree(FILE *pFile, int k);
		std::shared_ptr<CKDNode>								buildTree(Mat& data, const pair_mat_t& boundingBox);
		std::shared_ptr<const CKDNode>							findNearestNode(const Mat &key) const;


	private:
		std::shared_ptr<CKDNode>	m_root = nullptr;
	};
}