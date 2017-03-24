// k-D node class for k-D trees
// Written by Sergey Kosov in 2017 for Project X
#pragma once

#include "types.h"

namespace DirectGraphicalModels
{
	// ================================ Base Random Model Class ================================
	/**
	* @brief K-D %Node class for K-D Trees
	* @details This class is used for an implementation of a non-uniform <a href="https://en.wikipedia.org/wiki/K-d_tree">k-d tree</a>.
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CKDNode : public std::enable_shared_from_this<CKDNode>
	{
	public:
		/**
		* @brief Leaf node constructor
		* @param key The node key (k-d point): Mat(size: 1 x k; type: CV_8UC1))
		* @param value The node value
		*/
		DllExport CKDNode(Mat &key, byte value) 
			: CKDNode(key, value, std::make_pair(Mat(), Mat()), 0, 0, nullptr, nullptr) {}
		/**
		* @brief Branch node constructor
		* @details All the points with \b key[\b splitDim] < \b splitVal must be assigned to the \b left sub-tree,
		* and the points \b key[\b splitDim] >= \b splitVal - to the \b right.
		* @param boundingBox The spatial bounding box, containing all the keys for the current branch: pair<Mat, Mat>(minCoordinates, maxCoordinates).
		* @param splitVal The threshold in which the split of the k-d space is performed.
		* @param splitDim The dimension ( [0; k) ) in which the split of the k-d space is performed. 
		* @param left The pointer to the root of the \a left sub-tree. 
		* @param right The pointer to the root of the \a right sub-tree.
		*/
		DllExport CKDNode(pair_mat_t &boundingBox, byte splitVal, int splitDim, std::shared_ptr<CKDNode> left, std::shared_ptr<CKDNode> right)
			: CKDNode(Mat(), 0, boundingBox, splitVal, splitDim, left, right) {}
		// Copy constructor
		DllExport CKDNode(const CKDNode &) = delete;
		// Destructor
		DllExport ~CKDNode(void) {};
		// Assignment operator
		DllExport bool						operator=(const CKDNode)  = delete;

		/**
		* @brief Checks whether the node is either leaf or brach node
		* @retval true if the node is a leaf-node
		* @retval false if the node is a branch-node
		*/
		DllExport bool						isLeaf(void) const { return (!m_pLeft && !m_pRight); }
		/**
		* @brief Auxiliary recursive method for finding the nearest (in terms of the Euclidian distance between the keys) \b nearestNeighbor node.
		* @param[in] key The search key (k-d point): Mat(size: 1 x k; type: CV_8UC1))
		* @param[in,out] searchBox The bounding box, that is used to find the overlapping branch nodes
		* @param[in,out] searchRadius The radius of a k-d sphere, within which the node is searched
		* @param[out] nearestNeighbor The resulting nearest node
		*/
		DllExport void						findNearestNeighbor(const Mat &key, pair_mat_t &searchBox, float &searchRadius, std::shared_ptr<const CKDNode> &nearestNeighbor) const;
		/**
		* @brief Auxiliary recursive method for finding the ka-nearest (in terms of the Euclidian distance between the keys) \b nearestNeighbors nodes.
		* @param[in] key The search key (k-d point): Mat(size: 1 x k; type: CV_8UC1))
		* @param[in] k The number of desired neares neighbors
		* @param[in,out] searchBox The bounding box, that is used to find the overlapping branch nodes
		* @param[in,out] searchRadius The radius of a k-d sphere, within which the nodes are searched
		* @param[out] nearestNeighbors The resulting ka-nearest nodes
		*/
		DllExport void						findNearestNeighbors(const Mat &key, size_t k, pair_mat_t &searchBox, float &searchRadius, std::vector<std::shared_ptr<const CKDNode>> &nearestNeighbors) const;
		/**
		* @brief Returns the key of the leaf-node (k-d point)
		* @returns The key of the node: Mat(size: 1 x k; type: CV_8UC1)
		*/
		DllExport Mat						getKey(void)			const { return m_key; }
		/**
		* @brief Returns the value of the leaf-node
		* @retval The value of the node
		*/
		DllExport byte						getValue(void)			const { return m_value; }
		/**
		* @brief Returns the spatial bounding box, containing all the keys for the current branch.
		* @returns The bounding box (pair<Mat, Mat>(minCoordinates, maxCoordinates))
		*/
		DllExport pair_mat_t				getBoundingBox(void)	const { return isLeaf() ? std::make_pair(m_key, m_key) : m_boundingBox; }
		/**
		* @brief Returns the split value of the brach-node
		* @details The split value is a threshold in which the split of the k-d space is performed.
		* @returns The split value
		*/
		DllExport byte						getSplitVal(void)		const { return m_splitVal; }
		/**
		* @brief Returns the split dimension of the branch-node
		* @details The split dimension is the dimension in which the split of the k-d space is performed. 
		* @returns The split dimension: (a value from the interval [0; k))
		*/
		DllExport int						getSplitDim(void)		const { return m_splitDim; }
		/**
		* @brief  Returns the pointer to the \a left child
		* @returns The pointer to the root-node of the \a left sub-tree
		*/
		DllExport std::shared_ptr<CKDNode>	Left(void)				const { return m_pLeft; }
		/**
		* @brief Returns the pointer to the \a right child
		* @returns The pointer to the root-node of the \a right sub-tree
		*/
		DllExport std::shared_ptr<CKDNode>	Right(void)				const { return m_pRight; }


	private:
		DllExport CKDNode(Mat &key, byte value, pair_mat_t &boundingBox, byte splitVal, int splitDim, std::shared_ptr<CKDNode> left, std::shared_ptr<CKDNode> right);


	private:
		Mat									m_key;	
		byte								m_value;
		pair_mat_t							m_boundingBox;
		byte								m_splitVal;	
		int									m_splitDim;
		std::shared_ptr<CKDNode>			m_pLeft;
		std::shared_ptr<CKDNode>			m_pRight;
	};

}