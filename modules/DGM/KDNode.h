#pragma once

#include "types.h"

namespace DirectGraphicalModels
{
	class CKDNode
	{
	public:
		DllExport CKDNode(Mat &key, byte value) 
			: CKDNode(key, value, std::make_pair(Mat(), Mat()), 0, 0, nullptr, nullptr) {}
		DllExport CKDNode(pair_mat_t &boundingBox, byte splitVal, int splitDim, std::shared_ptr<CKDNode> left, std::shared_ptr<CKDNode> right)
			:CKDNode(Mat(), 0, boundingBox, splitVal, splitDim, left, right) {}
		DllExport virtual ~CKDNode(void) {};

		
		DllExport bool						isLeaf(void) const { return (!m_pLeft && !m_pRight); }

		DllExport void						findNearestNeighbor(Mat &key, float &minDistance, pair_mat_t &searchBox) const;

		DllExport Mat						getKey(void)			const { return m_key; }
		DllExport byte						getValue(void)			const { return m_value; }
		DllExport pair_mat_t				getBoundingBox(void)	const { return isLeaf() ? std::make_pair(m_key, m_key) : m_boundingBox; }
		DllExport byte						getSplitVal(void)		const { return m_splitVal; }
		DllExport int						getSplitDim(void)		const { return m_splitDim; }
		DllExport std::shared_ptr<CKDNode>	Left(void)				const { return m_pLeft; }
		DllExport std::shared_ptr<CKDNode>	Right(void)				const { return m_pRight; }


	protected:
		DllExport CKDNode(Mat &key, byte value, pair_mat_t &boundingBox, byte splitVal, int splitDim, std::shared_ptr<CKDNode> left, std::shared_ptr<CKDNode> right);


	private:
		Mat							m_key;	
		byte						m_value;
		pair_mat_t					m_boundingBox;
		byte						m_splitVal;	
		int							m_splitDim;
		std::shared_ptr<CKDNode>	m_pLeft;
		std::shared_ptr<CKDNode>	m_pRight;
	};

}