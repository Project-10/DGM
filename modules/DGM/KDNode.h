#pragma once

#include "types.h"

namespace DirectGraphicalModels
{
	DllExport class CKDNode
	{
	public:
		CKDNode(Mat &key, byte value);
		CKDNode(byte splitVal, int splitDim, std::shared_ptr<CKDNode> left, std::shared_ptr<CKDNode> right);
		~CKDNode(void);

		bool						isLeaf(void)		const { return (!m_pLeft && !m_pRight); }
		byte						getSplitVal(void)	const { return m_splitVal; }
		int							getSplitDim(void)	const { return m_splitDim; }
		std::shared_ptr<CKDNode>	Left(void)			const { return m_pLeft; }
		std::shared_ptr<CKDNode>	Right(void)			const { return m_pRight; }



	private:
		Mat							m_key;	
		byte						m_value;
		
		byte						m_splitVal;	
		int							m_splitDim;
		std::shared_ptr<CKDNode>	m_pLeft;
		std::shared_ptr<CKDNode>	m_pRight;
	};

}