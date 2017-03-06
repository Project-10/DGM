#pragma once

#include "types.h"

namespace DirectGraphicalModels
{
	class CKDNode
	{
	public:
		CKDNode(Mat &key);
		CKDNode(byte median, CKDNode *left, CKDNode *right);
		~CKDNode(void);


	private:
		Mat		  m_key;	
		byte	  m_value;	
		byte	  m_median;	
		CKDNode	* m_pLeft;
		CKDNode	* m_pRight;
	};

}