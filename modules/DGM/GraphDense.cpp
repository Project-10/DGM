#include "GraphDense.h"
#include "macroses.h"

namespace DirectGraphicalModels 
{
	// Add a new node to the graph
	size_t	CGraphDense::addNode(void)
	{
		size_t res = getNumNodes();
		m_nodePotentials.push_back(Mat(1, m_nStates, CV_32FC1, Scalar(1.0f / m_nStates)));
		return res;
	}
	
	// Add a new node to the graph with specified potentional
	size_t	CGraphDense::addNode(const Mat &pot)
	{
		size_t res = getNumNodes();
		m_nodePotentials.push_back(pot.t());
		return res;
	}

	// Set or change the potential of node idx
	void CGraphDense::setNode(size_t node, const Mat &pot)
	{
		DGM_ASSERT_MSG(node < getNumNodes(), "Node %zu is out of range %zu", node, getNumNodes());
		DGM_ASSERT_MSG((pot.cols == 1) && (pot.rows == m_nStates), "Potential size (%d x %d) does not match (%d x %d)", pot.cols, pot.rows, 1, m_nStates);
		
		m_nodePotentials.row(static_cast<int>(node)) = pot.t();
	}

	// Return node potential vector 
	void CGraphDense::getNode(size_t node, Mat &pot) const
	{
		DGM_ASSERT_MSG(node < getNumNodes(), "Node %zu is out of range %zu", node, getNumNodes());
		if (pot.empty() || pot.cols != 1 || pot.rows != m_nStates) pot = Mat(m_nStates, 1, CV_32FC1);
		
		const float *pPot = m_nodePotentials.ptr<float>(static_cast<int>(node));
		for (byte s = 0; s < m_nStates; s++)
			pot.at<float>(s, 0) = pPot[s];
	}
}