#include "GraphDense.h"
#include "macroses.h"

namespace DirectGraphicalModels 
{
	// Add a new node to the graph
	size_t CGraphDense::addNode(void)
	{
		size_t res = getNumNodes();
		m_nodePotentials.push_back(Mat(1, getNumStates(), CV_32FC1, Scalar(1.0f / getNumStates())));
		return res;
	}
	
	// Add a new node to the graph with specified potentional
	size_t CGraphDense::addNode(const Mat &pot)
	{
		size_t res = getNumNodes();
		m_nodePotentials.push_back(pot.t());
		return res;
	}

	void CGraphDense::addNodes(const Mat &pots)
	{
		m_nodePotentials.push_back(pots);
	}

	// Set or change the potential of node idx
	void CGraphDense::setNode(size_t node, const Mat &pot)
	{
		DGM_ASSERT_MSG(node < getNumNodes(), "Node %zu is out of range %zu", node, getNumNodes());
		DGM_ASSERT_MSG((pot.cols == 1) && (pot.rows == getNumStates()), "Potential size (%d x %d) does not match (%d x %d)", pot.cols, pot.rows, 1, getNumStates());
		
		m_nodePotentials.row(static_cast<int>(node)) = pot.t();
	}

	void CGraphDense::setNodes(const Mat &pots, size_t start_node)
	{
		DGM_ASSERT_MSG(start_node + pots.rows < getNumNodes(), "Node %zu is out of range %zu", start_node + pots.rows, getNumNodes());
		DGM_ASSERT_MSG(pots.cols == getNumStates(), "Potential size (%d) does not match (%d)", pots.cols, getNumStates());
		
		pots.copyTo(m_nodePotentials(Rect(0, static_cast<int>(start_node), getNumStates(), pots.rows)));
	}

	// Return node potential vector 
	void CGraphDense::getNode(size_t node, Mat &pot) const
	{
		DGM_ASSERT_MSG(node < getNumNodes(), "Node %zu is out of range %zu", node, getNumNodes());
		if (pot.empty() || pot.cols != 1 || pot.rows != getNumStates()) pot = Mat(getNumStates(), 1, CV_32FC1);
		
		const float *pPot = m_nodePotentials.ptr<float>(static_cast<int>(node));
		for (byte s = 0; s < getNumStates(); s++)
			pot.at<float>(s, 0) = pPot[s];
	}
}