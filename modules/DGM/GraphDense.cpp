#include "GraphDense.h"
#include "macroses.h"

namespace DirectGraphicalModels 
{
	// Add a new node to the graph with specified potentional
	size_t CGraphDense::addNode(const Mat &pot)
	{
		size_t res = getNumNodes();
		m_nodePotentials.push_back(pot.empty() ? Mat(1, getNumStates(), CV_32FC1, Scalar(1.0f / getNumStates())) : pot.t());
		return res;
	}

	// Set or change the potential of node idx
	void CGraphDense::setNode(size_t node, const Mat &pot)
	{
		// Assertions
		DGM_ASSERT_MSG(node < getNumNodes(), "Node %zu is out of range %zu", node, getNumNodes());
		DGM_ASSERT_MSG((pot.cols == 1) && (pot.rows == getNumStates()), "Potential size (%d x %d) does not match (%d x %d)", pot.cols, pot.rows, 1, getNumStates());
		DGM_ASSERT_MSG(pot.type() == CV_32FC1, "Potential type is not CV_32FC1");

		m_nodePotentials.row(static_cast<int>(node)) = pot.t();
	}

	void CGraphDense::setNodes(size_t start_node, const Mat &pots)
	{
		// Assertions
		DGM_ASSERT_MSG(start_node + pots.rows <= getNumNodes(), "Node %zu is out of range %zu", start_node + pots.rows, getNumNodes());
		DGM_ASSERT_MSG(pots.cols == getNumStates(), "Potential size (%d) does not match (%d)", pots.cols, getNumStates());
		DGM_ASSERT_MSG(pots.type() == CV_32FC1, "Potentials type is not CV_32FC1");

		pots.copyTo(m_nodePotentials(Rect(0, static_cast<int>(start_node), getNumStates(), pots.rows)));
	}

	// Return node potential vector 
	void CGraphDense::getNode(size_t node, Mat &pot) const
	{
		DGM_ASSERT_MSG(node < getNumNodes(), "Node %zu is out of range %zu", node, getNumNodes());
		if (pot.empty() || pot.cols != 1 || pot.rows != getNumStates() || pot.type() != CV_32FC1) 
			pot = Mat(getNumStates(), 1, CV_32FC1);
		
		const float *pPot = m_nodePotentials.ptr<float>(static_cast<int>(node));
		for (byte s = 0; s < getNumStates(); s++)
			pot.at<float>(s, 0) = pPot[s];
	}

	void CGraphDense::getNodes(size_t start_node, size_t num_nodes, Mat &pots) const
	{
		if (!num_nodes) num_nodes = getNumNodes() - start_node;
		DGM_ASSERT_MSG(start_node + num_nodes <= getNumNodes(), "The given ranges exceed the number of nodes(%zu)", getNumNodes());
//		if (pots.empty() || pots.cols != getNumStates() || pots.rows != num_nodes || pots.type() != CV_32FC1)
//			pots = Mat(static_cast<int>(num_nodes), getNumStates(), CV_32FC1);
		m_nodePotentials(Rect(0, static_cast<int>(start_node), getNumStates(), static_cast<int>(num_nodes))).copyTo(pots);
	}

	void CGraphDense::getChildNodes(size_t node, vec_size_t &vNodes) const
	{
		DGM_ASSERT_MSG(node < getNumNodes(), "Node %zu is out of range %zu", node, getNumNodes());
		if (!vNodes.empty()) vNodes.clear();
		for (size_t i = 0; i < getNumNodes(); i++)
			if (i != node)
				vNodes.push_back(i);
	}
}
