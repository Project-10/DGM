#include "Graph.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
	void CGraph::reset(void)
	{
		std::for_each(m_vNodes.begin(), m_vNodes.end(), [](Node node) {node.~Node(); });
		m_vNodes.clear();

		std::for_each(m_vEdges.begin(), m_vEdges.end(), [](Edge edge) {edge.~Edge(); });
		m_vEdges.clear();

		m_IDx = 0;
	}

	// Add a new node to the graph
	size_t CGraph::addNode(void)
	{
		m_vNodes.emplace_back(m_IDx);
		return m_IDx++;
	}

	// Add a new node to the graph with specified potentional
	size_t CGraph::addNode(const Mat &pot)
	{
		m_vNodes.emplace_back(m_IDx, pot);
		return m_IDx++;
	}

	// Set or change the potential of node idx
	void CGraph::setNode(size_t node, const Mat &pot)
	{
		DGM_ASSERT_MSG(node < m_vNodes.size(), "Node %zu is out of range %zu", node, m_vNodes.size());
		DGM_ASSERT_MSG((pot.cols == 1) && (pot.rows == m_nStates), "Potential size (%d x %d) does not match (%d x %d)", pot.cols, pot.rows, 1, m_nStates);

		if (!m_vNodes[node].Pot.empty()) m_vNodes[node].Pot.release();
		pot.copyTo(m_vNodes[node].Pot);
	}

	// Return node potential vector 
	void CGraph::getNode(size_t node, Mat &pot) const
	{
		DGM_ASSERT_MSG(node < m_vNodes.size(), "Node %zu is out of range %zu", node, m_vNodes.size());
		DGM_ASSERT_MSG(!m_vNodes[node].Pot.empty(), "Specified node %zu is not set", node);
		m_vNodes[node].Pot.copyTo(pot);
	}

	// Return child nodes ID's
	void CGraph::getChildNodes(size_t node, vec_size_t &vNodes) const
	{
		DGM_ASSERT_MSG(node < m_vNodes.size(), "Node %zu is out of range %zu", node, m_vNodes.size());
		if (!vNodes.empty()) vNodes.clear();
		std::for_each(m_vNodes[node].to.cbegin(), m_vNodes[node].to.cend(), [&](size_t e) {vNodes.push_back(m_vEdges[e].node2); });
	}

	// Return parent nodes ID's
	void CGraph::getParentNodes(size_t node, vec_size_t &vNodes) const
	{
		DGM_ASSERT_MSG(node < m_vNodes.size(), "Node %zu is out of range %zu", node, m_vNodes.size());
		if (!vNodes.empty()) vNodes.clear();
		std::for_each(m_vNodes[node].from.cbegin(), m_vNodes[node].from.cend(), [&](size_t e) {vNodes.push_back(m_vEdges[e].node1); });
	}

	// Add a new (directed) edge to the graph
	void CGraph::addEdge(size_t srcNode, size_t dstNode)
	{
		DGM_ASSERT_MSG(srcNode < m_vNodes.size(), "The source node index %zu is out of range %zu", srcNode, m_vNodes.size());
		DGM_ASSERT_MSG(dstNode < m_vNodes.size(), "The destination node index %zu is out of range %zu", dstNode, m_vNodes.size());
		size_t e = m_vEdges.size();
		m_vEdges.emplace_back(srcNode, dstNode);
		m_vNodes[srcNode].to.push_back(e);
		m_vNodes[dstNode].from.push_back(e);
	}

	// Add a new (directed) edge to the graph with specified potentional
	void CGraph::addEdge(size_t srcNode, size_t dstNode, const Mat &pot)
	{
		DGM_ASSERT_MSG(srcNode < m_vNodes.size(), "The source node index %zu is out of range %zu", srcNode, m_vNodes.size());
		DGM_ASSERT_MSG(dstNode < m_vNodes.size(), "The destination node index %zu is out of range %zu", dstNode, m_vNodes.size());

		// Check if the edge exists
		std::for_each(m_vNodes[srcNode].to.cbegin(), m_vNodes[srcNode].to.cend(), [&](size_t e) { DGM_ASSERT(m_vEdges[e].node2 != dstNode); });

		// Else: create a new one
		size_t e = m_vEdges.size();
		m_vEdges.emplace_back(srcNode, dstNode, pot);
		m_vNodes[srcNode].to.push_back(e);
		m_vNodes[dstNode].from.push_back(e);
	}

	// Set or change the potentional of an directed edge
	void CGraph::setEdge(size_t srcNode, size_t dstNode, const Mat &pot)
	{
		DGM_ASSERT_MSG(srcNode < m_vNodes.size(), "The source node index %zu is out of range %zu", srcNode, m_vNodes.size());
		DGM_ASSERT_MSG(dstNode < m_vNodes.size(), "The destination node index %zu is out of range %zu", dstNode, m_vNodes.size());

		vec_size_t::const_iterator e_t = std::find_if(m_vNodes[srcNode].to.cbegin(), m_vNodes[srcNode].to.cend(), [&](size_t e) {return (m_vEdges[e].node2 == dstNode); });
		DGM_ASSERT_MSG(e_t != m_vNodes[srcNode].to.end(), "The edge (%zu)->(%zu) is not found", srcNode, dstNode);

		if (!m_vEdges[*e_t].Pot.empty()) m_vEdges[*e_t].Pot.release();
		pot.copyTo(m_vEdges[*e_t].Pot);
	}

	// Return edge potential matrix
	void CGraph::getEdge(size_t srcNode, size_t dstNode, Mat &pot) const
	{
		DGM_ASSERT_MSG(srcNode < m_vNodes.size(), "The source node index %zu is out of range %zu", srcNode, m_vNodes.size());
		DGM_ASSERT_MSG(dstNode < m_vNodes.size(), "The destination node index %zu is out of range %zu", dstNode, m_vNodes.size());

		vec_size_t::const_iterator e_t = std::find_if(m_vNodes[srcNode].to.cbegin(), m_vNodes[srcNode].to.cend(), [&](size_t e) {return (m_vEdges[e].node2 == dstNode); });
		DGM_ASSERT_MSG(e_t != m_vNodes[srcNode].to.end(), "The edge (%zu)->(%zu) is not found", srcNode, dstNode);
		if (m_vEdges[*e_t].Pot.empty()) {
 			DGM_WARNING("Edge Potential is empty");
			if (!pot.empty()) pot.release();
		} else m_vEdges[*e_t].Pot.copyTo(pot);
	}

	void CGraph::removeEdge(size_t srcNode, size_t dstNode)
	{
		DGM_ASSERT_MSG(srcNode < m_vNodes.size(), "The source node index %zu is out of range %zu", srcNode, m_vNodes.size());
		DGM_ASSERT_MSG(dstNode < m_vNodes.size(), "The destination node index %zu is out of range %zu", dstNode, m_vNodes.size());

		vec_size_t::const_iterator e_t = std::find_if(m_vNodes[srcNode].to.cbegin(), m_vNodes[srcNode].to.cend(), [&](size_t e) { return (m_vEdges[e].node2 == dstNode); });

		DGM_ASSERT_MSG(e_t != m_vNodes[srcNode].to.end(), "The edge (%zu)->(%zu) is not found", srcNode, dstNode);

		removeEdge(*e_t);
	}

	// Add a new (undirected edge) ark to the graph
	void CGraph::addArk(size_t Node1, size_t Node2)
	{
		addEdge(Node1, Node2);
		addEdge(Node2, Node1);
	}

	// Add a new (undirected edge) ark to the graph with specified potentional
	void CGraph::addArk(size_t Node1, size_t Node2, const Mat &pot)
	{
		Mat Pot;
		sqrt(pot, Pot);
		addEdge(Node1, Node2, Pot);
		addEdge(Node2, Node1, Pot.t());
		Pot.release();
	}

	// Add a new (undirected edge) ark to the graph with specified potentional
	void CGraph::setArk(size_t Node1, size_t Node2, const Mat &pot)
	{
		Mat Pot;
		sqrt(pot, Pot);
		setEdge(Node1, Node2, Pot);
		setEdge(Node2, Node1, Pot.t());
		Pot.release();
	}

	void CGraph::removeArk(size_t Node1, size_t Node2)
	{
		removeEdge(Node1, Node2);
		removeEdge(Node2, Node1);
	}

	// ------------------------------ PRIVATE ------------------------------
	///@todo Optimize the edge removement
	void CGraph::removeEdge(size_t edge)
	{
		DGM_ASSERT_MSG(edge < m_vEdges.size(), "Edge %zu is out of range %zu", edge, m_vEdges.size());

		size_t srcNode = m_vEdges[edge].node1;
		size_t dstNode = m_vEdges[edge].node2;

		m_vEdges[edge].Pot.release();
		//m_vEdges.erase(m_vEdges.begin() + edge);
		
		vec_size_t::const_iterator e_t = std::find(m_vNodes[srcNode].to.cbegin(), m_vNodes[srcNode].to.cend(), edge);
		DGM_ASSERT(e_t != m_vNodes[srcNode].to.cend());
		m_vNodes[srcNode].to.erase(e_t);
		
		vec_size_t::const_iterator e_f = std::find(m_vNodes[dstNode].from.cbegin(), m_vNodes[dstNode].from.cend(), edge);
		DGM_ASSERT(e_f != m_vNodes[dstNode].from.cend());
		m_vNodes[dstNode].from.erase(e_f);
	}

}

