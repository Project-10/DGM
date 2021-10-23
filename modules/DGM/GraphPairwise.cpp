#include "GraphPairwise.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
	void CGraphPairwise::reset(void)
	{
		m_vNodes.clear();
		m_vEdges.clear();
		m_IDx = 0;
	}

	// Add a new node to the graph with specified potentional
	size_t CGraphPairwise::addNode(const Mat &pot)
	{
		m_vNodes.push_back(ptr_node_t(new Node(m_IDx, pot))); 
		return m_IDx++;
	}

	// Set or change the potential of node idx
	void CGraphPairwise::setNode(size_t node, const Mat &pot)
	{
		DGM_ASSERT_MSG(node < m_vNodes.size(), "Node %zu is out of range %zu", node, m_vNodes.size());
		DGM_ASSERT_MSG((pot.cols == 1) && (pot.rows == getNumStates()), "Potential size (%d x %d) does not match (%d x %d)", pot.cols, pot.rows, 1, getNumStates());

		if (!m_vNodes[node]->Pot.empty()) m_vNodes[node]->Pot.release();
		pot.copyTo(m_vNodes[node]->Pot);
	}

	// Return node potential vector 
	void CGraphPairwise::getNode(size_t node, Mat &pot) const
	{
		DGM_ASSERT_MSG(node < m_vNodes.size(), "Node %zu is out of range %zu", node, m_vNodes.size());
		DGM_ASSERT_MSG(!m_vNodes[node]->Pot.empty(), "Specified node %zu is not set", node);
		m_vNodes[node]->Pot.copyTo(pot);
	}

	// Return child nodes ID's
	void CGraphPairwise::getChildNodes(size_t node, vec_size_t &vNodes) const
	{
		DGM_ASSERT_MSG(node < getNumNodes(), "Node %zu is out of range %zu", node, getNumNodes());
		if (!vNodes.empty()) vNodes.clear();
		for (size_t e: m_vNodes[node]->to) { vNodes.push_back(m_vEdges[e]->node2); }
	}

	// Return parent nodes ID's
	void CGraphPairwise::getParentNodes(size_t node, vec_size_t &vNodes) const
	{
		DGM_ASSERT_MSG(node < getNumNodes(), "Node %zu is out of range %zu", node, getNumNodes());
		if (!vNodes.empty()) vNodes.clear();
		for (size_t e: m_vNodes[node]->from) { vNodes.push_back(m_vEdges[e]->node1); }
	}


	// Add a new (directed) edge to the graph with specified potentional
	void CGraphPairwise::addEdge(size_t srcNode, size_t dstNode, byte group, const Mat &pot)
	{
		DGM_ASSERT_MSG(srcNode < m_vNodes.size(), "The source node index %zu is out of range %zu", srcNode, m_vNodes.size());
		DGM_ASSERT_MSG(dstNode < m_vNodes.size(), "The destination node index %zu is out of range %zu", dstNode, m_vNodes.size());

		// Check if the edge exists
		if (m_vNodes[srcNode]->to.size() < m_vNodes[dstNode]->from.size())
			for (size_t &e : m_vNodes[srcNode]->to) { DGM_ASSERT(m_vEdges[e]->node2 != dstNode); }
		else
			for (size_t &e : m_vNodes[dstNode]->from) { DGM_ASSERT(m_vEdges[e]->node1 != srcNode); }

		// Else: create a new one
		size_t e = m_vEdges.size();
		m_vEdges.push_back(ptr_edge_t(new Edge(srcNode, dstNode, group, pot)));
		m_vNodes[srcNode]->to.push_back(e);
		m_vNodes[dstNode]->from.push_back(e);
	}

	// Set or change the potentional of an directed edge
	void CGraphPairwise::setEdge(size_t srcNode, size_t dstNode, const Mat &pot)
	{
		DGM_ASSERT_MSG(srcNode < m_vNodes.size(), "The source node index %zu is out of range %zu", srcNode, m_vNodes.size());
		DGM_ASSERT_MSG(dstNode < m_vNodes.size(), "The destination node index %zu is out of range %zu", dstNode, m_vNodes.size());

		vec_size_t::const_iterator e_t = std::find_if(m_vNodes[srcNode]->to.cbegin(), m_vNodes[srcNode]->to.cend(), [&](size_t e) { return (m_vEdges[e]->node2 == dstNode); });
		DGM_ASSERT_MSG(e_t != m_vNodes[srcNode]->to.end(), "The edge (%zu)->(%zu) is not found", srcNode, dstNode);

		pot.copyTo(m_vEdges[*e_t]->Pot);
	}

	void CGraphPairwise::setEdges(std::optional<byte> group, const Mat& pot)
	{
#ifdef ENABLE_PDP
		parallel_for_(Range(0, static_cast<int>(m_vEdges.size())), [group, &pot, this](const Range& range) {
			for (int i = range.start; i < range.end; i++) {
				ptr_edge_t& pEdge = m_vEdges[i];
				if (!group || pEdge->group_id == group.value())
					pot.copyTo(pEdge->Pot);
			}
		});
#else 			
		for (ptr_edge_t& pEdge : m_vEdges) {
			if(!group || pEdge->group_id == group.value())
					pot.copyTo(pEdge->Pot);	
		}
#endif
	}

	// Return edge potential matrix
	void CGraphPairwise::getEdge(size_t srcNode, size_t dstNode, Mat &pot) const
	{
		DGM_ASSERT_MSG(srcNode < m_vNodes.size(), "The source node index %zu is out of range %zu", srcNode, m_vNodes.size());
		DGM_ASSERT_MSG(dstNode < m_vNodes.size(), "The destination node index %zu is out of range %zu", dstNode, m_vNodes.size());

		vec_size_t::const_iterator e_t = std::find_if(m_vNodes[srcNode]->to.cbegin(), m_vNodes[srcNode]->to.cend(), [&](size_t e) { return (m_vEdges[e]->node2 == dstNode); });
		DGM_ASSERT_MSG(e_t != m_vNodes[srcNode]->to.end(), "The edge (%zu)->(%zu) is not found", srcNode, dstNode);
		if (m_vEdges[*e_t]->Pot.empty()) {
 			DGM_WARNING("Edge Potential is empty");
			if (!pot.empty()) pot.release();
		} else m_vEdges[*e_t]->Pot.copyTo(pot);
	}

	void CGraphPairwise::setEdgeGroup(size_t srcNode, size_t dstNode, byte group)
	{
		DGM_ASSERT_MSG(srcNode < m_vNodes.size(), "The source node index %zu is out of range %zu", srcNode, m_vNodes.size());
		DGM_ASSERT_MSG(dstNode < m_vNodes.size(), "The destination node index %zu is out of range %zu", dstNode, m_vNodes.size());

		vec_size_t::const_iterator e_t = std::find_if(m_vNodes[srcNode]->to.cbegin(), m_vNodes[srcNode]->to.cend(), [&](size_t e) { return (m_vEdges[e]->node2 == dstNode); });
		DGM_ASSERT_MSG(e_t != m_vNodes[srcNode]->to.end(), "The edge (%zu)->(%zu) is not found", srcNode, dstNode);
		m_vEdges[*e_t]->group_id = group;
	}

	byte CGraphPairwise::getEdgeGroup(size_t srcNode, size_t dstNode) const
	{
		DGM_ASSERT_MSG(srcNode < m_vNodes.size(), "The source node index %zu is out of range %zu", srcNode, m_vNodes.size());
		DGM_ASSERT_MSG(dstNode < m_vNodes.size(), "The destination node index %zu is out of range %zu", dstNode, m_vNodes.size());

		vec_size_t::const_iterator e_t = std::find_if(m_vNodes[srcNode]->to.cbegin(), m_vNodes[srcNode]->to.cend(), [&](size_t e) { return (m_vEdges[e]->node2 == dstNode); });
		DGM_ASSERT_MSG(e_t != m_vNodes[srcNode]->to.end(), "The edge (%zu)->(%zu) is not found", srcNode, dstNode);

		return m_vEdges[*e_t]->group_id;
	}

	void CGraphPairwise::removeEdge(size_t srcNode, size_t dstNode)
	{
		DGM_ASSERT_MSG(srcNode < m_vNodes.size(), "The source node index %zu is out of range %zu", srcNode, m_vNodes.size());
		DGM_ASSERT_MSG(dstNode < m_vNodes.size(), "The destination node index %zu is out of range %zu", dstNode, m_vNodes.size());

		vec_size_t::const_iterator e_t = std::find_if(m_vNodes[srcNode]->to.cbegin(), m_vNodes[srcNode]->to.cend(), [&](size_t e) { return (m_vEdges[e]->node2 == dstNode); });

		DGM_ASSERT_MSG(e_t != m_vNodes[srcNode]->to.end(), "The edge (%zu)->(%zu) is not found", srcNode, dstNode);

		removeEdge(*e_t);
	}

	bool CGraphPairwise::isEdgeExists(size_t srcNode, size_t dstNode) const
	{
		DGM_ASSERT_MSG(srcNode < m_vNodes.size(), "The source node index %zu is out of range %zu", srcNode, m_vNodes.size());
		DGM_ASSERT_MSG(dstNode < m_vNodes.size(), "The destination node index %zu is out of range %zu", dstNode, m_vNodes.size());
		
		auto e_t = std::find_if(m_vNodes[srcNode]->to.cbegin(), m_vNodes[srcNode]->to.cend(), [&](size_t e) { return (m_vEdges[e]->node2 == dstNode); });

		if (e_t == m_vNodes[srcNode]->to.cend()) return false;
		else									 return true;
	}


    // ------------------------------ PRIVATE ------------------------------
	///@todo Optimize the edge removement
	void CGraphPairwise::removeEdge(size_t edge)
	{
		DGM_ASSERT_MSG(edge < m_vEdges.size(), "Edge %zu is out of range %zu", edge, m_vEdges.size());

		size_t srcNode = m_vEdges[edge]->node1;
		size_t dstNode = m_vEdges[edge]->node2;

		m_vEdges[edge]->Pot.release();
		//m_vEdges.erase(m_vEdges.begin() + edge);
		
		vec_size_t::const_iterator e_t = std::find(m_vNodes[srcNode]->to.cbegin(), m_vNodes[srcNode]->to.cend(), edge);
		DGM_ASSERT(e_t != m_vNodes[srcNode]->to.cend());
		m_vNodes[srcNode]->to.erase(e_t);
		
		vec_size_t::const_iterator e_f = std::find(m_vNodes[dstNode]->from.cbegin(), m_vNodes[dstNode]->from.cend(), edge);
		DGM_ASSERT(e_f != m_vNodes[dstNode]->from.cend());
		m_vNodes[dstNode]->from.erase(e_f);
	}
}

