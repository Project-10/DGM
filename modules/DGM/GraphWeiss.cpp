#include "GraphWeiss.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
	// Constructor
	CGraphWeiss::CGraphWeiss(byte nStates) 
		: IGraphPairwise(nStates)
		, m_IDx(0)
	{}

	// Destructor: clean up the Node objects
	CGraphWeiss::~CGraphWeiss(void)
	{
		size_t nNodes = m_vpNodes.size();
		for (size_t n = 0; n < nNodes; n++) 
			delete m_vpNodes.at(n);	
		m_vpNodes.clear();
	}

	void CGraphWeiss::reset(void)
	{
		size_t nNodes = m_vpNodes.size();
		for (size_t n = 0; n < nNodes; n++) 
			delete m_vpNodes.at(n);	
		m_vpNodes.clear();	
		m_IDx = 0;
	}

	// Add a new node to the graph with specified potentional
	size_t CGraphWeiss::addNode(const Mat &pot)
	{
		Node *n = new Node(pot);
		n->id = m_IDx;
		m_vpNodes.push_back(n);
		return m_IDx++;
	}

	// Set or change the potential of node idx
	void CGraphWeiss::setNode(size_t node, const Mat &pot)
	{
		// Assertions
		DGM_ASSERT_MSG(node < m_vpNodes.size(), "Node %zu is out of range %zu", node, m_vpNodes.size());

		if (!m_vpNodes.at(node)->Pot.empty()) m_vpNodes.at(node)->Pot.release();
		pot.copyTo(m_vpNodes.at(node)->Pot);
	}

	// Return node potential vector 
	void CGraphWeiss::getNode(size_t node, Mat &pot) const
	{
		// Assertions
		DGM_ASSERT_MSG(node < m_vpNodes.size(), "Node %zu is out of range %zu", node, m_vpNodes.size());
		DGM_ASSERT_MSG(!m_vpNodes.at(node)->Pot.empty(), "Specified node %zu is not set", node);

		m_vpNodes.at(node)->Pot.copyTo(pot);
	}

	// Return child nodes ID's
	void CGraphWeiss::getChildNodes(size_t node, vec_size_t &vNodes) const
	{
		// Assertion
		DGM_ASSERT_MSG(node < m_vpNodes.size(), "Node %zu is out of range %zu", node, m_vpNodes.size());
	
		for (size_t e_t = 0; e_t < m_vpNodes.at(node)->to.size(); e_t++) 
			vNodes.push_back(m_vpNodes.at(node)->to.at(e_t)->node2->id);
	}

	// Return parent nodes ID's
	void CGraphWeiss::getParentNodes(size_t node, vec_size_t &vNodes) const
	{
		// Assertion
		DGM_ASSERT_MSG(node < m_vpNodes.size(), "Node %zu is out of range %zu", node, m_vpNodes.size());
	
		for (size_t e_f = 0; e_f < m_vpNodes.at(node)->from.size(); e_f++) 
			vNodes.push_back(m_vpNodes.at(node)->from.at(e_f)->node1->id);
	}

	// Add a new (directed) edge to the graph with specified potentional
	void CGraphWeiss::addEdge(size_t srcNode, size_t dstNode, const Mat &pot)
	{
		// Assertions
		DGM_ASSERT_MSG(srcNode < m_vpNodes.size(), "The source node index %zu is out of range %zu", srcNode, m_vpNodes.size());
		DGM_ASSERT_MSG(dstNode < m_vpNodes.size(), "The destination node index %zu is out of range %zu", dstNode, m_vpNodes.size());

		// Check if the edge exists
		for (size_t e_t = 0; e_t < m_vpNodes.at(srcNode)->to.size(); e_t++) {
			Edge *edge_to = m_vpNodes.at(srcNode)->to.at(e_t);
			DGM_ASSERT(edge_to->node2->id != m_vpNodes.at(dstNode)->id);
		} // e_t
		
		// Else: create a new one
		Edge *e = new Edge();
		e->node1 = m_vpNodes.at(srcNode);
		e->node2 = m_vpNodes.at(dstNode);
		if (!pot.empty()) pot.copyTo(e->Pot);
		m_vpNodes.at(srcNode)->to.push_back(e);
		m_vpNodes.at(dstNode)->from.push_back(e);
	}

	// Set or change the potentional of an directed edge
	void CGraphWeiss::setEdge(size_t srcNode, size_t dstNode, const Mat &pot)
	{
		// Assertions
		DGM_ASSERT_MSG(srcNode < m_vpNodes.size(), "The source node index %zu is out of range %zu", srcNode, m_vpNodes.size());
		DGM_ASSERT_MSG(dstNode < m_vpNodes.size(), "The destination node index %zu is out of range %zu", dstNode, m_vpNodes.size());

		bool res = false;
		for(size_t e_t = 0; e_t < m_vpNodes.at(srcNode)->to.size(); e_t++) {
			Edge *edge_to = m_vpNodes.at(srcNode)->to.at(e_t);
			if (edge_to->node2->id == dstNode) {
				if (!edge_to->Pot.empty()) edge_to->Pot.release();
				pot.copyTo(edge_to->Pot);
				res = true;
				break;
			}
		} // e_t

		DGM_ASSERT_MSG(res, "The edge is not set: destination node is not found");
	}

	// Return edge potential matrix
	void CGraphWeiss::getEdge(size_t srcNode, size_t dstNode, Mat &pot) const
	{
		// Assertions
		DGM_ASSERT_MSG(srcNode < m_vpNodes.size(), "The source node index %zu is out of range %zu", srcNode, m_vpNodes.size());
		DGM_ASSERT_MSG(dstNode < m_vpNodes.size(), "The destination node index %zu is out of range %zu", dstNode, m_vpNodes.size());

		bool res = false;
		for(size_t e_t = 0; e_t < m_vpNodes.at(srcNode)->to.size(); e_t++) {
			Edge *edge_to = m_vpNodes.at(srcNode)->to.at(e_t);
			if (edge_to->node2->id == dstNode) {
				if (edge_to->Pot.empty()) break;
				edge_to->Pot.copyTo(pot);
				res = true;
				break;
			}
		} // e_t

		DGM_ASSERT_MSG(res, "The requiered edge is not found");
	}
}
