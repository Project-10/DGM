// (pairwise) Graph class interface;
// Written by Sergey G. Kosov in 2015 for Project X 
#pragma once

#include "IGraphPairwise.h"

namespace DirectGraphicalModels
{
	// =============================== Node Structure ==============================
	/**
	* @brief %Node structure
	* @details Basic info for each node.
	*/
	struct Node {
		size_t		id;		    ///< %Node ID
		Mat			Pot;	    ///< %Node potentials: Mat(size: nStates x 1; type: CV_32FC1)
		byte		sol;
		vec_size_t	to;		    ///< Array of edge ids, pointing to the Child vertices
		vec_size_t	from;	    ///< Array of edge ids, coming from the Parent vertices

		Node(void) = delete;
		Node(size_t _id, const Mat &p = EmptyMat) : id(_id), Pot(p.empty() ? Mat() : p.clone()), sol(0) {}
	};
	using ptr_node_t = std::unique_ptr<Node>;
	using vec_node_t = std::vector<ptr_node_t>;

	// =============================== Edge Structure ==============================
	/**
	* @brief %Edge structure
	* @details Basic item stored in adjacency list.
	*/
	struct Edge {
		size_t	  node1;		///< First (source) node in edge
		size_t	  node2;		///< Second (destination) node in edge
		Mat		  Pot;			///< The edge potentials: Mat(size: nStates x nStates; type: CV_32FC1)
		float	* msg;			///< Message (used in message-passing algorithms): Mat(size: nStates x 1; type: CV_32FC1)
		float	* msg_temp;		///< Temp Message (used in message-passing algorithms): Mat(size: nStates x 1; type: CV_32FC1)
		byte	  group_id;		///< ID of the group, to which the edge belongs

		Edge(void) = delete;
		Edge(size_t n1, size_t n2, byte group = 0, const Mat &p = EmptyMat) : node1(n1), node2(n2), Pot(p.empty() ? Mat() : p.clone()), msg(NULL), msg_temp(NULL), group_id(group) {}

		~Edge(void) {
			if (msg)	  delete msg;
			if (msg_temp) delete msg_temp;
		}

		void msg_swap(void) {
			float *tmp = msg;
			msg = msg_temp;
			msg_temp = tmp;
		}
	};
	using	ptr_edge_t = std::unique_ptr<Edge>;
	using	vec_edge_t = std::vector<ptr_edge_t>;

	// ================================ Graph Class ================================
	/**
	* @brief Pairwise graph class
	* @ingroup moduleGraph
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CGraphPairwise : public IGraphPairwise
	{
		friend class CMessagePassing;
		friend class CInferChain;
		friend class CInferTree;
		friend class CInferLBP;
		friend class CInferViterbi;
		friend class CInferTRW;

        
	public:
		/**
		* @brief Constructor
		* @param nStates the number of States (classes)
		*/
		DllExport CGraphPairwise(byte nStates) : IGraphPairwise(nStates), m_IDx(0) {}
        DllExport virtual ~CGraphPairwise(void) = default;

		// CGraph
		DllExport void		reset(void) override;
		DllExport size_t	addNode		  (const Mat &pot = EmptyMat) override;
		DllExport void		setNode       (size_t node, const Mat &pot) override;
		DllExport void		getNode       (size_t node, Mat &pot) const override;
		DllExport void		getChildNodes (size_t node, vec_size_t &vNodes) const override;
		DllExport void		getParentNodes(size_t node, vec_size_t &vNodes) const override;
		DllExport size_t	getNumNodes(void) const override { return m_vNodes.size(); }
		DllExport size_t	getNumEdges(void) const override { return m_vEdges.size(); } 
		
//     DllExport virtual void      marginalize(const vec_size_t &nodes);
		
		DllExport void		addEdge		(size_t srcNode, size_t dstNode, byte group, const Mat &pot) override;
		DllExport void		setEdge		(size_t srcNode, size_t dstNode, const Mat &pot) override;
		DllExport void		setEdges	(std::optional<byte> group, const Mat& pot) override;
		DllExport void		getEdge		(size_t srcNode, size_t dstNode, Mat &pot) const override;
		DllExport void		setEdgeGroup(size_t srcNode, size_t dstNode, byte group) override;
		DllExport byte		getEdgeGroup(size_t srcNode, size_t dstNode) const override;
		DllExport void		removeEdge	(size_t srcNode, size_t dstNode) override;
		DllExport bool		isEdgeExists(size_t srcNode, size_t dstNode) const override;

#ifdef DEBUG_MODE
		/**
		* @brief Returns the edge container
		* @warning Using this function is not safe. It is added exclusively for the debugging purposes.
		* @return The edge container.
		*/
		DllExport vec_edge_t* getEdgesContainer(void) { return &m_vEdges; }
#endif

	private:
		/**
		* @brief Removes the specified edge
		* @param edge index of the edge
		*/
		DllExport void				removeEdge(size_t edge);


	private:
		size_t		m_IDx;			// = 0;	Primary Key
		vec_node_t	m_vNodes;		// Nodes container
		vec_edge_t	m_vEdges;		// Edges container
	};
}

