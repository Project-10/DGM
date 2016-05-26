// (pairwise) Graph class interface;
// Written by Sergey G. Kosov in 2011 - 2014 for Project X (based on M. A. Weiss recommendations) 
#pragma once

#include "BaseGraph.h"

namespace DirectGraphicalModels
{

	// =============================== Node Structure ==============================
	/**
	@brief %Node structure
	@details Basic info for each node.
	*/
	struct Node {
		size_t		id;		///< %Node ID
		Mat			Pot;	///< %Node potentials: Mat(size: nStates x 1; type: CV_32FC1)
		byte		sol;
		vec_size_t	to;		///< Array of edge ids, pointing to the Child vertices
		vec_size_t	from;	///< Array of edge ids, coming from the Parent vertices


		Node(void) : Pot(Mat()), sol(0) {}

		Node(size_t _id) : id(_id), Pot(Mat()), sol(0) {}

		Node(size_t _id, const Mat &p) : id(_id), sol(0) { p.copyTo(Pot); }
	};


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
		bool	  suspend;		///< Flag, indicating weather the message calculation must be postponed (used in message-passing algorithms)

		Edge(void) : Pot(Mat()), msg(NULL), msg_temp(NULL), suspend(false) {}

		Edge(size_t n1, size_t n2) : node1(n1), node2(n2), Pot(Mat()), msg(NULL), msg_temp(NULL), suspend(false) {}

		Edge(size_t n1, size_t n2, const Mat &p) : node1(n1), node2(n2), msg(NULL), msg_temp(NULL), suspend(false) { p.copyTo(Pot); }

		void msg_swap(void) {
			float *tmp = msg;
			msg = msg_temp;
			msg_temp = tmp;
		}
	};


	// ================================ Graph Class ================================
	/**
	* @brief Pairwise graph class
	* @ingroup moduleGraph
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CGraph : public CBaseGraph
	{
		friend class CMessagePassing;

		friend class CInferExact;
		friend class CInferChain;
		friend class CInferTree;
		friend class CInferLBP;
		friend class CInferViterbi;
		friend class CInferTRW;

		friend class CDecode;


	public:
		/**
		* @brief Constructor
		* @param nStates the number of States (classes)
		*/
		DllExport CGraph(byte nStates) : CBaseGraph(nStates), m_IDx(0) {}
		DllExport virtual ~CGraph(void) {}

		DllExport virtual void		reset(void);
		DllExport virtual size_t	addNode(void);
		DllExport virtual size_t	addNode(const Mat &pot);
		DllExport virtual void		setNode       (size_t node, const Mat &pot);
		DllExport virtual void		getNode       (size_t node, Mat &pot) const;
		DllExport virtual void		getChildNodes (size_t node, vec_size_t &vNodes) const;
		DllExport virtual void		getParentNodes(size_t node, vec_size_t &vNodes) const;
		
		DllExport virtual void		addEdge   (size_t srcNode, size_t dstNode);
		DllExport virtual void		addEdge   (size_t srcNode, size_t dstNode, const Mat &pot);
		DllExport virtual void		setEdge   (size_t srcNode, size_t dstNode, const Mat &pot);
		DllExport virtual void		getEdge   (size_t srcNode, size_t dstNode, Mat &pot) const;
		DllExport virtual void		removeEdge(size_t srcNode, size_t dstNode);
		
		DllExport virtual void		addArc    (size_t Node1, size_t Node2);
		DllExport virtual void		addArc    (size_t Node1, size_t Node2, const Mat &pot);
		DllExport virtual void		setArc    (size_t Node1, size_t Node2, const Mat &pot);
		DllExport virtual void		removeArc (size_t Node1, size_t Node2);
		
		DllExport virtual size_t	getNumNodes(void) const { return m_vNodes.size(); }
		DllExport virtual size_t	getNumEdges(void) const { return m_vEdges.size(); }


	protected:
		/**
		* @brief Removes the specified edge
		* @param edge index of the edge
		*/
		DllExport virtual void		removeEdge(size_t edge);

	private:
		size_t		m_IDx;			// = 0;	Primary Key
		vec_node_t	m_vNodes;		// Nodes container
		vec_edge_t	m_vEdges;		// Edges container

	};
}

