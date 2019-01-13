// (pairwise) Graph class interface;
// Written by Sergey G. Kosov in 2011 - 2014 for Project X (based on M. A. Weiss recommendations) 
#pragma once

#include "IGraphPairwise.h"


namespace DirectGraphicalModels
{
	// ================================ Graph Class ================================
	/**
	* @brief Pairwise graph class
	* @details Implementation is based on M. A. Weiss recommendations
	* @warning This class is added for academic reasons. Do not use with Inference / Decoding classes
	* @ingroup moduleGraph
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CGraphWeiss : public IGraphPairwise
	{
	public:
		struct Node;
		
		// =============================== Edge Structure ==============================
		/**
		* @brief %Edge structure
		* @details Basic item stored in adjacency list.
		*/
		struct Edge {
			Node	* node1;		///< First node in edge
			Node	* node2;		///< Second node in edge
			Mat		  Pot;			///< The edge potentials: Mat(size: nStates x nStates; type: CV_32FC1)
			float	* msg;			///< Message (for the loopyBP algorithm: class CDecodeLPB): Mat(size: nStates x 1; type: CV_32FC1)
			float	* msg_temp;		///< Temp Message (for the loopyBP algorithm: class CDecodeLPB): Mat(size: nStates x 1; type: CV_32FC1)
			byte	  group_id;		///< ID of the group, to which the edge belongs
			
			Edge(void) = delete;
			Edge(Node* n1, Node* n2, byte group = 0, const Mat &p = EmptyMat) : node1(n1), node2(n2), Pot(p.empty() ? Mat() : p.clone()), msg(NULL), msg_temp(NULL), group_id(group) {}
			
			~Edge(void) {
				if (msg)		delete[] msg;		msg = NULL;
				if (msg_temp)	delete[] msg_temp;	msg_temp = NULL;
			}
			
			void msg_swap(void) {
				float *tmp = msg;
				msg = msg_temp;
				msg_temp = tmp;
			}
		};

		/// @todo Use smart pointers here
		using vec_pEdge_t = std::vector<Edge *>;

		// =============================== Node Structure ==============================
		/**
		* @brief %Node structure
		* @details Basic info for each node.
		*/
		struct Node {
			size_t		id;		///< Node ID
			Mat			Pot;	///< Node potentials: Mat(size: nStates x 1; type: CV_32FC1)
			vec_pEdge_t	to;		///< Child vertices (and potentials)
			vec_pEdge_t	from;	///< Parent vertices (and potentials)

			Node(void) = delete;
			Node(size_t _id, const Mat& p = EmptyMat) : id(_id), Pot(p.empty() ? Mat() : p.clone()) {}
			~Node() {
				for (Edge* e : to)
					delete e;
			}
		};

		/// @todo Use smart pointers here
		using vec_pNode_t = std::vector<Node *>;
	
	
	public:
		/**
		* @brief Constructor
		* @param nStates the number of States (classes)
		*/	
		DllExport CGraphWeiss(byte nStates);
		DllExport virtual ~CGraphWeiss(void);
		

		DllExport void		reset(void) override;
		DllExport size_t	addNode(const Mat &pot = EmptyMat) override;
		DllExport void		setNode(size_t node, const Mat &pot) override;
		DllExport void		getNode(size_t node, Mat &pot) const override;
		DllExport void		getChildNodes(size_t node, vec_size_t &vNodes) const override;
		DllExport void		getParentNodes(size_t node, vec_size_t &vNodes) const override;
		DllExport size_t	getNumNodes(void) const override { return m_vpNodes.size(); }
		DllExport size_t	getNumEdges(void) const override;

		DllExport void		addEdge		(size_t srcNode, size_t dstNode, byte group, const Mat &pot) override;
		DllExport void		setEdge		(size_t srcNode, size_t dstNode, const Mat &pot) override;
		DllExport void		setEdges	(std::optional<byte> group, const Mat& pot) override;
		DllExport void		getEdge		(size_t srcNode, size_t dstNode, Mat &pot) const override;
		DllExport void		setEdgeGroup(size_t srcNode, size_t dstNode, byte group) override;
		DllExport byte		getEdgeGroup(size_t srcNode, size_t dstNode) const override;
		DllExport void		removeEdge	(size_t srcNode, size_t dstNode) override;
		DllExport bool		isEdgeExists(size_t srcNode, size_t dstNode) const override;


	protected:
		/**
		* @brief Finds and returns the %Edge defined by two nodes
		* @param srcNode index of the source node
		* @param dstNode index of the destination node
		* @return Pointer to the %Edge if found, NULL otherwise
		*/
		DllExport  Edge*			findEdge(size_t srcNode, size_t dstNode) const;

	private:
		size_t		m_IDx;			// = 0;	Primary Key
		vec_pNode_t m_vpNodes;		// Nodes container
	};
}

