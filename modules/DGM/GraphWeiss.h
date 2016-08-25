// (pairwise) Graph class interface;
// Written by Sergey G. Kosov in 2011 - 2014 for Project X (based on M. A. Weiss recommendations) 
#pragma once

#include "BaseGraph.h"


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
	class CGraphWeiss : public CBaseGraph
	{
	friend class CInferLBP;
	friend class CInferViterbi;

	friend class CDecodeExact;
	friend class CDecodeTRW;
	
	
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
			float	* msg;			///< Message (for the loopyBP algorithm: class CDecodeLPB): Mat(size: nStates x 1; type: CV_32FC1)
			float	* msg_temp;		///< Temp Message (for the loopyBP algorithm: class CDecodeLPB): Mat(size: nStates x 1; type: CV_32FC1)
			Mat		  Pot;			///< The edge potentials: Mat(size: nStates x nStates; type: CV_32FC1)

			Edge() : node1(NULL), node2(NULL), msg(NULL), msg_temp(NULL), Pot(Mat()) {}
			Edge(Node *n1, Node *n2, Mat p) : node1(n1), node2(n2), msg(NULL), msg_temp(NULL) { p.copyTo(Pot); }
			~Edge() {
				if (!Pot.empty()) Pot.release();
				if (msg != NULL) delete[] msg; msg = NULL;
				if (msg_temp != NULL) delete[] msg_temp; msg_temp = NULL;
			}
			void msg_swap(void) {
				float *tmp = msg;
				msg = msg_temp;
				msg_temp = tmp;
			}
		};

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

			Node() : Pot(Mat()) {}
			~Node() {
				if (!Pot.empty()) Pot.release();
				size_t n = to.size();
				for (size_t e = 0; e < n; e++)
					delete to.at(e);
			}
		};

		using vec_pNode_t = std::vector<Node *>;
	
	
	public:
		/**
		* @brief Constructor
		* @param nStates the number of States (classes)
		*/	
		DllExport CGraphWeiss(byte nStates);
		DllExport virtual ~CGraphWeiss(void);
		

		DllExport void		reset(void);
		DllExport size_t	addNode(void);			
		DllExport size_t	addNode(const Mat &pot);
		DllExport void		setNode(size_t node, const Mat &pot);
		DllExport void		getNode(size_t node, Mat &pot) const;
		DllExport void		getChildNodes(size_t node, vec_size_t &vNodes) const;
		DllExport void		getParentNodes(size_t node, vec_size_t &vNodes) const;

		DllExport void		addEdge(size_t srcNode, size_t dstNode);
		DllExport void		addEdge(size_t srcNode, size_t dstNode, const Mat &pot);
		DllExport void		setEdge(size_t srcNode, size_t dstNode, const Mat &pot);
		DllExport void		getEdge(size_t srcNode, size_t dstNode, Mat &pot) const;

		DllExport void		addArk(size_t Node1, size_t Node2);
		DllExport void		addArk(size_t Node1, size_t Node2, const Mat &pot);
		DllExport void		setArk(size_t Node1, size_t Node2, const Mat &pot);

		DllExport size_t	getNumNodes(void) const { return m_vpNodes.size(); }


	private:
		size_t		m_IDx;			// = 0;	Primary Key
		byte		m_nStates;		// The number of states (classes)
		vec_pNode_t m_vpNodes;		// Nodes container
	};
}

