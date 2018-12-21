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

			Node(const Mat& p = EmptyMat) : Pot(p.empty() ? Mat() : p.clone()) {}
			~Node() {
				if (!Pot.empty()) Pot.release();
				size_t n = to.size();
				for (size_t e = 0; e < n; e++)
					delete to.at(e);
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
		

		DllExport virtual void		reset(void) override;
		DllExport virtual size_t	addNode(const Mat &pot = EmptyMat) override;
		DllExport virtual void		setNode(size_t node, const Mat &pot) override;
		DllExport virtual void		getNode(size_t node, Mat &pot) const override;
		DllExport virtual void		getChildNodes(size_t node, vec_size_t &vNodes) const override;
		DllExport virtual void		getParentNodes(size_t node, vec_size_t &vNodes) const override;

		DllExport virtual void		addEdge(size_t srcNode, size_t dstNode, const Mat &pot = EmptyMat) override;
		DllExport virtual void		setEdge(size_t srcNode, size_t dstNode, const Mat &pot) override;
		DllExport virtual void		getEdge(size_t srcNode, size_t dstNode, Mat &pot) const override;

		DllExport virtual size_t	getNumNodes(void) const override { return m_vpNodes.size(); }


	private:
		size_t		m_IDx;			// = 0;	Primary Key
		vec_pNode_t m_vpNodes;		// Nodes container
	};
}

