#pragma once
#include "MessagePassing.h"

namespace DirectGraphicalModels
{

	class CInferTRW_S : public CMessagePassing
	{
	private:
		struct NODE;
		struct EDGE;


	public:
		DllExport CInferTRW_S(CGraph *pGraph) : CMessagePassing(pGraph), m_nodeFirst(NULL), m_nodeLast(NULL), m_nodeNum(0) {}
		DllExport virtual ~CInferTRW_S(void) {}

		DllExport virtual void infer(unsigned int nIt = 1);

	
	protected:
		DllExport virtual void calculateMessages(unsigned int nIt);
		void calculateMessage(EDGE *edge, float *temp, float *data);


	private:
		NODE	* AddNode(float *data);
		void	  AddEdge(NODE *i, NODE *j, float *data);

		NODE	* m_nodeFirst;
		NODE	* m_nodeLast;
		int		  m_nodeNum;

		struct NODE {
			int			  m_id; 			///< unique integer in [0,m_nodeNum-1)
			EDGE		* m_firstForward; 	///< first edge going to nodes with greater m_ordering
			EDGE		* m_firstBackward; 	///< first edge going to nodes with smaller m_ordering
			NODE		* m_prev; 			///< previous and next
			NODE		* m_next; 			///< nodes according to m_ordering
			int			  m_solution; 		///< integer in [0,m_D.m_K)
			float		* m_D;				///< node potential
		};

		struct EDGE {
			EDGE		* m_nextForward; 	///< next forward edge with the same tail
			EDGE		* m_nextBackward; 	///< next backward edge with the same head
			NODE		* m_tail;
			NODE		* m_head;
			float		* m_D;				///< edge potential
			float		* m_msg;			///< message
		};
	};

}