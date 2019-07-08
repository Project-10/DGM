// Tree-reweighted inference class interface
// Written by Sergey G. Kosov in 2016 for Project X
#pragma once

#include "MessagePassing.h"

namespace DirectGraphicalModels
{
	struct Edge;
	
	// ==================== Microsoft TRW Decode Class ==================
	/**
	* @ingroup moduleDecode
	* @brief Tree-reweighted inference class
	* @details This class is based on the Tree-reweighted message passing algorithm (a modification of a max-poduct LBP algorithm), 
	* described in the paper <a href="http://pub.ist.ac.at/~vnk/papers/TRW-S-PAMI.pdf" target="_blank">Convergent Tree-reweighted Message Passing for Energy Minimization</a>
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CInferTRW : public CMessagePassing
	{
	public:
		/**
		* @brief Constructor
		* @param graph The graph
		*/
		DllExport CInferTRW(CGraphPairwise &graph) : CMessagePassing(graph) {}
		DllExport virtual ~CInferTRW(void) = default;

		DllExport virtual void infer(unsigned int nIt = 1);


	protected:
		DllExport virtual void	calculateMessages(unsigned int nIt);
		void					calculateMessage(float* msg, Edge& edge, float* temp, float* data);
	};
}
