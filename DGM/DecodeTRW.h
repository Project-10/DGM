// Microsoft Research TRW class interface
// Written by Sergey G. Kosov in 2013 for Project X
#pragma once

#include "decode.h"

namespace DirectGraphicalModels
{
	// ==================== Microsoft TRW Decode Class ==================
	/**
	* @ingroup moduleDecode
	* @brief Microsoft Tree-reweighted decoding class
	* @details This class is based on the <a href="http://research.microsoft.com/en-us/downloads/dad6c31e-2c04-471f-b724-ded18bf70fe3/" target="_blank">Tree-reweighted message passing algorithm for energy minimization</a> v.1.3 
	* (a modification of a max-poduct LBP algorithm), described in the paper <a href="http://pub.ist.ac.at/~vnk/papers/TRW-S-PAMI.pdf" target="_blank">Convergent Tree-reweighted Message Passing for Energy Minimization</a>
	* @note This class supports only undirected arcs with symmetric potentials in the graph.
	* @warning Do not use this class with directed or mixed graphs
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CDecodeTRW : public CDecode
	{
	public:
		/**
		* @brief Constructor
		* @param pGraph The graph
		*/	
		DllExport CDecodeTRW(CGraph *pGraph) : CDecode(pGraph) {};
		DllExport virtual ~CDecodeTRW(void) {};

		/**
		* @brief Aproximate decoding
		* @param nIt Number of iterations
		* @param lossMatrix is not used
		* @return The most probable configuration
		*/
		DllExport virtual vec_byte_t decode(unsigned int nIt = 10, Mat &lossMatrix = Mat()) const;
	};
}

