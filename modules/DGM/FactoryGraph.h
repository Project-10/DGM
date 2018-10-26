// Abstract Factory class for constructing the Graph instances
// Written by Sergey Kosov in 2018 for Project X
#pragma once

#include "Graph.h"
#include "GraphDense.h"
#include "GraphPairwise.h"

#include "Infer.h"
#include "InferDense.h"
#include "MessagePassing.h"
#include "InferLBP.h"
#include "InferTRW.h"
#include "InferViterbi.h"

#include "GraphExt.h"
#include "GraphDenseExt.h"
#include "GraphPairwiseExt.h"


namespace DirectGraphicalModels
{
    // ================================ Graph Factory Class ===============================
    /**
     * @brief Abstract Factory class for constructing Graph objects
     * @ingroup moduleGraph
     * @author Sergey G. Kosov, sergey.kosov@project-10.de
     */
    class CFactoryGraph {
    public:
        /**
         * @param nStates the number of States (classes)
         */
        virtual CGraph& getGraph() = 0;
        /**
         * @param graph The graph
         */
        virtual CInfer& getInfer() = 0;
        /**
         */
        virtual CGraphExt& getGraphExt() = 0;
    };
    
    
    // ================================ Pairwise Graph Factory Class ===============================
    /**
     * @brief Factory class for Pairwise Graphs
     * @ingroup moduleGraph
     * @author Sergey G. Kosov, sergey.kosov@project-10.de
     */
    class CFactoryGraphPairwise : public CFactoryGraph {
        enum class infer { lbp, trw, viterbi };
    public:
        CFactoryGraphPairwise(byte nStates, infer i = infer::lbp)
            : m_graph(nStates)
            , m_ext(m_graph)
        {
            switch (i) {
                case infer::lbp:     m_pInfer = std::make_unique<CInferLBP>(m_graph); break;
                case infer::trw:     m_pInfer = std::make_unique<CInferTRW>(m_graph); break;
                case infer::viterbi: m_pInfer = std::make_unique<CInferViterbi>(m_graph); break;
            }
        }
        virtual CGraph&     getGraph() { return m_graph; }
        virtual CInfer&     getInfer() { return *m_pInfer; }
        virtual CGraphExt&  getGraphExt() { return m_ext; }
        
        
    private:
        CGraphPairwise    m_graph;
        std::unique_ptr<CMessagePassing> m_pInfer;
        CGraphPairwiseExt m_ext;
        
    };
    
    
    // ================================ Dense Graph Factory Class ===============================
    /**
     * @brief Factory class for dense graphs
     * @ingroup moduleGraph
     * @author Sergey G. Kosov, sergey.kosov@project-10.de
     */
    class CFactoryGraphDense : public CFactoryGraph {
    public:
        CFactoryGraphDense(byte nStates)
            : m_graph(nStates)
            , m_infer(m_graph)
            , m_ext(m_graph)
        {}
        virtual CGraph&         getGraph() { return m_graph; }
        virtual CInfer&         getInfer() { return m_infer; }
        virtual CGraphExt&      getGraphExt() { return m_ext; }
        
        
    private:
        CGraphDense    m_graph;
        CInferDense    m_infer;
        CGraphDenseExt m_ext;
    };
}

