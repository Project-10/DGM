// Extended Graph abstract class interface;
// Written by Sergey Kosov in 2018 for Project X
#pragma once

#include "types.h"

namespace DirectGraphicalModels
{
    // ================================ Extended Graph Class ================================
    /**
    * @brief Extended graph abstract class for 2D image classifaction
    * @ingroup moduleGraph
    * @details This class provides wrapper functions for simplifying operations with graphs, when they are used for 2D image classification
    * @author Sergey G. Kosov, sergey.kosov@project-10.de
    */
    class CGraphExt
    {
    public:
        DllExport CGraphExt(void) = default;
        DllExport CGraphExt(const CGraphExt&) = delete;
		DllExport virtual ~CGraphExt(void) = default;
        DllExport const CGraphExt& operator=(const CGraphExt&) = delete;

        /**
        * @brief Builds a 2D graph of size corresponding to the image resolution
		* @details When called multiple times, previouse graph structure is always replaced
        * @param graphSize The size of the graph (image resolution)
        */
        DllExport virtual void buildGraph(Size graphSize) = 0;
        /**
        * @brief Fills an existing 2D graph with potentials or builds a new 2D graph of size corresponding to \b pots.size() with potentials
        * @details
        * If the graph was not build beforehand, or the size of existing graph does not correspond to \b pots.size() this function calls first
        * @code
        * buildGraph(pots.size())
        * @endcode
        * @param pots A block of node potentials: Mat(size: image width x image height; type: CV_32FC(nStates)). It may be obtained by:
        * @code
        * CTrainNode::getNodePotentials()
        * @endcode
        */
        DllExport virtual void setGraph(const Mat& pots) = 0;
        /**
		* @brief Adds default data-independet edge model
		* @param val Value, specifying the smoothness strength 
		* > See concrete class implementation for more details
        * @param weight The weighting parameter
		*/
		DllExport virtual void addDefaultEdgesModel(float val, float weight = 1.0f) = 0;
		/**
		* @brief Adds default contrast-sensitive edge model
		* @param featureVectors Multi-channel matrix, each element of which is a multi-dimensinal point: Mat(type: CV_8UC<nFeatures>)
        * @param val Value, specifying the smoothness strength
		* > See concrete class implementation for more details
        * @param weight The weighting parameter
		*/
		DllExport virtual void addDefaultEdgesModel(const Mat &featureVectors, float val, float weight = 1.0f) = 0;
		/**
        * @brief Adds default contrast-sensitive edge model
        * @param featureVectors Vector of size \a nFeatures, each element of which is a single feature - image: Mat(type: CV_8UC1)
        * @param val Value, specifying the smoothness strength
		* > See concrete class implementation for more details
        * @param weight The weighting parameter
        */
        DllExport virtual void addDefaultEdgesModel(const vec_mat_t &featureVectors, float val, float weight = 1.0f) = 0;
        /**
        * @brief Returns the size of the graph
        * @return The size of the Graph
        */
        DllExport virtual Size getSize() const = 0;
    };
}