// Histogram Marker Class 
// Written by Sergey Kosov in 2015 for Project X
#pragma once

#include "marker.h"

namespace DirectGraphicalModels
{
	class CTrainNode;
	// ================================ Histogram Marker Class ================================
	/**
	* @ingroup moduleVis
	* @brief Histogram Marker class
	* @details This class allows to visualize the feature densitiy distributions (feature histograms), used in the naive bayes random model (Ref. @ref CTrainNodeNaiveBayes).
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/	
	class CMarkerHistogram : public CMarker
	{
	public:
/**
@brief Constructor with default palette
@param pNodeTrainer Pointer to the CTrainNode class.
@param palette One of the default palletes (Ref. @ref default_pallete).
@param ppFeatureNames Optional list of feature names. 
For optimal performance, each feature name should have maximal 17 symbols.
*/
		DllExport CMarkerHistogram(CTrainNode *pNodeTrainer, default_pallete palette = DEF_PALETTE_12, char *ppFeatureNames[] = NULL) : CMarker(palette), m_pNodeTrainer(pNodeTrainer), m_ppFeatureNames(ppFeatureNames) {}
		/**
		* @brief Constructor with custom palette
		* @param pNodeTrainer Pointer to the CTrainNode class.
		* @param vPalette Custom palette. It is represented as a std::vector of the custom entries of type: @code std::make_pair(CV_RGB(r, g, b), "class name"). @endcode
		* For optimal performance, class name should have maximal 10 symbols. 
		* @param ppFeatureNames Optional list of feature names.
		* For optimal performance, each feature name should have maximal 17 symbols.
		*/
		DllExport CMarkerHistogram(CTrainNode *pNodeTrainer, const vec_nColor_t &vPalette, char *ppFeatureNames[] = NULL) : CMarker(vPalette), m_pNodeTrainer(pNodeTrainer), m_ppFeatureNames(ppFeatureNames) {}
		DllExport virtual ~CMarkerHistogram(void) {}

/**
@brief Draws the figure with a visualization of feature densitiy distributions.
@details This function also allows for active user interaction: after the figure is drawn, user may click upon it, and the color of the clicked point is than transmitted to the fuction as argument. 
The function will redraw the fugure, depending on the input color.
@param color Color of a pixel from the resulting histogram. This optional parameter is used for active user interaction.  
@return Figure with visualized histograms of the feature distributions.
*/
		DllExport Mat	drawHistogram(Scalar color = CV_RGB(0,0,0)) const;



	#ifdef DEBUG_MODE	// --- Debugging ---
/**
@brief Visualises the 2-dimensional node potentials histogram.
@return Figure with visualized histogram. 
@warning Used only for test purposes. Capable to visualize only 2-dimensional feature space.
*/
		DllExport Mat	TEST_drawHistogram(CTrainNode *pTrain) const;
	#endif				// --- --------- ---

	
	protected:
/**
@brief Retrieves a chosen by an user state, from the color.
@param color Color of a pixel from the histogram figure. 
@return Active state (class) if \a color matches the palette, -1 othervise.
*/
		int				getActiveState(Scalar color) const;
/**
@brief Draws a single feature histogram.
@param f Feature.
@param activeState Desired state (class).
@return Figure with visualized feature histogram. Mat(100, 256, CV_8UC1).
*/
		Mat				drawFeatureHistogram(byte f, int activeState = -1) const;
/**
@brief Draws a legend to the main figure.
@param maxHeight The maximal height of the legend figure.
@param activeState Desired state (class).
@return Figure with visualized legend to the main figure.
*/
		Mat				drawLegend(int maxHeight, int activeState = -1) const;

	protected:
		CTrainNode			*  m_pNodeTrainer;		///< Pointer to the  CTrainNode class


	private:
		static const CvSize    margin;
		static const byte	   bkgIntencity;
		static const double	   frgWeight;

		char				** m_ppFeatureNames;
	};

}