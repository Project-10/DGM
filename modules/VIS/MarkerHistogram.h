// Histogram Marker Class 
// Written by Sergey Kosov in 2015 for Project X
#pragma once

#include "Marker.h"

namespace DirectGraphicalModels { 
	class CTrainNode;
	namespace vis
{
	
	// ================================ Histogram Marker Class ================================
	/**
	* @ingroup moduleVIS
	* @brief Histogram Marker class
	* @details This class allows to visualize the feature densitiy distributions (feature histograms), used in the naive bayes random model (Ref. @ref CTrainNodeBayes).
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/	
	class CMarkerHistogram : public CMarker
	{
	public:
		/**
		* @brief Constructor with default palette
		* @param nodeTrainer The node trainer.
		* @param palette One of the default palletes (Ref. @ref default_pallete).
		* @param vFeatureNames Optional list of feature names. 
		* For optimal performance, each feature name should have maximal 17 symbols.
		*/
		DllExport CMarkerHistogram(const CTrainNode &nodeTrainer, default_pallete palette = DEF_PALETTE_12, vec_string_t vFeatureNames = vec_string_t())
            : CMarker(palette), m_nodeTrainer(nodeTrainer), m_vFeatureNames(vFeatureNames) {}
		/**
		* @brief Constructor with custom palette
		* @param nodeTrainer Pointer to the CTrainNode class.
		* @param vPalette Custom palette. It is represented as a std::vector of the custom entries of type: @code std::make_pair(CV_RGB(r, g, b), "class name"). @endcode
		* For optimal performance, class name should have maximal 10 symbols. 
		* @param vFeatureNames Optional list of feature names.
		* For optimal performance, each feature name should have maximal 17 symbols.
		*/
		DllExport CMarkerHistogram(const CTrainNode &nodeTrainer, const vec_nColor_t &vPalette, vec_string_t vFeatureNames = vec_string_t())
			: CMarker(vPalette), m_nodeTrainer(nodeTrainer), m_vFeatureNames(vFeatureNames) {}
		DllExport virtual ~CMarkerHistogram(void) {}

		/**
		* @brief Draws a figure with the visualization of feature densitiy distributions.
		* @return Figure with visualized histograms of the feature distributions.
		*/
		DllExport Mat	drawHistogram(void) const { return drawHistogram(CV_RGB(0, 0, 0)); }
		/**
		* @brief Draws a figure with the visualization of 2-dimensional node potentials histogram.
		* @note Used for test purposes. Capable to visualize only 2-dimensional feature space.
		* @return Figure with visualized histogram.
		*/
		DllExport Mat	drawHistogram2D(void) const { return drawHistogram2D(CV_RGB(0, 0, 0)); }
		/**
		* @brief Draws a figure with the visualization of 2-dimensional classification map.
		* @details This function calls the underlying node trainer to classify the area of 256 x 256 pixels,
		* where every pixel (x, y) represents a 2-dimensional feature. The outcome visualization
		* should illustrate how the underlying node trainer represents the feature distribution,
		* which may be visualized with the drawHistogram2D() function
		* @note Used for test purposes. Capable to visualize only 2-dimensional feature space.
		* @param Z The value of partition function for calling the CTrainNode::getNodePotentials() function.
		* @return Figure with visualized classification map.
		*/
		DllExport Mat	drawClassificationMap2D(float Z) const;
		/**
		* @brief Visualizes the feature densitiy distributions in a separate window with user interaction.
		* @details This function creates an OpenCV window with the visualized histograms.
		* Click on the color box for specific state (class) visualuzation.
		*/
		DllExport void	showHistogram(void);
		/**
		* @brief Closes the histogram window
		*/
		DllExport void	close(void) const;


	private:
		/**
		* @brief Draws the figure with a visualization of feature densitiy distributions.
		* @details This function also allows for active user interaction: after the figure is drawn, user may click upon it, and the color of the clicked point is than transmitted to the fuction as argument.
		* The function will redraw the fugure, depending on the input color.
		* @param color Color of a pixel from the resulting histogram. This optional parameter is used for active user interaction.
		* @return Figure with visualized histograms of the feature distributions.
		*/
		DllExport Mat	drawHistogram(Scalar color) const;
		/**
		* @brief Draws the figure with the visualization of 2-dimensional node potentials histogram.
		* @note Used for test purposes. Capable to visualize only 2-dimensional feature space.
		* @param color Color of a pixel from the resulting histogram. This optional parameter is used for active user interaction.
		* @return Figure with visualized classification map.
		*/
		DllExport Mat	drawHistogram2D(Scalar color) const;
		/**
		* @brief Retrieves a chosen by an user state, from the color.
		* @param color Color of a pixel from the histogram figure. 
		* @return Active state (class) if \a color matches the palette, -1 othervise.
		*/
		int				getActiveState(Scalar color) const;
		/**
		* @brief Draws a single feature histogram.
		* @param f Feature.
		* @param activeState Desired state (class).
		* @return Figure with visualized feature histogram: Mat(100, 256, CV_8UC3).
		*/
		Mat				drawFeatureHistogram(word f, int activeState = -1) const;
		/**
		* @brief Draws a 2-dimensional feature histogram.
		* @param f Feature.
		* @param activeState Desired state (class).
		* @return Figure with visualized 2-dimensional feature histogram.
		*/
		Mat				drawFeatureHistogram2D(word f, int activeState = -1) const;
		/**
		* @brief Draws a legend to the main figure.
		* @param maxHeight The maximal height of the legend figure.
		* @param activeState Desired state (class).
		* @return Figure with visualized legend to the main figure.
		*/
		Mat				drawLegend(int maxHeight, int activeState = -1) const;
	

	protected:
		const CTrainNode			      & m_nodeTrainer;		///< The node trainer


	private:
		static const cv::Size			margin;
		static const byte			bkgIntencity;
		static const double			frgWeight;
		static const std::string	wndName;

		vec_string_t				m_vFeatureNames;
	};

} }
