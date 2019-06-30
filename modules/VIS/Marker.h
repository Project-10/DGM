// Marker Class 
// Written by Sergey Kosov in 2008 - 2010 for MPII
// Adopted by Sergey Kosov in 2012 for Project X
// Modified by Sergey Kosov in 2015 for Project X
#pragma once

#include "types.h"

namespace DirectGraphicalModels { namespace vis
{
	// Default paletes
	const static vec_scalar_t colors24 =  
	{ 
		CV_RGB(255, 0,   0),		// Red
		CV_RGB(255, 128, 0),		// Orange
		CV_RGB(255, 255, 0),		// Yellow
		CV_RGB(128, 255, 0),		// Chartreuse
		CV_RGB(0,   255, 0),		// Green
		CV_RGB(0,   255, 128),		// Spring green
		CV_RGB(0,   255, 255),		// Cyan
		CV_RGB(0,   128, 255),		// 
		CV_RGB(0,   0,   255),		// Blue
		CV_RGB(128, 0,   255),		// 
		CV_RGB(255, 0,   255),		// Purple
		CV_RGB(255, 0,   128),		// 

		CV_RGB(128, 0,   0),		// Dark Red
		CV_RGB(128, 64,  0),		// Dark Orange
		CV_RGB(128, 128, 0),		// Dark Yellow
		CV_RGB(64,  128, 0),		// Dark Chartreuse
		CV_RGB(0,   128, 0),		// Dark Green
		CV_RGB(0,   128, 64),		// Dark Spring green
		CV_RGB(0,   128, 128),		// Dark Cyan
		CV_RGB(0,   64,  128),		// 
		CV_RGB(0,   0,   128),		// Dark Blue
		CV_RGB(64,  0,   128),		// 
		CV_RGB(128, 0,   128),		// Dark Purple
		CV_RGB(128, 0,   64)		// 
	}; 

	///@brief Default palettes.
	enum default_pallete {
		DEF_PALETTE_3,		///< Default Pallete with 3 colors
		DEF_PALETTE_3_INV,	///< Default Pallete with 3 colors
		DEF_PALETTE_6,		///< Default Pallete with 6 colors
		DEF_PALETTE_6_INV,	///< Default Pallete with 6 colors
		DEF_PALETTE_12,		///< Default Pallete with 12 colors
		DEF_PALETTE_12_INV,	///< Default Pallete with 12 colors
		DEF_PALETTE_24,		///< Default Pallete with 24 colors
		DEF_PALETTE_24_INV,	///< Default Pallete with 24 colors
		DEF_PALETTE_24_M,	///< Default Pallete with 24 colors
		DEF_PALETTE_36,		///< Default Pallete with 36 colors
		DEF_PALETTE_36_INV,	///< Default Pallete with 36 colors
		DEF_PALETTE_72,		///< Default Pallete with 72 colors
		DEF_PALETTE_72_INV	///< Default Pallete with 72 colors
	};
	/**
	* @brief Visualization flags.
	* @details Used in CMarker::markClasses() and CMarker::markPotentials() functions.
	*/
	enum mark_flags {
		MARK_GRID		= 1,	///< Visualizes only the odd pixels
		MARK_OVER		= 2,	///< Blends the base image
		MARK_NO_ZERO	= 4,	///< The class with index 0 will not be visualized
		MARK_BW			= 8,	///< Mark in "black and white" palette
		MARK_PERCLASS	= 16,	///< Mark per-class accuracies in the confusion matrix
		MARK_PERCENT	= 128	///< Adds percent symbol to the output text values
	};
	

	// ================================ Marker Class ================================
	/**
	* @ingroup moduleVIS
	* @brief Marker class
	* @details This class allows to visualize the results of graphical models decoding
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CMarker
	{
		enum textProp
		{
			TP_CENTER	= 0,
			TP_LEFT		= 1,
			TP_RIGHT	= 2,
			TP_TOP		= 4,
			TP_BOTTOM	= 8,

			TP_PERCENT	= 128		// This flag may be mixed with the <mark_flags>
		};
	
	public:
		/**
		* @brief Constructor with a default palette
		* @param palette One of the default palletes (Ref. @ref default_pallete).
		*/
		DllExport CMarker(default_pallete palette = DEF_PALETTE_12);
		/**
		* @brief Constructor with a custom palette
		* @param vPalette Custom palette. It is represented as a std::vector of the custom entries of type: @code std::make_pair(CV_RGB(r, g, b), "class name"). @endcode
		* For optimal performance, class name should have maximal 10 symbols. 
		*/
		DllExport CMarker(const vec_nColor_t &vPalette);
		DllExport virtual ~CMarker(void) = default;
	

		/**
		* @brief Visualizes the classes.
		* @details Draws the \a classes image on \a base image. 
		* @param[in,out] base Base image on which the classes will be mapped. Image of type: CV_8UC3. May be empty.
		* @param[in] classes Class map image. Image of type: CV_8UC1.
		* @param[in] flag Mapping flag (Ref. @ref mark_flags).
		*/	
		DllExport void			markClasses(Mat &base, const Mat &classes, byte flag = 0) const;				// Does nothing on error
		/**
		* @brief Visualizes the potentials
		* @details Draws <node / edge / triplet> [potential / prior] <vector / matrix / voxel>
		* > This function is also suit for the confusion matrix visualization, but function drawConfusionMatrix() is more preferable for such task
		* @param potential %Node, edge or triplet potential or prior. One, two or tree dimensional Mat of type: CV_32FC1
		* @param flag Mapping flag (Ref. @ref mark_flags)
		* @returns Figure with visualized potential or prior
		* @note This function curently does not support triplet [potential / prior] voxels
		*/
		DllExport Mat			drawPotentials(const Mat &potential, byte flag = 0) const;
		/**
		* @brief Visualizes a confusion matrix
		* @details This function visualizes a confusion matrix, where gthe values are given in percents ofthe overall number of estimated samples. 
		* The function additionally computes and visualizes the recall and precision values for each class. If the flag \b MARK_PERCLASS is set,
		* the classification rate per each class is visualized, and the recall and precision values are not computed.
		* @param confusionMat Confusion matrix: Mat of type: CV_32FC1
		* @param flag Mapping flag (Ref. @ref mark_flags)
		* @returns Figure with visualized confusion matrix
		*/
		DllExport Mat			drawConfusionMatrix(const Mat &confusionMat, byte flag = 0) const;
	
	
	protected:
		vec_nColor_t			m_vPalette;						///< Pointer to the container with the palette


	private:
		Mat	drawVector(const Mat &potential, byte flag) const;
		Mat	drawMatrix(const Mat &potential, byte flag) const;
		Mat	drawVoxel (const Mat &potential, byte flag) const;	// WARNING: not implemented

		template<typename T> 
		void drawSquare(Mat &img, byte x, byte y, const Scalar &color, T val, double fontScale = 1.0, byte textProp = 0) const;
		void drawRectangle(Mat &img, Point pt1, Point pt2, const Scalar &color, float val, double fontScale = 1.0, byte textProp = TP_CENTER) const;
		void drawRectangle(Mat &img, Point pt1, Point pt2, const Scalar &color, const std::string &str = std::string(), double fontScale = 1.0, byte textProp = TP_CENTER) const;


	private:
		static const byte	bkgIntencity;
		static const byte	frgIntensity;
		static const int	ds;								// size of the cell
			

	private:
		// Copy semantics are disabled
		CMarker(const CMarker&) = delete;
		const CMarker& operator=(const CMarker&) = delete;
	};


	/**
	* @ingroup moduleVIS
	* @brief Generates a vector with colors according to a default palette \b palette
	* @param palette One of the default palletes (Ref. @ref default_pallete)
	* @returns A vector with colors according to a default palette \b palette
	*/
	DllExport vec_scalar_t generateDefaultPalette(default_pallete palette = DEF_PALETTE_12);
	/**
	* @ingroup moduleVIS
	* @brief Visualizes a sparse coding dictionary
	* @details This function visualizes a dictionary, that is returned by fex::CSparseDictionary::getDictionary() fucntion
	* @param  dictionary Dictionary: Mat of type: CV_64FC1
	* @param m The magnifier koefficients for scaling the dictionary values
	* @returns Figure with visualized dictionary
	*/
	DllExport Mat drawDictionary(const Mat &dictionary, double m = 1);

} }

