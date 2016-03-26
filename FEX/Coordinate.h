// Coordinate feature extraction class interface
// Written by Sergey G. Kosov in 2015 for Project X
#pragma once

#include "basefeatureextractor.h"

namespace DirectGraphicalModels { namespace fex
{
	/**
	* @brief Types of the coordinate feature.
	*/
	enum coordinateType {
		COORDINATE_ORDINATE,	///< Coordinate feature depend on the pixel's ordinate (y-coordinate). 
		COORDINATE_ABSCISS,		///< Coordinate feature depend on the pixel's absciss (x-coordinate). 
		COORDINATE_RADIUS		///< Coordinate feature depend on the pixel's distance to the image center. 
	};
	
	// ================================ Coordinate Class ==============================
	/**
	* @brief Coordinate feature extraction class.
	* @details This class allows extracting features, which depend only on the coordinates of the coresponding image pixels.
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/	
	class CCoordinate :	public CBaseFeatureExtractor
	{
	public:
		/**
		* @brief Constructor.
		* @param img Input image.
		*/
		DllExport CCoordinate(const Mat &img) : CBaseFeatureExtractor(img) {}
		DllExport virtual ~CCoordinate(void) {}

		DllExport virtual Mat	get(void) const {return get(m_img);}

		/**
		* @brief Extracts a coordinate feature.
		* @details This function calculates the coordinate feature of image pixels, based inly on theirs coordinates.
		* @param img Input image.
		* @param type Type of the coordinate feature (Ref. @ref coordinateType).
		* @return The coordinate feature image of type \b CV_8UC1.
		*/
		DllExport static Mat	get(const Mat &img, coordinateType type = COORDINATE_ORDINATE);

	};
} }