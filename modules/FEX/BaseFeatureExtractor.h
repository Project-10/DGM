// Base abstract class for feature extraction algorithms
// Writtem by Sergey G. Kosov in 2015 for Project X
#pragma once

#include "types.h"

namespace DirectGraphicalModels { namespace fex
{
	// ================================ Base Feature Extractor Class ==============================
	/**
	* @ingroup moduleFEX
	* @brief Base abstract class for feature extraction algorithms
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/		
	class CBaseFeatureExtractor
	{
	public:
		/**
		* @brief Constructor.
		* @param img Input image.
		*/
		DllExport CBaseFeatureExtractor(const Mat &img) : m_img(img) {}
		DllExport virtual ~CBaseFeatureExtractor(void) {}

		/**
		* @brief Extracts and returns the required feature.
		* @returns The feature image.
		*/
		DllExport virtual Mat	get(void) const = 0;

		/**
		* @brief Linear 1D mapping.
		* @details This function perform linear mapping of the value \b val from one interval to another: \f$val\in[min; max]\rightarrow res\in[0; 255]\f$, such that:
		* \f{eqnarray*}{ min&\rightarrow&0 \\ max&\rightarrow&255 \f}
		* @param val The value to map. 
		* @param min The lower boundary of the \b val.
		* @param max The higher bounday of the \b val.
		* @returns The mapped value \b res.
		*/
		DllExport static byte	linear_mapper(float val, float min, float max);
		/**
		* @brief Two-linear 1D mapping.
		* @details This function perform linear mapping of the value \b val from one interval to another: \f$val\in[min; max]\rightarrow res\in[0; 255]\f$, such that:
		* \f{eqnarray*}{ min&\rightarrow&0 \\ mid&\rightarrow&midPoint \\ max&\rightarrow&255 \f}
		* For more detail please refer to the \b Figure \b 1.
		* @image html two_linear_mapping.gif "Fig. 1"
		* @param val The value to map. 
		* @param min The lower boundary of the \b val.
		* @param max The higher bounday of the \b val.
		* @param mid The x-coordinate of the intersection point, \f$mid\in(min; max)\f$ (Ref. \b Figure \b 1).
		* @param midPoint The y-coordinate of the intersection point, \f$midPoint\in[0; 255]\f$ (Ref. \b Figure \b 1).
		* @returns The mapped value \b res.
		*/
		DllExport static byte	two_linear_mapper(float val, float min, float max, float mid, byte midPoint);


	protected:
		const Mat	m_img;		///< Container for the image, from which the feature are to be extracted.
	};

} }