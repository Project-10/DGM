// NDVI feature extraction class interface
// Written by Sergey G. Kosov in 2015 for Project X
#pragma once

#include "ILocalFeatureExtractor.h"

namespace DirectGraphicalModels { namespace fex
{
	// ================================ NDVI Class ==============================
	/**
	* @brief NDVI (<a href="http://en.wikipedia.org/wiki/Normalized_Difference_Vegetation_Index">normalized difference vegetation index</a>) feature extraction class.
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/	
	class CNDVI : public ILocalFeatureExtractor
	{
	public:
		/**
		* @brief Constructor.
		* @param img Input image of type \b CV_8UC3.
		*/
		DllExport CNDVI(const Mat &img) : ILocalFeatureExtractor(img) {}
		DllExport virtual ~CNDVI(void) {}

		DllExport virtual Mat	get(void) const { return get(m_img); }
		
		/**
		* @brief Extracts the NDVI feature.
		* @details This function calculates the NDVI from the input image as follows: \f[ NDVI=\frac{NIR-VIS}{NIR+VIS},\f] 
		* where \f$NIR\f$ designates the \a near-infra-red data, and \f$VIS\f$ - \a visible data. 
		* The algorithm assumes that the near-infra-red data is stored in the red image channel and the visible data - in the remaining two channels: 
		* \f{eqnarray*}{NIR&=&img.RED \\ VIS&=&\frac{1}{2}\,img.GREEN+\frac{1}{2}\,img.BLUE.\f} 
		* As \f$NDVI\in[-1; 1]\f$, this function performs two-linear mapping of the NDVI values to the interval \f$[0; 255]\f$, such that:
		* \f{eqnarray*}{-1&\rightarrow&0 \\  0&\rightarrow&midPoint \\  1&\rightarrow&255\f} 
		* For more details on mapping refer to the @ref two_linear_mapper() function.
		* @param img Input image of type \b CV_8UC3, where near-infra-red data is stored in the red channel.
		* @param midPoint Parameter for the two-linear mapping of the feature (Ref. @ref two_linear_mapper()). 
		* > Common values are: 
		* > - 0&nbsp;&nbsp;&nbsp;&nbsp; - cut off the negative NDVI values;
		* > - 127 - linear mapping; 
		* > - 255 - cut off the positive NDVI values.
		* @return The NDVI feature image of type \b CV_8UC1.
		*/
		DllExport static Mat	get(const Mat &img, byte midPoint = 127);
	};
} }