// Gradient feature extraction class interface
// Written by Sergey G. Kosov in 2015 for Project X
#pragma once

#include "BaseFeatureExtractor.h"

namespace DirectGraphicalModels { namespace fex
{
	const static float GRADIENT_MAX_VALUE = 255 * sqrtf(2);
	// ================================ Gradient Class ==============================
	/**
	* @brief Gradient feature extraction class
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/		
	class CGradient : public CBaseFeatureExtractor
	{
	friend class CHOG;
	public:
		/**
		* @brief Constructor.
		* @param img Input image of type \b CV_8UC1 or \b CV_8UC3.
		*/
		DllExport CGradient(const Mat &img) : CBaseFeatureExtractor(img) {}
		DllExport virtual ~CGradient(void) {}

		DllExport virtual Mat get(void) const {return get(m_img);}

		/**
		* @brief Extracts the gradient feature.
		* @details This function calculates the magnitude of gradient of the input image as follows: \f[gradient=\sqrt{\left(\frac{d\,img}{dx}\right)^2+\left(\frac{d\,img}{dy}\right)^2},\f]
		* where \f$\frac{d\,img}{dx}\f$ and \f$\frac{d\,img}{dy}\f$ are the first \a x and \a y central derivatives of the input image.\n
		* As \f$gradient\in[0; 255\,\sqrt{2}]\f$, this function performs two-linear mapping of the gradient values to the interval \f$[0; 255]\f$, such that:
		* \f{eqnarray*}{0&\rightarrow&0 \\  mid&\rightarrow&255 \\  255\,\sqrt{2}&\rightarrow&255\f} 
		* For more details on mapping refer to the @ref two_linear_mapper() function.
		* @param img Input image of type \b CV_8UC1 or \b CV_8UC3.
		* @param mid Parameter for the two-linear mapping of the feature: \f$mid\in(0;255\sqrt{2}]\f$. (Ref. @ref two_linear_mapper()). 
		* @return The gradient feature image of type \b CV_8UC1.
		*/		
		DllExport static Mat get(const Mat &img, float mid = GRADIENT_MAX_VALUE);

	protected:
		static Mat getDerivativeX(const Mat &img);
		static Mat getDerivativeY(const Mat &img);
	};
} }

