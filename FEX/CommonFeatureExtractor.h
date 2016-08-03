// A common feature extraction class, which unites other feature extraction algorithms
// Writtem by Sergey G. Kosov in 2015 for Project X
#pragma once

#include "BaseFeatureExtractor.h"
#include "Coordinate.h"
#include "Intensity.h"
#include "HSV.h"
#include "Gradient.h"
#include "NDVI.h"
#include "Distance.h"
#include "HOG.h"
#include "Variance.h"
#include "Scale.h"
#include "SparseCoding.h"
#include "macroses.h"

namespace DirectGraphicalModels { namespace fex
{
	/// @brief Channels in the BGR color space.
	enum ChannelsRGB {
		CH_BLUE,			///< Blue channel
		CH_GREEN,			///< Green channel
		CH_RED				///< Red channel
	};

	/// @brief Channels in the HSV color space.
	enum ChannelsHSV {
		CH_HUE,				///< Hue channel
		CH_SATURATION,		///< Saturation channel
		CH_VALUE			///< Value channel
	};



	// ================================ Common Feature Extractor Class ==============================
	/**
	* @brief Common class, which unites feature extraction algorithms.
	* @details In order to simplify the user feature extraction code, this class provides <a href="https://en.wikipedia.org/wiki/Fluent_interface">fluent interface</a>. 
	* Please see the example code below for more details.
	* @code
	* CCommonFeatureExtractor fex(img);
	* Mat saturation = fex.getSaturation().invert().get();					// Inverted saturation feature
	* Mat variance   = fex.getGradient().getVariance().blur().get();			// Varience of the gradient feature, after Gaussian blur
	* Mat intesity   = fex.getIntensity().reScale(sqNeighbourhood(2)).get();	// Intencity feature, calculated at scale of window size 5 x 5 pixels
	* @endcode
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/			
	class CCommonFeatureExtractor : public CBaseFeatureExtractor
	{
	public:
		/**
		* @brief Constructor.
		* @param img Input image.
		*/		
		DllExport CCommonFeatureExtractor(Mat &img) : CBaseFeatureExtractor(img) {}
		DllExport virtual ~CCommonFeatureExtractor(void) {}

		/**
		* @brief Returns the input image.
		* @returns The input image.
		*/
		DllExport Mat virtual get(void) const { return m_img; }

		/**
		* @brief Extracts a coordinate feature.
		* @details This function calculates the coordinate feature of image pixels, based inly on theirs coordinates.
		* @param type Type of the coordinate feature (Ref. @ref coordinateType).
		* @return Common feature extractor class with extracted coordinate feature of type \b CV_8UC1.
		*/		
		DllExport CCommonFeatureExtractor getCoordinate(coordinateType type = COORDINATE_ORDINATE) const { return CCommonFeatureExtractor(CCoordinate::get(m_img, type)); }
		/**
		* @brief Extracts the intesity feature.
		* @details This function calculates the intesity of the input image as follows: \f[ intensity=weight_0\cdot img.RED+weight_1\cdot img.GREEN+weight_2\cdot img.BLUE \f]
		* @param weight The weight coefficients, which determine the contribution of each color channel to the resulting intensity.
		* @return Common feature extractor class with extracted intensity feature of type \b CV_8UC1.
		*/		
		DllExport CCommonFeatureExtractor getIntensity(CvScalar weight = CV_RGB(0.333, 0.333, 0.333)) const { return CCommonFeatureExtractor(CIntensity::get(m_img, weight)); }
		/**
		* @brief Extracts the HSV feature.
		* @details This function transforms the input image into HSV (hue-saturation-value) color space.
		* @return The (hue-saturation-value) feature image of type \b CV_8UC3.
		*/
		DllExport CCommonFeatureExtractor getHSV(void) const { return CCommonFeatureExtractor(CHSV::get(m_img)); }
		/**
		* @brief Extracts the hue feature.
		* @details This function represents the input image in HSV (hue-saturation-value) color model and returns the hue channel.
		* @return Common feature extractor class with extracted hue feature of type \b CV_8UC1.
		*/
		DllExport CCommonFeatureExtractor getHue(void) const { return CCommonFeatureExtractor(CHSV::get(m_img)).getChannel(CH_HUE); }
		/**
		* @brief Extracts the saturation feature.
		* @details This function represents the input image in HSV (hue-saturation-value) color model and returns the saturation channel.
		* @return Common feature extractor class with extracted saturation feature of type \b CV_8UC1.
		*/
		DllExport CCommonFeatureExtractor getSaturation(void) const { return CCommonFeatureExtractor(CHSV::get(m_img)).getChannel(CH_SATURATION); }
		/**
		* @brief Extracts the brightness feature.
		* @details This function represents the input image in HSV (hue-saturation-value) color model and returns the value channel.
		* @return Common feature extractor class with extracted brightness feature of type \b CV_8UC1.
		*/
		DllExport CCommonFeatureExtractor getBrightness(void) const { return CCommonFeatureExtractor(CHSV::get(m_img)).getChannel(CH_VALUE); }
		/**
		* @brief Extracts the gradient feature.
		* @details This function calculates the magnitude of gradient of the input image as follows: \f[gradient=\sqrt{\left(\frac{d\,img}{dx}\right)^2+\left(\frac{d\,img}{dy}\right)^2},\f]
		* where \f$\frac{d\,img}{dx}\f$ and \f$\frac{d\,img}{dy}\f$ are the first \a x and \a y central derivatives of the input image.\n
		* As \f$gradient\in[0; 255\,\sqrt{2}]\f$, this function performs two-linear mapping of the gradient values to the interval \f$[0; 255]\f$, such that:
		* \f{eqnarray*}{0&\rightarrow&0 \\  mid&\rightarrow&255 \\  255\,\sqrt{2}&\rightarrow&255\f} 
		* For more details on mapping refer to the @ref two_linear_mapper() function.
		* @param mid Parameter for the two-linear mapping of the feature: \f$mid\in(0;255\sqrt{2}]\f$. (Ref. @ref two_linear_mapper()). 
		* @return Common feature extractor class with extracted gradient feature of type \b CV_8UC1.
		*/	
		DllExport CCommonFeatureExtractor getGradient(float mid = GRADIENT_MAX_VALUE) const { return CCommonFeatureExtractor(CGradient::get(m_img, mid)); }
		/**
		* @brief Extracts the NDVI (<a href="http://en.wikipedia.org/wiki/Normalized_Difference_Vegetation_Index">normalized difference vegetation index</a>) feature.
		* @details This function calculates the NDVI from the input image as follows: \f[ NDVI=\frac{NIR-VIS}{NIR+VIS},\f] 
		* where \f$NIR\f$ designates the \a near-infra-red data, and \f$VIS\f$ - \a visible data. 
		* The algorithm assumes that the near-infra-red data is stored in the red image channel and the visible data - in the remaining two channels: 
		* \f{eqnarray*}{NIR&=&img.RED \\ VIS&=&\frac{1}{2}\,img.GREEN+\frac{1}{2}\,img.BLUE.\f} 
		* As \f$NDVI\in[-1; 1]\f$, this function performs two-linear mapping of the NDVI values to the interval \f$[0; 255]\f$, such that:
		* \f{eqnarray*}{-1&\rightarrow&0 \\  0&\rightarrow&midPoint \\  1&\rightarrow&255\f} 
		* For more details on mapping refer to the @ref two_linear_mapper() function.
		* @param midPoint Parameter for the two-linear mapping of the feature (Ref. @ref two_linear_mapper()). 
		* > Common values are: 
		* > - 0&nbsp;&nbsp;&nbsp;&nbsp; - cut off the negative NDVI values;
		* > - 127 - linear mapping; 
		* > - 255 - cut off the positive NDVI values.
		* @return Common feature extractor class with extracted NDVI feature of type \b CV_8UC1.
		*/	
		DllExport CCommonFeatureExtractor getNDVI(byte midPoint = 127) const  { return CCommonFeatureExtractor(CNDVI::get(m_img, midPoint)); }
		/**
		* @brief Extracts the distance feature.
		* @details For each pixel of the source image this function calculates the distance to the closest pixel, which value is larger or equal to \b threshold. 
		* Resulting feature image is multiplied by \b multiplier
		* @param threshold Threshold value. 
		* @param multiplier Amplification coefficient for the resulting feature image.
		* @return Common feature extractor class with extracted distance feature of type \b CV_8UC1.
		*/
		DllExport CCommonFeatureExtractor getDistance(byte threshold = 16, double multiplier = 4.0) const { return CCommonFeatureExtractor(CDistance::get(m_img, threshold, multiplier)); }
		/**
		* @brief Extracts the HOG (<a href="http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients">histogram of oriented gradients</a>) feature.
		* @details For each pixel of the source image this function calculates the histogram of oriented gradients inside the pixel's neighbourhood \b nbhd.
		* The histogram consists of \b nBins values, it is normalized, and stored as \b nBins channel image, thus, the channel index corresponds to the histogram index.
		* @param nBins Number of bins. Hence a single bin covers an angle of \f$\frac{180^\circ}{nBins}\f$.
		* @param nbhd Neighborhood around the pixel, where its histogram is estimated. (Ref. @ref SqNeighbourhood).
		* @return Common feature extractor class with extracted HOG feature of type \b CV_8UC{n}, where \f$n=nBins\f$.
		*/
		DllExport CCommonFeatureExtractor getHOG(int nBins = 9, SqNeighbourhood nbhd = sqNeighbourhood(5)) const { return CCommonFeatureExtractor(CHOG::get(m_img, nBins, nbhd)); }
		/**
		* @brief Extracts the variance feature.
		* @details For each pixel of the source image this function calculates the variance within the pixel's neighbourhood \b nbhd.
		* @param nbhd Neighborhood around the pixel, where the variance is estimated. (Ref. @ref SqNeighbourhood).
		* @return Common feature extractor class with extracted variance feature of type \b CV_8UC1.
		*/		
		DllExport CCommonFeatureExtractor getVariance(SqNeighbourhood nbhd = sqNeighbourhood(5)) const { return CCommonFeatureExtractor(CVariance::get(m_img, nbhd)); }
		/**
		* @brief Extracts the sparse coding feature.
		* @details For each pixel of the source image this function calculates the sparse coding feature within the pixel's neighbourhood \b nbhd. 
		* > This fuction supports dictionaries with \a nWords less or equal to 512 words. For larger dictionaries use CSparseCoding::get_v() function directly.
		* @param D Sparse dictionary \f$D\f$:  Mat(size nWords x blockSize^2; type CV_32FC1).
		* > Dictionary should be learned from a training data with CSparseDictionary::train() function,<br>
		* > or it may be loaded directed from a \a dic file with CSparseDictionary::getDictionary("dictionary.dic").
		* @param nbhd Neighborhood around the pixel, where the feature is estimated. (Ref. @ref SqNeighbourhood).
		* @return Common feature extractor class with extracted sparse coding feature of type \b CV_8UC{nWords}.
		*/
		DllExport CCommonFeatureExtractor getSparseCoding(const Mat &D, SqNeighbourhood nbhd = sqNeighbourhood(3)) const { return CCommonFeatureExtractor(CSparseCoding::get(m_img, D, nbhd)); }
		/**
		* @brief Extracts the scale feature.
		* @details For each pixel of the source image this function calculates the mean value within the pixel's neighbourhood \b nbhd.
		* Using different neighbourhood radii, it alows for different scale representations of the nput features.
		* @param nbhd Neighborhood around the pixel, where the mean is estimated. (Ref. @ref SqNeighbourhood).
		* @return Common feature extractor class with extracted scale feature of type \b CV_8UC1.
		*/
		DllExport CCommonFeatureExtractor reScale(SqNeighbourhood nbhd = sqNeighbourhood(5)) const { return CCommonFeatureExtractor(CScale::get(m_img, nbhd)); }
		/**
		* @brief Inverts the source image
		* @returns Common feature extractor class with the inverted feature with the same number of channels.
		*/
		DllExport inline CCommonFeatureExtractor invert(void) const 
		{
			Mat res;
			bitwise_not(m_img, res);
			return CCommonFeatureExtractor(res);
		}
		/**
		* @brief Performs Gaussian blurring of the source image
		* @param R Radius of the Gaussian filter box: \f$(2R+1)\times(2R+1)\f$.
		* @returns Common feature extractor class with blurred feature with the same number of channels.
		*/
		DllExport inline CCommonFeatureExtractor blur(int R = 2) const
		{
			Mat res;
			R = 2 * R + 1;
			GaussianBlur(m_img, res, cvSize(R, R), 0.0, 0.0, BORDER_REFLECT);
			return CCommonFeatureExtractor(res);
		}
		/**
		* @brief Extracts one channel from the source image
		* @param channel Index of the required channel.
		* @returns Common feature extractor class with the required channel as a feature.
		*/
		DllExport inline CCommonFeatureExtractor getChannel(int channel) const
		{
			DGM_ASSERT_MSG(channel < m_img.channels(), "The required channel %d does not exist in the %d-channel source image", channel, m_img.channels());
			Mat res;
			vec_mat_t vChannels;
			split(m_img, vChannels);
			vChannels.at(channel).copyTo(res);
			vChannels.clear();
			return CCommonFeatureExtractor(res);
		}
	};
} }