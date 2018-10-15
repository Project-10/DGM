// Color Space Transformation Class
// Written by Sergey G. Kosov in 2008 for MPII
// Extended by Sergey G. Kosov in 2014 - 2016 for Project X
#pragma once

#include "types.h"

#define DGM_HSV(h, s, v) cv::Scalar(h, s, v, 0 )

namespace DirectGraphicalModels { namespace vis
{
	/**
	* @brief Color space transformations
	* @details This namespace collects methods for converting color between color spaces
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	namespace colorspaces {
		/**
		* @brief Transforms color from \b RGB to \b BGR 
		* @param rgb Color in Red-Green-Blue format 
		* @return Color in Blue-Green-Red format
		*/
		inline cv::Scalar rgb2bgr(cv::Scalar rgb)
		{
			cv::Scalar bgr;
			bgr.val[0] = rgb.val[2];
			bgr.val[1] = rgb.val[1];
			bgr.val[2] = rgb.val[0];
			bgr.val[3] = rgb.val[3];
			return bgr;
		}
		/**
		* @brief Transforms color from \b RGB to \b BGR 
		* @param bgr Color in Blue-Green-Red format
		* @return Color in Red-Green-Blue format 
		*/
		inline cv::Scalar bgr2rgb(cv::Scalar bgr) { return rgb2bgr(bgr); }
		/**
		* @brief Transforms color from \b HSV to \b BGR space
		* @param hsv Color in Hue-Saturation-Value color space<br>
		* Hue:[0; 360); Saturation:[0; 255]: Value:[0; 255]
		* @return  Color in Blue-Green-Red format
		*/
		inline cv::Scalar hsv2bgr(cv::Scalar hsv)
		{
			double      hh, p, q, t, ff;
			long        i;
			cv::Scalar	out;

			if (hsv.val[1] <= 0.0) {			// < is bogus, just shuts up warnings
				out.val[0] = hsv.val[2];
				out.val[1] = hsv.val[2];
				out.val[2] = hsv.val[2];
				return out;
			}
			hh = hsv.val[0];
			if (hh >= 360.0) hh = 0.0;
			hh /= 60.0;
			i = (long)hh;
			ff = hh - i;
			p = hsv.val[2] * (1.0 - hsv.val[1] / 255.0);
			q = hsv.val[2] * (1.0 - (hsv.val[1] * ff) / 255.0);
			t = hsv.val[2] * (1.0 - (hsv.val[1] * (1.0 - ff)) / 255.0);

			switch (i) {
			case 0:
				out.val[0] = hsv.val[2];
				out.val[1] = t;
				out.val[2] = p;
				break;
			case 1:
				out.val[0] = q;
				out.val[1] = hsv.val[2];
				out.val[2] = p;
				break;
			case 2:
				out.val[0] = p;
				out.val[1] = hsv.val[2];
				out.val[2] = t;
				break;

			case 3:
				out.val[0] = p;
				out.val[1] = q;
				out.val[2] = hsv.val[2];
				break;
			case 4:
				out.val[0] = t;
				out.val[1] = p;
				out.val[2] = hsv.val[2];
				break;
			case 5:
			default:
				out.val[0] = hsv.val[2];
				out.val[1] = p;
				out.val[2] = q;
				break;
			}
			return out;
		}
		/**
		* @brief Transforms color from \b HSV to \b RGB space 
		* @param hsv Color in Hue-Saturation-Value color space<br>
		* Hue:[0; 360); Saturation:[0; 255]: Value:[0; 255]
		* @return Color in Red-Green-Blue format 
		*/
		inline cv::Scalar hsv2rgb(cv::Scalar hsv) { return bgr2rgb(hsv2bgr(hsv)); }
	}
} }