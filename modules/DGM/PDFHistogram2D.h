// Histogram-based Probability Density function class
// Written by Sergey Kosov in 2017 for Project X
#pragma once

#include "IPDF.h"

namespace DirectGraphicalModels
{
	// ================================ Histogram PDF Class ==============================
	/**
	* @brief Histogram-based PDF class (2D)
	* @details This class makes use of distribution histograms in order to estimate the PDF. The length of the histogram is 255 x 255,
	* thus arguments \b point of the addPoint() and getDensity() functions should be 8-bit long.
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CPDFHistogram2D : public IPDF
	{
	public:
		DllExport CPDFHistogram2D(void);
		DllExport virtual ~CPDFHistogram2D(void);

		DllExport virtual void		reset(void);

		DllExport virtual void		addPoint(Scalar point);
		DllExport virtual double	getDensity(Scalar point);
		DllExport virtual Scalar	min(void) const { return Scalar(0); }
		DllExport virtual Scalar	max(void) const { return Scalar(255); }

		/**
		* @brief Performs the gaussian smoothing on histogram
		* @details Performs \b nIt iterations of gaussian smothing of the histograms in order to overcome the "over-fitting" problem
		* @param nIt Number of iterations
		*/
		DllExport void				smooth(int nIt = 1);


	protected:
		DllExport virtual void		saveFile(FILE *pFile) const;
		DllExport virtual void		loadFile(FILE *pFile);


	private:
		long m_data[256][256];
	};

}