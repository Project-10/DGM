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
	 * thus arguments \b point of the addPoint() and getDensity() functions should be Scalar(a, b) , where a and b are 8-bit long values.
	 * > This class is curently used for test purposes. It works with 2-dimensional feature spaces only.
	 * @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CPDFHistogram2D : public IPDF
	{
	public:
		DllExport CPDFHistogram2D(void);
		DllExport virtual ~CPDFHistogram2D(void) = default;

		DllExport virtual void		reset(void) override;

		DllExport virtual void		addPoint(Scalar point) override;
		DllExport virtual double	getDensity(Scalar point) override;
		DllExport virtual void		smooth(unsigned int nIt) override;
		DllExport virtual Scalar	min(void) const override { return Scalar(0); }
		DllExport virtual Scalar	max(void) const override { return Scalar(255); }


	protected:
		DllExport virtual void		saveFile(FILE *pFile) const override;
		DllExport virtual void		loadFile(FILE *pFile) override;


	private:
		long m_data[256][256];
	};

}
