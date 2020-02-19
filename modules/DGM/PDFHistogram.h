// Histogram-based Probability Density function class
// Written by Sergey Kosov in 2015 for Project X
#pragma once

#include "IPDF.h"

namespace DirectGraphicalModels 
{
	// ================================ Histogram PDF Class ==============================
	/**
	 * @brief Histogram-based PDF class (1D)
	 * @details This class makes use of distribution histograms in order to estimate the PDF. The length of the histogram is 255,
	 * thus arguments \b point of the addPoint() and getDensity() functions should be also in range [0; 255].
	 * @author Sergey G. Kosov, sergey.kosov@project-10.de
	 */	
	class CPDFHistogram : public IPDF
	{
	public:
		DllExport CPDFHistogram(void);
		DllExport virtual ~CPDFHistogram(void) = default;

		DllExport virtual void		reset(void) override;

		DllExport virtual void		addPoint(Scalar point) override;
		DllExport virtual double	getDensity(Scalar point) override;
		DllExport virtual void		smooth(unsigned int nIt) override;
		DllExport virtual Scalar	min(void) const override { return Scalar(0); }
		DllExport virtual Scalar	max(void) const override { return Scalar(255); }


	protected:
		DllExport virtual void	saveFile(FILE *pFile) const override;
		DllExport virtual void	loadFile(FILE *pFile) override;


	private:
		long m_data[256];
	};

}
