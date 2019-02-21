#include "Powell.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
	// Constructor
	CPowell::CPowell(size_t nParams) 
		: m_nParams(nParams)
		, m_vParams(nParams)
		, m_vDeltas(nParams)
		, m_vMin(nParams)
		, m_vMax(nParams)
		, m_vKappa(3)
		, m_vConverged(nParams)
	{
		reset();
	}

	void CPowell::reset(void)
	{
		m_paramID		= 0;				// first parameter
		m_nSteps		= 0;			
		m_koeff			= 1.0f;				// identity koefficient
		m_acceleration  = 0.1f;				// default search acceleration

		std::fill(m_vParams.begin(), m_vParams.end(),  0.0f);
		std::fill(m_vDeltas.begin(), m_vDeltas.end(),  0.1f);
		std::fill(m_vMin.begin(), m_vMin.end(), -FLT_MAX);
		std::fill(m_vMax.begin(), m_vMax.end(),  FLT_MAX);
		std::fill(m_vConverged.begin(), m_vConverged.end(), false);
		std::fill(m_vKappa.begin(), m_vKappa.end(), -1.0f);
	}

	void CPowell::setInitParams(const vec_float_t& vParams) 
	{
		DGM_ASSERT_MSG(m_vParams.size() == vParams.size(), "The size of the argument (%zu) ddoes not correspond to the number of parameters (%zu)", vParams.size(), m_nParams);
		
		for (size_t p = 0; p < vParams.size(); p++) {
			const float& param = vParams[p];
			if (param > m_vMax[p]) {
				DGM_WARNING("Argument[%zu]=%.2f exceeds the upper boundary %.2f and will not be set", p, param, m_vMax[p]);
				continue;
			}
			if (param < m_vMin[p]) {
				DGM_WARNING("Argument[%zu]=%.2f exceeds the lower boundary %.2f and will not be set", p, param, m_vMin[p]);
				continue;
			}
			m_vParams[p] = param;
		}
	}

	void CPowell::setMinParams(const vec_float_t& vMinParam) 
	{ 
		DGM_ASSERT_MSG(m_vMin.size() == vMinParam.size(), "The size of the argument (%zu) ddoes not correspond to the number of parameters (%zu)", vMinParam.size(), m_nParams);

		for (size_t p = 0; p < vMinParam.size(); p++) {
			const float& minParam = vMinParam[p];
			if (minParam > m_vParams[p]) DGM_WARNING("Argument[%zu]=%.2f contradicts the parameter value %.2f and will not be set", p, minParam, m_vParams[p]);
			else m_vMin[p] =minParam;
		}
	}

	void CPowell::setMaxParams(const vec_float_t& vMaxParam) 
	{ 
		DGM_ASSERT_MSG(m_vMax.size() == vMaxParam.size(), "The size of the argument (%zu) ddoes not correspond to the number of parameters (%zu)", vMaxParam.size(), m_nParams);
		
		for (size_t p = 0; p < vMaxParam.size(); p++) {
			const float& maxParam = vMaxParam[p];
			if (maxParam < m_vParams[p]) DGM_WARNING("Argument[%zu]=%.2f contradicts the parameter value %.2f and will not be set", p, maxParam, m_vParams[p]);
			else m_vMax[p] = maxParam;
		}
	}

	void CPowell::setDeltas(const vec_float_t& vDeltas) 
	{ 
		DGM_ASSERT_MSG(m_vDeltas.size() == vDeltas.size(), "The size of the argument (%zu) ddoes not correspond to the number of parameters (%zu)", vDeltas.size(), m_nParams);
		
		m_vDeltas = vDeltas;
	}

	void CPowell::setAcceleration(float acceleration)
	{
		if (acceleration >= 0.0f) m_acceleration = acceleration;
		else DGM_WARNING("Negative acceleration value was not set");
	}

	vec_float_t CPowell::getParams(float kappa)
	{
		// Assertions
		DGM_ASSERT_MSG(kappa > 0.0f, "Negative kappa values are not allowed");

#ifdef DEBUG_PRINT_INFO
		// Printing out the information
		printf("[%zu]:\t", m_paramID);
		for (float& param : m_vParams) printf("%.2f\t", param);
		printf("%.2f\n", kappa);
#endif

		// If converged, no further steps are required
		if (isConverged()) return m_vParams;

		// =============== Fill all 3 kappa values ===============
			 if (m_vKappa[oD] < 0) { m_vKappa[oD] = kappa; m_midPoint = curArg; } 
		else if (m_vKappa[mD] < 0)   m_vKappa[mD] = kappa;
		else if (m_vKappa[pD] < 0)   m_vKappa[pD] = kappa;

		while (true) {
			// Need kappa: -1
			if (m_vKappa[mD] < 0) {
				if (m_midPoint == minArg) m_vKappa[mD] = 0.0f;
				else {
					curArg = MAX(minArg, m_midPoint - m_koeff * delta);
					return m_vParams;
				}
			}

			// Need kappa: +1
			if (m_vKappa[pD] < 0) {
				if (m_midPoint == maxArg) m_vKappa[pD] = 0.0f;
				else {
					curArg = MIN(maxArg, m_midPoint + m_koeff * delta);
					return m_vParams;
				}
			}

			// =============== All 3 kappas are ready ===============
			float maxKappa = *std::max_element(m_vKappa.begin(), m_vKappa.end());

			if (maxKappa == m_vKappa[oD]) {			// >>>>> Middle value -> Proceed to the next argument
				convArg = true;
				curArg = m_midPoint;

				if (isConverged()) return m_vParams;				// we have converged

				m_paramID = (m_paramID + 1) % m_nParams;			// new argument

				// reset variabels for new argument
				m_vKappa[mD] = -1;
				m_vKappa[pD] = -1;
				m_nSteps = 0;
				m_koeff = 1.0;

				m_midPoint = curArg;							// refresh the middle point
			}
			else if (maxKappa == m_vKappa[mD]) {	// >>>>> Lower value -> Step argument down
				std::fill(m_vConverged.begin(), m_vConverged.end(), false);		// reset convergence

				m_midPoint = MAX(minArg, m_midPoint - m_koeff * delta);			// refresh the middle point

				// shift kappa
				m_vKappa[pD] = m_vKappa[oD];
				m_vKappa[oD] = m_vKappa[mD];
				m_vKappa[mD] = -1.0f;

				// increase the search step
				m_nSteps++;
				m_koeff += m_acceleration * m_nSteps;
			}
			else if (maxKappa == m_vKappa[pD]) {	// >>>>> Upper value -> Step argument up
				std::fill(m_vConverged.begin(), m_vConverged.end(), false);		// reset convergence

				m_midPoint = MIN(maxArg, m_midPoint + m_koeff * delta);			// refresh the middle point

				// shift kappa
				m_vKappa[mD] = m_vKappa[oD];
				m_vKappa[oD] = m_vKappa[pD];
				m_vKappa[pD] = -1.0f;

				// increase the search step
				m_nSteps++;
				m_koeff += m_acceleration * m_nSteps;
			}
		} // infinite loop
	}

	bool CPowell::isConverged(void) const
	{
		for (const bool& converged : m_vConverged) if (!converged) return false;
		return true;
	}
}
