# include"TrackballCamera.h"

namespace DirectGraphicalModels { namespace vis 
{
	void CTrackballCamera::reset(void)
	{
		m_theta				= m_initTheta;
		m_phi				= m_initPhi;
		m_radius			= m_initRadius;
		m_up				= m_phi >= 0 ? 1.0f : -1.0f;
		m_target			= glm::vec3(0);
		m_viewNeedsUpdate	= true;
	}
	
	glm::mat4 CTrackballCamera::getViewMatrix(void)
	{
		if (m_viewNeedsUpdate) {
			updateViewMatrix();
			m_viewNeedsUpdate = false;
		}
		return m_view;
	}

	// ====================================== Potected ======================================
	void CTrackballCamera::rotate(float dTheta, float dPhi)
	{
		m_viewNeedsUpdate = true;

		// Update theta and phi
		if (m_up > 0.0f) m_theta += dTheta;
		else 			 m_theta -= dTheta;
		m_phi += dPhi;

		// Keep phi within -2Pi to +2Pi for easy 'up' comparison 
		if (m_phi > 2 * glm::pi<float>())	m_phi -= 2 * glm::pi<float>();
		else if (m_phi < -2 * glm::pi<float>())	m_phi += 2 * glm::pi<float>();

		// If phi is between 0 to Pi or -Pi to -2Pi, make 'up' be positive Y, other wise make it negative Y 
		if ((m_phi > 0 && m_phi < glm::pi<float>()) || (m_phi < -glm::pi<float>() && m_phi > -2 * glm::pi<float>())) m_up = 1.0f;
		else																										 m_up = -1.0f;
	}

	void CTrackballCamera::zoom(float dRadius)
	{
		m_viewNeedsUpdate = true;
		m_radius -= dRadius;
		if (m_radius <= 0.1f) m_radius = 0.1f;
	}

	void CTrackballCamera::pan(float dx, float dy)
	{
		m_viewNeedsUpdate = true;

		glm::vec3 look = glm::normalize(getCameraOrintation());
		glm::vec3 worldUp = glm::vec3(0.0f, m_up, 0.0f);
		glm::vec3 right = glm::cross(look, worldUp);
		glm::vec3 up = glm::cross(look, right);

		m_target += (right * dx) + (up * dy);
	}

} }