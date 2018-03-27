#include "CameraControl.h"

namespace DirectGraphicalModels { namespace vis
{
	// Constructor
	CCameraControl::CCameraControl(GLFWwindow * window, float theta, float phi, float radius, float turnSpeed, float scrollSpeed, float panSpeed)
		: CTrackballCamera(theta, phi, radius)
		, m_mouseEvent(mouseEvent::none)
        , m_turnSpeed(turnSpeed)
        , m_scrollSpeed(scrollSpeed)
        , m_panSpeed(panSpeed)
	{
		glfwSetWindowUserPointer(window, this);
		glfwSetMouseButtonCallback(window, [](GLFWwindow *window, int button, int action, int mods) {
			static_cast<CCameraControl *>(glfwGetWindowUserPointer(window))->mouseButtonCallback(button, action, mods);
		});
		glfwSetCursorPosCallback(window, [](GLFWwindow *window, double x, double y) {
			static_cast<CCameraControl *>(glfwGetWindowUserPointer(window))->cursorPosCallback(x, y);
		});
		glfwSetScrollCallback(window, [](GLFWwindow *window, double x, double y) {
			static_cast<CCameraControl *>(glfwGetWindowUserPointer(window))->scrollCallback(x, y);
		});
	}

	// ------------------------------------- Callbacks -------------------------------------
	void CCameraControl::mouseButtonCallback(int button, int action, int mods)
	{
		if (button == GLFW_MOUSE_BUTTON_LEFT) {
			if (action == GLFW_PRESS)	m_mouseEvent = mouseEvent::start_turn;
			if (action == GLFW_RELEASE)	m_mouseEvent = mouseEvent::none;
		}
		if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
			if (action == GLFW_PRESS)	m_mouseEvent = mouseEvent::start_pan;
			if (action == GLFW_RELEASE)	m_mouseEvent = mouseEvent::none;
		}
	}

	void CCameraControl::cursorPosCallback(double x, double y)
	{
		if (m_mouseEvent == mouseEvent::none) return;

		float dx = m_mouseLastPos.x - static_cast<float>(x);	// dTheta
		float dy = m_mouseLastPos.y - static_cast<float>(y);	// dPhi
		m_mouseLastPos.x = static_cast<float>(x);
		m_mouseLastPos.y = static_cast<float>(y);

		switch (m_mouseEvent) {
			case mouseEvent::start_turn: m_mouseEvent = mouseEvent::turn; break;
			case mouseEvent::turn: rotate(dx * m_turnSpeed, dy * m_turnSpeed); break;
			case mouseEvent::start_pan: m_mouseEvent = m_mouseEvent = mouseEvent::pan; break;
			case mouseEvent::pan: pan(-dx * m_panSpeed, dy * m_panSpeed); break;
            case mouseEvent::none: break;
		}
	}

	void CCameraControl::scrollCallback(double x, double y)
	{
		zoom(static_cast<float>(y) * m_scrollSpeed);
	}
} }
