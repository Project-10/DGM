#pragma once

#include <string.h>
#include <assert.h>

template <class T> class MRFEnergy;

class TypeGeneral
{
private:
	struct Vector;		// node parameters and messages
	struct Edge;		// stores edge information and either forward or backward message


public:
	typedef enum
	{
		GENERAL,		// edge information is stored as Ki*Kj matrix. Inefficient!
		POTTS			// edge information is stored as one number (lambdaPotts).
	} Type;

	struct NodeData;	// argument to MRFEnergy::AddNode()
	struct EdgeData;	// argument to MRFEnergy::AddEdge()

	struct NodeData
	{
		NodeData(double* data); // data = pointer to array of size MRFEnergy::m_Kglobal

	private:
	friend struct Vector;
	friend struct Edge;
		double * m_data;
	};

	struct EdgeData
	{
		EdgeData(Type type, double lambdaPotts);	// type must be POTTS
		EdgeData(Type type, double * data);			// type must be GENERAL. data = pointer to array of size Ki*Kj
													// such that V(ki,kj) = data[ki + Ki*kj]

	private:
	friend struct Vector;
	friend struct Edge;
		Type		m_type;
		union
		{
			double	m_lambdaPotts;
			double*	m_dataGeneral;
		};
	};


	//////////////////////////////////////////////////////////////////////////////////
	////////////////////////// Visible only to MRFEnergy /////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////

private:
	friend class MRFEnergy<TypeGeneral>;

	struct Vector
	{
		static int	GetSizeInBytes(int K);				// returns -1 if invalid K's
		void		Initialize(int K, NodeData data);	// called once when user adds a node
		void		Add(int K, NodeData data);			// called once when user calls MRFEnergy::AddNodeData()

		void		SetZero(int K);                     // set this[k] = 0
		void		Copy(int K, Vector* V);             // set this[k] = V[k]
		void		Add(int K, Vector* V);              // set this[k] = this[k] + V[k]
		double		GetValue(int K, int k);             // return this[k]
		double		ComputeMin(int K, int& kMin);       // return min_k { this[k] }, set kMin
		double		ComputeAndSubtractMin(int K);       // same as previous, but additionally set this[k] -= vMin (and kMin is not returned)

		double		GetArrayValue(int K, int k);		// note: k is an integer in [0..GetArraySize()-1].
																					// For Potts functions GetArrayValue() and GetValue() are the same,
																					// but they are different for, say, 2-dimensional labels.
		void		SetArrayValue(int K, int k, double x);

	private:
	friend struct Edge;
		double		m_data[1];														// actual size is MRFEnergy::m_Kglobal
	};

	struct Edge
	{
		static int	GetSizeInBytes(int Ki, int Kj, EdgeData data);	// returns -1 if invalid data
		static int	GetBufSizeInBytes(int vectorMaxSizeInBytes);									// returns size of buffer need for UpdateMessage()
		void		Initialize(int Ki, int Kj, EdgeData data, Vector* Di, Vector* Dj); // called once when user adds an edge
		Vector	  * GetMessagePtr();
		void		Swap(int Ki, int Kj);							// if the client calls this function, then the meaning of 'dir'
																									// in distance transform functions is swapped

		// When UpdateMessage() is called, edge contains message from dest to source.
		// The function must replace it with the message from source to dest.
		// The update rule is given below assuming that source corresponds to tail (i) and dest corresponds
		// to head (j) (which is the case if dir==0).
		//
		// 1. Compute Di[ki] = gamma*source[ki] - message[ki].  (Note: message = message from j to i).
		// 2. Compute distance transform: set
		//       message[kj] = min_{ki} (Di[ki] + V(ki,kj)). (Note: message = message from i to j).
		// 3. Compute vMin = min_{kj} m_message[kj].
		// 4. Set m_message[kj] -= vMin.
		// 5. Return vMin.
		//
		// If dir==1 then source corresponds to j, sink corresponds to i. Then the update rule is
		//
		// 1. Compute Dj[kj] = gamma*source[kj] - message[kj].  (Note: message = message from i to j).
		// 2. Compute distance transform: set
		//       message[ki] = min_{kj} (Dj[kj] + V(ki,kj)). (Note: message = message from j to i).
		// 3. Compute vMin = min_{ki} m_message[ki].
		// 4. Set m_message[ki] -= vMin.
		// 5. Return vMin.
		//
		// If Edge::Swap has been called odd number of times, then the meaning of dir is swapped.
		//
		// Vector 'source' must not be modified. Function may use 'buf' as a temporary storage.
		double UpdateMessage(int Ksource, int Kdest, Vector* source, double gamma, int dir, void* buf);

		// If dir==0, then sets dest[kj] += V(ksource,kj).
		// If dir==1, then sets dest[ki] += V(ki,ksource).
		// If Swap() has been called odd number of times, then the meaning of dir is swapped.
		void AddColumn(int Ksource, int Kdest, int ksource, Vector* dest, int dir);

	protected:

		Type		m_type;

		// message
		Vector	* m_message;
	};

	struct EdgePotts : Edge
	{
	private:
	friend struct Edge;
		double	m_lambdaPotts;
	};

	struct EdgeGeneral : Edge
	{
	private:
	friend struct Edge;
		int		m_dir;				// 0 if Swap() was called even number of times, 1 otherwise
		double	m_data[1];			// array of size Ki*Kj
	};
};

//////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// Implementation ///////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

///////////////////// NodeData and EdgeData ///////////////////////

inline TypeGeneral::NodeData::NodeData(double* data)
{
	m_data = data;
}

inline TypeGeneral::EdgeData::EdgeData(Type type, double lambdaPotts)
{
	assert(type == POTTS);
	m_type = type;
	m_lambdaPotts = lambdaPotts;
}

inline TypeGeneral::EdgeData::EdgeData(Type type, double* data)
{
	assert(type == GENERAL);
	m_type = type;
	m_dataGeneral = data;
}

///////////////////// Vector ///////////////////////

inline int TypeGeneral::Vector::GetSizeInBytes(int K)
{
	return (K < 1) ? -1 : K * sizeof(double);
}

inline void TypeGeneral::Vector::Initialize(int K, NodeData data)
{
	memcpy(m_data, data.m_data, K * sizeof(double));
}

inline void TypeGeneral::Vector::Add(int K, NodeData data)
{
	for (int k = 0; k < K; k++)
		m_data[k] += data.m_data[k];
}

inline void TypeGeneral::Vector::SetZero(int K)
{
	memset(m_data, 0, K * sizeof(double));
}

inline void TypeGeneral::Vector::Copy(int K, Vector* V)
{
	memcpy(m_data, V->m_data, K * sizeof(double));
}

inline void TypeGeneral::Vector::Add(int K, Vector* V)
{
	for (int k = 0; k < K; k++)
		m_data[k] += V->m_data[k];
}

inline double TypeGeneral::Vector::GetValue(int K, int k)
{
	assert(k>=0 && k<K.m_K);
	return m_data[k];
}

inline double TypeGeneral::Vector::ComputeMin(int K, int& kMin)
{
	double vMin = m_data[0];
	kMin = 0;
	for (int k = 1; k < K; k++)
	{
		if (vMin > m_data[k])
		{
			vMin = m_data[k];
			kMin = k;
		}
	}

	return vMin;
}

inline double TypeGeneral::Vector::ComputeAndSubtractMin(int K)
{
	double vMin = m_data[0];

	for (int k = 1; k < K; k++)
		if (vMin > m_data[k]) vMin = m_data[k];

	for (int k = 0; k < K; k++)
		m_data[k] -= vMin;

	return vMin;
}

inline double TypeGeneral::Vector::GetArrayValue(int K, int k)
{
	assert(k >= 0 && k < K);
	return m_data[k];
}

inline void TypeGeneral::Vector::SetArrayValue(int K, int k, double x)
{
	assert(k>=0 && k<K.m_K);
	m_data[k] = x;
}

///////////////////// EdgeDataAndMessage implementation /////////////////////////

inline int TypeGeneral::Edge::GetSizeInBytes(int Ki, int Kj, EdgeData data)
{
	int messageSizeInBytes = ((Ki > Kj) ? Ki : Kj) * sizeof(double);

	switch (data.m_type)
	{
		case POTTS:
			if (Ki != Kj || data.m_lambdaPotts < 0)
			{
				return -1;
			}
			return sizeof(EdgePotts) + messageSizeInBytes;
		case GENERAL:
			return sizeof(EdgeGeneral) - sizeof(double) + Ki * Kj * sizeof(double) + messageSizeInBytes;
		default:
			return -1;
	}
}

inline int TypeGeneral::Edge::GetBufSizeInBytes(int vectorMaxSizeInBytes)
{
	return vectorMaxSizeInBytes;
}

inline void TypeGeneral::Edge::Initialize(int Ki, int Kj, EdgeData data, Vector* Di, Vector* Dj)
{
	m_type = data.m_type;

	switch (m_type)
	{
		case POTTS:
			((EdgePotts*)this)->m_lambdaPotts = data.m_lambdaPotts;
			m_message = (Vector*)((char*)this + sizeof(EdgePotts));
			break;
		case GENERAL:
			((EdgeGeneral*)this)->m_dir = 0;
			memcpy(((EdgeGeneral*)this)->m_data, data.m_dataGeneral, Ki * Kj * sizeof(double));
			m_message = (Vector*)((char*)this + sizeof(EdgeGeneral) - sizeof(double) + Ki * Kj * sizeof(double));
			break;
		default:
			assert(0);
	}

	memset(m_message->m_data, 0, ((Ki > Kj) ? Ki : Kj)*sizeof(double));
}

inline TypeGeneral::Vector* TypeGeneral::Edge::GetMessagePtr()
{
	return m_message;
}

inline void TypeGeneral::Edge::Swap(int Ki, int Kj)
{
	if (m_type == GENERAL)
	{
		((EdgeGeneral*)this)->m_dir = 1 - ((EdgeGeneral*)this)->m_dir;
	}
}

inline double TypeGeneral::Edge::UpdateMessage(int Ksource, int Kdest, Vector* source, double gamma, int dir, void* _buf)
{
	Vector* buf = (Vector*) _buf;
	double vMin;

	if (m_type == POTTS)
	{
		assert(Ksource.m_K == Kdest.m_K);

		int k, kMin;

		m_message->m_data[0] = gamma*source->m_data[0] - m_message->m_data[0];
		kMin = 0;
		vMin = m_message->m_data[0];

		for (k = 1; k < Ksource; k++)
		{
			m_message->m_data[k] = gamma*source->m_data[k] - m_message->m_data[k];
			kMin = 0;
			vMin = buf->m_data[0];
			if (vMin > m_message->m_data[k])
			{
				kMin = k;
				vMin = m_message->m_data[k];
			}
		}

		for (k = 0; k < Ksource; k++)
		{
			m_message->m_data[k] -= vMin;
			if (m_message->m_data[k] > ((EdgePotts*)this)->m_lambdaPotts)
			{
				m_message->m_data[k] = ((EdgePotts*)this)->m_lambdaPotts;
			}
		}
	}
	else if (m_type == GENERAL)
	{
		int ksource, kdest;
		double* data = ((EdgeGeneral*)this)->m_data;

		for (ksource = 0; ksource < Ksource; ksource++)
		{
			buf->m_data[ksource] = gamma*source->m_data[ksource] - m_message->m_data[ksource];
		}

		if (dir == ((EdgeGeneral*)this)->m_dir)
		{
			for (kdest = 0; kdest < Kdest; kdest++)
			{
				vMin = buf->m_data[0] + data[0 + kdest * Ksource];
				for (ksource=1; ksource < Ksource; ksource++)
				{
					if (vMin > buf->m_data[ksource] + data[ksource + kdest * Ksource])
					{
						vMin = buf->m_data[ksource] + data[ksource + kdest * Ksource];
					}
				}
				m_message->m_data[kdest] = vMin;
			}
		}
		else
		{
			for (kdest=0; kdest < Kdest; kdest++)
			{
				vMin = buf->m_data[0] + data[kdest + 0 * Kdest];
				for (ksource=1; ksource < Ksource; ksource++)
				{
					if (vMin > buf->m_data[ksource] + data[kdest + ksource * Kdest])
					{
						vMin = buf->m_data[ksource] + data[kdest + ksource * Kdest];
					}
				}
				m_message->m_data[kdest] = vMin;
			}
		}

		vMin = m_message->m_data[0];
		for (kdest = 1; kdest < Kdest; kdest++)
		{
			if (vMin > m_message->m_data[kdest])
			{
				vMin = m_message->m_data[kdest];
			}
		}

		for (kdest=0; kdest < Kdest; kdest++)
			m_message->m_data[kdest] -= vMin;

	} else {
		assert(0);
	}

	return vMin;
}

inline void TypeGeneral::Edge::AddColumn(int Ksource, int Kdest, int ksource, Vector* dest, int dir)
{
	assert(ksource >= 0 && ksource < Ksource);

	int k;

	if (m_type == POTTS) {
		for (k=0; k < ksource; k++)
			dest->m_data[k] += ((EdgePotts*)this)->m_lambdaPotts;
		for (k++; k < Kdest; k++)
			dest->m_data[k] += ((EdgePotts*)this)->m_lambdaPotts;
	}
	else if (m_type == GENERAL)	{
		double* data = ((EdgeGeneral*)this)->m_data;

		if (dir == ((EdgeGeneral*)this)->m_dir) {
			for (k = 0; k < Kdest; k++)
				dest->m_data[k] += data[ksource + k * Ksource];
		} else {
			for (k = 0; k < Kdest; k++)
				dest->m_data[k] += data[k + ksource * Kdest];
		}
	} else	{
		assert(0);
	}
}

