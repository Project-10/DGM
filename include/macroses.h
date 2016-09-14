#pragma once

namespace DirectGraphicalModels {
#define __ATTRIBUTES__ " in \"" __FILE__ "\", line " _CRT_STRINGIZE(__LINE__) ""
#define DGM_ASSERT(_condition_) \
	do { \
	    if (!(_condition_)) { \
	        printf("Assertion failed: %s", #_condition_ __ATTRIBUTES__); \
	        abort (); \
	    } \
	} while (0)

#define DGM_ASSERT_MSG(_condition_, _format_, ...) \
	do { \
	    if (!(_condition_)) { \
			printf("Assertion failed: %s\n",  #_condition_ __ATTRIBUTES__); \
			printf(_format_"\n", ##__VA_ARGS__); \
	        abort (); \
	    } \
	} while (0)

#define DGM_IF_WARNING(_condition_, _format_, ...) \
	do { \
	    if (_condition_) { \
			printf("WARNING: %s:\n",  #_condition_ __ATTRIBUTES__); \
			printf(_format_"\n", ##__VA_ARGS__); \
	    } \
	} while (0)

#define DGM_WARNING(_format_, ...) \
	do { \
		printf("WARNING:%s:\n",  __ATTRIBUTES__); \
		printf(_format_"\n", ##__VA_ARGS__); \
	} while (0)

#define SIGN(a) (((a) >= 0) ? 1 : -1)


	// Approximative pow() function
	// Taken from: http://martin.ankerl.com/2012/01/25/optimized-approximative-pow-in-c-and-cpp/
	inline double fastPow(double a, double b)
	{
		union {
			double d;
			int x[2];
		} u = { a };
		u.x[1] = (int)(b * (u.x[1] - 1072632447) + 1072632447);
		u.x[0] = 0;
		return u.d;
	}

	//template<typename T>
	//inline T SIGN(T a) {return (a >= 0) ? 1 : -1;}

	template<typename T, void (T::*SomeMethod)(byte b)>
	inline void DGM_ELEMENTWISE1(T &self, const Mat &m)
	{
		// Assertions
		DGM_ASSERT(m.type() == CV_8UC1);

		for (register int y = 0; y < m.rows; y++) {
			const byte *pM = m.ptr<byte>(y);
			for (register int x = 0; x < m.cols; x++)
				(self.*SomeMethod)(pM[x]);
		}
	}

	template<typename T, void (T::*SomeMethod)(byte b1, byte b2)>
	inline void DGM_ELEMENTWISE2(T &self, const Mat &m1, const Mat &m2)
	{
		// Assertions
		DGM_ASSERT(m1.size() == m2.size());
		DGM_ASSERT(m1.type() == m2.type());
		DGM_ASSERT(m1.type() == CV_8UC1);

		for (register int y = 0; y < m1.rows; y++) {
			const byte *pM1 = m1.ptr<byte>(y);
			const byte *pM2 = m2.ptr<byte>(y);
			for (register int x = 0; x < m1.cols; x++)
				(self.*SomeMethod)(pM1[x], pM2[x]);
		}
	}

	template<typename T, void (T::*SomeMethod)(byte b1, byte b2)>
	inline void DGM_ELEMENTWISE2(T &self, const Mat &m1, const Mat &m2, const Mat &mask)
	{
		// Assertions
		DGM_ASSERT((m1.size() == m2.size()) && (m2.size() == mask.size()));
		DGM_ASSERT((m1.type() == m2.type()) && mask.type() == CV_8UC1);
		DGM_ASSERT(m1.type() == CV_8UC1);

		for (register int y = 0; y < m1.rows; y++) {
			const byte *pM1 = m1.ptr<byte>(y);
			const byte *pM2 = m2.ptr<byte>(y);
			const byte *pMask = mask.ptr<byte>(y);
			for (register int x = 0; x < m1.cols; x++)
				if (pMask[x]) (self.*SomeMethod)(pM1[x], pM2[x]);
		}
	}

	template<typename T, void (T::*SomeMethod)(const Mat &vec, byte b)>
	inline void DGM_VECTORWISE1(T &self, const Mat &m1, const Mat &m2)
	{
		// Assertions
		DGM_ASSERT(m1.size() == m2.size());
		DGM_ASSERT(m1.depth() == CV_8U);
		DGM_ASSERT(m2.type() == CV_8UC1);

		Mat vec(m1.channels(), 1, CV_8UC1);
		for (register int y = 0; y < m2.rows; y++) {
			const byte *pM1 = m1.ptr<byte>(y);
			const byte *pM2 = m2.ptr<byte>(y);
			for (register int x = 0; x < m2.cols; x++) {
				for (register int f = 0; f < vec.rows; f++) vec.at<byte>(f, 0) = pM1[vec.rows * x + f];
				(self.*SomeMethod)(vec, pM2[x]);
			} // x
		} // y
	}

	template<typename T, void (T::*SomeMethod)(const Mat &vec, byte b)>
	inline void DGM_VECTORWISE1(T &self, const vec_mat_t &m1, const Mat &m2)
	{
		// Assertions
		DGM_ASSERT(m1[0].size() == m2.size());
		DGM_ASSERT(m1[0].type() == CV_8UC1);
		DGM_ASSERT(m2.type() == CV_8UC1);

		Mat vec(static_cast<word>(m1.size()), 1, CV_8UC1);
		for (register int y = 0; y < m2.rows; y++) {
			byte const **pM1 = new const byte *[vec.rows];
			for (register int f = 0; f < vec.rows; f++) pM1[f] = m1[f].ptr<byte>(y);
			const byte *pM2 = m2.ptr<byte>(y);
			for (register int x = 0; x < m2.cols; x++) {
				for (register int f = 0; f < vec.rows; f++) vec.at<byte>(f, 0) = pM1[f][x];
				(self.*SomeMethod)(vec, pM2[x]);
			} // x
		} // y
	}


	template<typename T, void (T::*SomeMethod)(const Mat &vec, byte b1, byte b2)>
	inline void DGM_VECTORWISE2(T &self, const Mat &m1, const Mat &m2, const Mat &m3)
	{
		// Assertions
		DGM_ASSERT(m1.size() == m2.size());
		DGM_ASSERT(m1.size() == m3.size());
		DGM_ASSERT(m1.depth() == CV_8U);
		DGM_ASSERT(m2.type() == CV_8UC1);
		DGM_ASSERT(m3.type() == CV_8UC1);

		Mat vec(m1.channels(), 1, CV_8UC1);
		for (register int y = 0; y < m2.rows; y++) {
			const byte *pM1 = m1.ptr<byte>(y);
			const byte *pM2 = m2.ptr<byte>(y);
			const byte *pM3 = m3.ptr<byte>(y);
			for (register int x = 0; x < m2.cols; x++) {
				for (register int f = 0; f < vec.rows; f++) vec.at<byte>(f, 0) = pM1[vec.rows * x + f];
				(self.*SomeMethod)(vec, pM2[x], pM3[x]);
			} // x
		} // y
	}
}