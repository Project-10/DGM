
#pragma once

#define DGM_VERSION_MAJOR 1
#define DGM_VERSION_MINOR 5
#define DGM_VERSION_PATCH 1

#define DEBUG_MODE			
#define DEBUG_PRINT_INFO	
#define ENABLE_PPL
/* #undef ENABLE_AMP */
#define USE_OPENGL
#define USE_SHERWOOD


#include <vector>
#include <memory>
#include <thread>
#include <math.h>
#ifdef ENABLE_PPL
#include <ppl.h>
#endif
#ifdef ENABLE_AMP
#include <amp.h>
#endif
#include "opencv2\opencv.hpp"

using namespace cv;

using byte	= unsigned __int8;
using word	= unsigned __int16;
using dword	= unsigned __int32;
using qword	= unsigned __int64;

using vec_mat_t			= std::vector<Mat>;
using vec_bool_t		= std::vector<bool>;
using vec_byte_t		= std::vector<byte>;
using vec_word_t		= std::vector<word>;
using vec_int_t			= std::vector<int>;
using vec_float_t		= std::vector<float>;
using vec_size_t		= std::vector<size_t>;
using vec_string_t		= std::vector<std::string>;
using vec_scalar_t		= std::vector<CvScalar>;

using ptr_float_t	= std::unique_ptr<float[]>;

const double	Pi	= 3.1415926;			///< Pi number
const float		Pif	= 3.1415926f;			///< Pi number

#define isnan		_isnan
#define isinf		std::isinf
#define DllExport	__declspec(dllexport)

// DGM lib
namespace DirectGraphicalModels
{
	class	CNDGauss;
	using	vec_NDGauss_t	= std::vector<CNDGauss>;
	using   vec_nColor_t	= std::vector<std::pair<Scalar, std::string>>;
	const	size_t	STR_LEN	= 256; 
}
