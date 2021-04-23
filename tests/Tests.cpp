#include "Tests.h"
#include "DGM/parallel.h"
#include "DGM/random.h"

using namespace DirectGraphicalModels;

TEST_F(CTests, parallel_gemm) 
{
#if defined(ENABLE_PDP) && defined(ENABLE_AMP)	
	int width   = random::u<int>(10, 1000);
	int height  = random::u<int>(10, 1000);
	float alpha = random::U<float>(0.0f, 1.0f);
	float beta  = random::U<float>(0.0f, 1.0f);

	Mat A = random::U(Size(width, height),  CV_32FC1, 0.0, 100.0);
	Mat B = random::U(Size(height, width),  CV_32FC1, 0.0, 100.0);
	Mat C = random::U(Size(height, height), CV_32FC1, 0.0, 100.0);
	Mat ppl_res, amp_res;
	
	parallel::impl::ppl_gemm(A, B, alpha, C, beta, ppl_res);
	parallel::impl::amp_gemm(A, B, alpha, C, beta, amp_res);

	ASSERT_TRUE(std::equal(ppl_res.begin<float>(), ppl_res.end<float>(), amp_res.begin<float>()));
#endif
}

