#include "MAP.h"
#include "macroses.h"

namespace DirectGraphicalModels 
{
	float getMAP(const vec_byte_t &predictions, const vec_float_t &potentials, byte gt)
	{
		// Assertions
		DGM_ASSERT(predictions.size() == potentials.size());
		
		std::vector<std::pair<byte, float>> container;
		for (size_t i = 0; i < predictions.size(); i++)
			container.push_back(std::make_pair(predictions[i], potentials[i]));
		std::sort(container.begin(), container.end(), [](auto &left, auto &right) { return left.second > right.second; });

		float	res = 0.0f;
		int		sum	= 0;								// number of correct predictions
		int		i = 0;
		for (auto &c : container) {
			i++;
			if (c.first == gt) res += static_cast<float>(++sum) / i;
		}
		
		if (sum > 0) res /= sum;

		return res;
	}

}