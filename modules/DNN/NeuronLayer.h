#pragma once

#include "Neuron.h"

namespace DirectGraphicalModels {
    namespace dnn
    {
        class CNeuronLayer
        {
        public:
            DllExport CNeuronLayer(size_t numNeurons, size_t numConnection);
            DllExport CNeuronLayer(const CNeuronLayer&) = delete;
            DllExport ~CNeuronLayer(void) = default;

            DllExport bool      operator=(const CNeuronLayer&) = delete;

            DllExport void      generateRandomWeights(void);
            DllExport void      setValues(const Mat& values);
            DllExport Mat       getValues(void) const;
            DllExport void      dotProd(const CNeuronLayer& layer);

            // TODO: move this method to a proper place
            DllExport static void      backPropagate(CNeuronLayer& layerA, CNeuronLayer& layerB, CNeuronLayer& layerC, const Mat& resultErrorRate, float learningRate);


            // Accessors
            DllExport size_t    getNumNeurons(void) const { return m_vpNeurons.size(); }


        private:
            std::vector<ptr_neuron_t>   m_vpNeurons;
        };
    }
}