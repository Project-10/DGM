#pragma once

#include "DGM.h"
#include "DNN.h"
#include "types.h"

namespace DirectGraphicalModels {
    namespace dnn
    {
        class CNeuron
        {
        public:
            
            static const int SIZE = 36;

            DllExport CNeuron(void) {
                //printf("CNeuron constructor\n");
            }
            DllExport ~CNeuron(void) = default;
            
            void setNodeValue(double thisValue) {
              value = thisValue;
            }

            double getNodeValue() {
              return value;
            }         
            
            void setWeight(int index, double x){
                weight[index] = x;
            }

            double getWeight(int i) {
                return weight[i];
            }
            
        private:
            double value;
            double weight[SIZE];
        };
    }
}
