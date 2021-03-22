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
            
            static const int SIZE= 76;

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
            
            void generateWeights(){
                unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
                srand(seed);
                
                for (int i= 0; i < SIZE; i++) {
                    double f = (double)rand() / RAND_MAX;
                    double var = -0.5 + f * ((0.5) - (-0.5));
                    weight[i] = var;
//                    std::cout<<weight[i]<<" ";
                }
//                std::cout<<"\n";
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
