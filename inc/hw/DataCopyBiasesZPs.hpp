#ifndef __COPY_BIASES_ZP_HEADER__
#define __COPY_BIASES_ZP_HEADER__

#include <global/net_headers.hpp>

inline void copy_bias_zp(dType_8u *weight_zp, dType_8u *biases_zp_local_1)
{
//All Layers
LOOP_Biases_ZP:
    for (int idx = 0; idx < __TOTAL_BIAS_ZERO_POINTS_LENGTH__; idx++)
    {
#pragma HLS PIPELINE II = 1
        biases_zp_local_1[idx] = weight_zp[idx];
    }
}


template <unsigned LAYER_1_LENGTH,
          unsigned LAYER_2_LENGTH,
          unsigned LAYER_3_LENGTH>
void copy_bias_zp_head(dType_8u *weight_zp,
                       dType_8u biases_zp_local_1[LAYER_1_LENGTH],
                       dType_4u biases_zp_local_2[LAYER_2_LENGTH],
                       dType_4u biases_zp_local_3[LAYER_3_LENGTH])
{
//Layers_1
LOOP_Biases_ZP_layer_1:
    for (int idx = 0; idx < LAYER_1_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        biases_zp_local_1[idx] = weight_zp[idx];
    }
//Layers_2
LOOP_Biases_ZP_layer_2:
    for (int idx = LAYER_1_LENGTH, j = 0; idx < LAYER_1_LENGTH + LAYER_2_LENGTH; idx++, j++)
    {
#pragma HLS PIPELINE II = 1
        biases_zp_local_1[j] = weight_zp[idx];
    }
//Layers_3
LOOP_Biases_ZP_layer_3:
    for (int idx = LAYER_1_LENGTH + LAYER_2_LENGTH, j = 0; idx < (LAYER_1_LENGTH + LAYER_2_LENGTH + LAYER_3_LENGTH); idx++, j++)
    {
#pragma HLS PIPELINE II = 1
        biases_zp_local_1[j] = weight_zp[idx];
    }
}


#endif