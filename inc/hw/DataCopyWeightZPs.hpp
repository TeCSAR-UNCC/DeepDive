#ifndef __COPY_WEIGHT_ZP_HEADER__
#define __COPY_WEIGHT_ZP_HEADER__

#include <global/net_headers.hpp>

template <
    unsigned t_LAYER_i_LENGTH,
    unsigned t_LAYER_i_OFFSET,
    unsigned t_LAYER_i_1_LENGTH,
    unsigned t_LAYER_i_1_OFFSET,
    unsigned t_LAYER_i_2_LENGTH,
    unsigned t_LAYER_i_2_OFFSET>
void copy_weight_zp(dType_8u *weight_zp,
                    dType_8u *weights_zp_local_1,
                    dType_4u *weights_zp_local_2,
                    dType_4u *weights_zp_local_3)
{

#ifdef __DEBUG__
    int counter = 0;
#endif

//Layer 1
LOOP_1_ZP:
    for (int idx = 0; idx < t_LAYER_i_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        weights_zp_local_1[idx] = weight_zp[idx + t_LAYER_i_OFFSET];
#ifdef __DEBUG__
        counter++;
#endif
    }

//Layer 2
LOOP_2_ZP:
    for (int idx = 0; idx < t_LAYER_i_1_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        weights_zp_local_2[idx] = weight_zp[idx + t_LAYER_i_1_OFFSET];
#ifdef __DEBUG__
        counter++;
#endif
    }

//Layer 3
LOOP_3_ZP:
    for (int idx = 0; idx < t_LAYER_i_2_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        weights_zp_local_3[idx] = weight_zp[idx + t_LAYER_i_2_OFFSET];
#ifdef __DEBUG__
        counter++;
#endif
    }
    C_ASSERT((counter == __FEATURES_0_0_W_ZERO_POINT_LENGTH__ + __FEATURES_1_CONV_0_W_ZERO_POINT_LENGTH__ + __FEATURES_1_CONV_2_W_ZERO_POINT_LENGTH__), "x> Weight ZP lenght is not correct.")
}

#endif