#ifndef __COPY_WEIGHT_ZP_BODY_HEADER__
#define __COPY_WEIGHT_ZP_BODY_HEADER__

#include <global/net_headers.hpp>

inline void copy_weight_zp_body(dType_8u *weight_zp, dType_4u *weights_zp_PW_EXPND, dType_4u *weights_zp_DW, dType_4u *weights_zp_PW_PRJ,
                                dType_Reg t_LAYER_i_LENGTH, dType_Reg t_LAYER_i_OFFSET,
                                dType_Reg t_LAYER_i_1_LENGTH, dType_Reg t_LAYER_i_1_OFFSET,
                                dType_Reg t_LAYER_i_2_LENGTH, dType_Reg t_LAYER_i_2_OFFSET)
{
// #pragma HLS INLINE
//Layer 1
LOOP_1_ZP:
    for (int idx = 0; idx < t_LAYER_i_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        weights_zp_PW_EXPND[idx] = weight_zp[idx + t_LAYER_i_OFFSET];
    }

//Layer 2
LOOP_2_ZP:
    for (int idx = 0; idx < t_LAYER_i_1_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        weights_zp_DW[idx] = weight_zp[idx + t_LAYER_i_1_OFFSET];
    }

//Layer 3
// LOOP_3_ZP:
//     for (int idx = 0; idx < t_LAYER_i_1_LENGTH; idx++)
//     {
// #pragma HLS PIPELINE II = 1
//         weights_zp_DW_ST2[idx] = weight_zp[idx + t_LAYER_i_1_OFFSET];
//     }

//Layer 4
LOOP_4_ZP:
    for (int idx = 0; idx < t_LAYER_i_2_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        weights_zp_PW_PRJ[idx] = weight_zp[idx + t_LAYER_i_2_OFFSET];
    }
}
#endif