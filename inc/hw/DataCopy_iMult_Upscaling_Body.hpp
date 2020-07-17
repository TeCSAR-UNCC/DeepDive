#ifndef __COPY_iMult_US_BODY_HEADER__
#define __COPY_iMult_US_BODY_HEADER__

#include <global/net_headers.hpp>

inline void copy_iMult_upscaling_body(dType_8u *iMult_upscaling_bias, dType_8u *iMult_biases_acc_PW_EXPND, dType_8u *iMult_biases_acc_DW, dType_8u *iMult_biases_acc_PW_PRJ,
                                      dType_Reg t_LAYER_i_LENGTH, dType_Reg t_LAYER_i_OFFSET,
                                      dType_Reg t_LAYER_i_1_LENGTH, dType_Reg t_LAYER_i_1_OFFSET,
                                      dType_Reg t_LAYER_i_2_LENGTH, dType_Reg t_LAYER_i_2_OFFSET)
{
// #pragma HLS INLINE
//Layer 1
LOOP_1_ACC_BODY:
    for (int idx = 0; idx < t_LAYER_i_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        iMult_biases_acc_PW_EXPND[idx] = iMult_upscaling_bias[idx + t_LAYER_i_OFFSET];
    }

//Layer 2
LOOP_2_ACC_BODY:
    for (int idx = 0; idx < t_LAYER_i_1_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        iMult_biases_acc_DW[idx] = iMult_upscaling_bias[idx + t_LAYER_i_1_OFFSET];
    }

//Layer 3
// LOOP_3_ACC_BODY:
//     for (int idx = 0; idx < t_LAYER_i_1_LENGTH; idx++)
//     {
// #pragma HLS PIPELINE II = 1
//         iMult_biases_acc_DW_ST2[idx] = iMult_upscaling_bias[idx + t_LAYER_i_1_OFFSET];
//     }

//Layer 4
LOOP_4_ACC_BODY:
    for (int idx = 0; idx < t_LAYER_i_2_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        iMult_biases_acc_PW_PRJ[idx] = iMult_upscaling_bias[idx + t_LAYER_i_2_OFFSET];
    }
}

#endif