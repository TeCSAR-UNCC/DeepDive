#ifndef __COPY_iMult_DS_BODY_HEADER__
#define __COPY_iMult_DS_BODY_HEADER__

#include <global/net_headers.hpp>

inline void copy_iMult_downscaling_body(dType_8u *iMult_output, dType_8u *downscaling_iMult_PW_EXPND, dType_8u *downscaling_iMult_DW, dType_8u *downscaling_iMult_PW_PRJ, 
                                        dType_Reg t_LAYER_i_LENGTH, dType_Reg t_LAYER_i_OFFSET,
                                        dType_Reg t_LAYER_i_1_LENGTH, dType_Reg t_LAYER_i_1_OFFSET,
                                        dType_Reg t_LAYER_i_2_LENGTH, dType_Reg t_LAYER_i_2_OFFSET)
{
// #pragma HLS INLINE
// Layer i
LOOP_1_ACC_BODY:
    for (int idx = 0; idx < t_LAYER_i_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        downscaling_iMult_PW_EXPND[idx] = iMult_output[idx + t_LAYER_i_OFFSET];
    }

// Layer i+1
LOOP_2_ACC_BODY:
    for (int idx = 0; idx < t_LAYER_i_1_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        downscaling_iMult_DW[idx] = iMult_output[idx + t_LAYER_i_1_OFFSET];
    }

// Layer i+2
// LOOP_3_ACC_BODY:
//     for (int idx = 0; idx < t_LAYER_i_1_LENGTH; idx++)
//     {
// #pragma HLS PIPELINE II = 1
//         downscaling_iMult_DW_ST2[idx] = iMult_output[idx + t_LAYER_i_1_OFFSET];
//     }

// Layer i+3
LOOP_4_ACC_BODY:
    for (int idx = 0; idx < t_LAYER_i_2_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        downscaling_iMult_PW_PRJ[idx] = iMult_output[idx + t_LAYER_i_2_OFFSET];
    }
}

#endif