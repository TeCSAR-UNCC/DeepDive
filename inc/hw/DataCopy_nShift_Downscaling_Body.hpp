#ifndef __COPY_nShift_DS_BODY_HEADER__
#define __COPY_nShift_DS_BODY_HEADER__

#include <global/net_headers.hpp>

inline void copy_nShift_downscaling_body(dType_8u *nShift_output, dType_8u *downscaling_nShift_PW_EXPND, dType_8u *downscaling_nShift_DW, dType_8u *downscaling_nShift_PW_PRJ, 
                                         dType_Reg t_LAYER_i_LENGTH, dType_Reg t_LAYER_i_OFFSET,
                                         dType_Reg t_LAYER_i_1_LENGTH, dType_Reg t_LAYER_i_1_OFFSET,
                                         dType_Reg t_LAYER_i_2_LENGTH, dType_Reg t_LAYER_i_2_OFFSET)
{
// #pragma HLS INLINE
//Layer 1
LOOP_1_nSHIFT_ACC_BODY:
    for (int idx = 0; idx < t_LAYER_i_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        downscaling_nShift_PW_EXPND[idx] = nShift_output[idx + t_LAYER_i_OFFSET];
    }

//Layer 2
LOOP_2_nSHIFT_ACC_BODY:
    for (int idx = 0; idx < t_LAYER_i_1_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        downscaling_nShift_DW[idx] = nShift_output[idx + t_LAYER_i_1_OFFSET];
    }

//Layer 3
// LOOP_3_nSHIFT_ACC_BODY:
//     for (int idx = 0; idx < t_LAYER_i_1_LENGTH; idx++)
//     {
// #pragma HLS PIPELINE II = 1
//         downscaling_nShift_DW_ST2[idx] = nShift_output[idx + t_LAYER_i_1_OFFSET];
//     }

//Layer 4
LOOP_4_nSHIFT_ACC_BODY:
    for (int idx = 0; idx < t_LAYER_i_2_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        downscaling_nShift_PW_PRJ[idx] = nShift_output[idx + t_LAYER_i_2_OFFSET];
    }
}
#endif