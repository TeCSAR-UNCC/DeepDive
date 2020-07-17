#ifndef __COPY_nSHIFT_UPSCALING_BODY__
#define __COPY_nSHIFT_UPSCALING_BODY__

#include <global/net_headers.hpp>

inline void copy_nShift_upscaling_body(dType_8t *nShift_bias_acc, dType_8t *nShif_biases_acc_PW_EXPND, dType_8t *nShif_biases_acc_DW, dType_8t *nShif_biases_acc_PW_PRJ, //dType_8t *nShif_biases_acc_DW_ST2,
                                       dType_Reg t_LAYER_i_LENGTH, dType_Reg t_LAYER_i_OFFSET,
                                       dType_Reg t_LAYER_i_1_LENGTH, dType_Reg t_LAYER_i_1_OFFSET,
                                       dType_Reg t_LAYER_i_2_LENGTH, dType_Reg t_LAYER_i_2_OFFSET)
{
// #pragma HLS INLINE
//Layer 1
LOOP_1_nShift_US_BODY:
    for (int idx = 0; idx < t_LAYER_i_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        nShif_biases_acc_PW_EXPND[idx] = nShift_bias_acc[idx + t_LAYER_i_OFFSET];
    }

//Layer 2
LOOP_2_nShift_US_BODY:
    for (int idx = 0; idx < t_LAYER_i_1_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        nShif_biases_acc_DW[idx] = nShift_bias_acc[idx + t_LAYER_i_1_OFFSET];
    }

//Layer 3
// LOOP_3_nShift_US_BODY:
//     for (int idx = 0; idx < t_LAYER_i_1_LENGTH; idx++)
//     {
// #pragma HLS PIPELINE II = 1
//         nShif_biases_acc_DW_ST2[idx] = nShift_bias_acc[idx + t_LAYER_i_1_OFFSET];
//     }

//Layer 4
LOOP_4_nShift_US_BODY:
    for (int idx = 0; idx < t_LAYER_i_2_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        nShif_biases_acc_PW_PRJ[idx] = nShift_bias_acc[idx + t_LAYER_i_2_OFFSET];
    }
}

#endif
