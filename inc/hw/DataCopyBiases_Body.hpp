#ifndef __COPY_BIASES_HEADER_BODY__
#define __COPY_BIASES_HEADER_BODY__

#include <global/net_headers.hpp>

inline void copy_biases_body(dType_8u *biases, dType_4u *biases_local_PW_EXPND, dType_4u *biases_local_DW, dType_4u *biases_local_PW_PRJ, 
                             dType_Reg t_LAYER_i_LENGTH, dType_Reg t_LAYER_i_OFFSET,
                             dType_Reg t_LAYER_i_1_LENGTH, dType_Reg t_LAYER_i_1_OFFSET,
                             dType_Reg t_LAYER_i_2_LENGTH, dType_Reg t_LAYER_i_2_OFFSET)
{
// #pragma HLS INLINE
//Layer 1
LOOP_0_BQ_BODY:
    for (int idx = 0; idx < t_LAYER_i_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        biases_local_PW_EXPND[idx] = biases[idx + t_LAYER_i_OFFSET];
    }

//Layer 2
LOOP_2_BQ_BODY:
    for (int idx = 0; idx < t_LAYER_i_1_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        biases_local_DW[idx] = biases[idx + t_LAYER_i_1_OFFSET];
    }

//Layer 3
// LOOP_3_BQ_BODY:
//     for (int idx = 0; idx < t_LAYER_i_1_LENGTH; idx++)
//     {
// #pragma HLS PIPELINE II = 1
//         biases_local_DW_ST2[idx] = biases[idx + t_LAYER_i_1_OFFSET];
//     }

//Layer 4
LOOP_4_BQ_BODY:
    for (int idx = 0; idx < t_LAYER_i_2_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        biases_local_PW_PRJ[idx] = biases[idx + t_LAYER_i_2_OFFSET];
    }
}

#endif