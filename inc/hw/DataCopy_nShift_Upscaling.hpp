#ifndef __COPY_nSHIFT_UPSCALING_HEADER__
#define __COPY_nSHIFT_UPSCALING_HEADER__

#include <global/net_headers.hpp>

template <
    unsigned t_LAYER_i_LENGTH,
    unsigned t_LAYER_i_OFFSET,
    unsigned t_LAYER_i_1_LENGTH,
    unsigned t_LAYER_i_1_OFFSET,
    unsigned t_LAYER_i_2_LENGTH,
    unsigned t_LAYER_i_2_OFFSET>
void copy_nShift_upscaling(dType_8t *nShift_bias_acc,
                           dType_8t *nShif_biases_acc_local_1,
                           dType_8t *nShif_biases_acc_local_2,
                           dType_8t *nShif_biases_acc_local_3)
{
//Layer 1
LOOP_1_nShift_US:
    for (int idx = 0; idx < t_LAYER_i_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        nShif_biases_acc_local_1[idx] = nShift_bias_acc[idx + t_LAYER_i_OFFSET];
    }

//Layer 2
LOOP_2_nShift_US:
    for (int idx = 0; idx < t_LAYER_i_1_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        nShif_biases_acc_local_2[idx] = nShift_bias_acc[idx + t_LAYER_i_1_OFFSET];
    }

//Layer 3
LOOP_3_nShift_US:
    for (int idx = 0; idx < t_LAYER_i_2_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        nShif_biases_acc_local_3[idx] = nShift_bias_acc[idx + t_LAYER_i_2_OFFSET];
    }
}

#endif
