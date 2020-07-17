#ifndef __COPY_nSHIFT_UPSCALING_PW_HEADER__
#define __COPY_nSHIFT_UPSCALING_PW_HEADER__

#include <global/net_headers.hpp>

template <
    unsigned t_LAYER_i_LENGTH,
    unsigned t_LAYER_i_OFFSET>
void copy_nShift_upscaling_linear(dType_8t *nShift_bias_acc,
                                  dType_8t *nShif_biases_acc_local_1)
{
//Layer 1
LOOP_1_nShift_US_PW:
    for (int idx = 0; idx < t_LAYER_i_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        nShif_biases_acc_local_1[idx] = nShift_bias_acc[idx + t_LAYER_i_OFFSET];
    }
}

#endif
