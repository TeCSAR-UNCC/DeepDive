#ifndef __COPY_nShift_DS_PW_HEADER__
#define __COPY_nShift_DS_PW_HEADER__

#include <global/net_headers.hpp>

template <
    unsigned t_LAYER_i_LENGTH,
    unsigned t_LAYER_i_OFFSET>
void copy_nShift_downscaling_linear(dType_8u *nShift_output,
                                    dType_8u *downscaling_nShift_local_1)
{
//Layer 1
LOOP_1_nSHIFT_ACC_PW:
    for (int idx = 0; idx < t_LAYER_i_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        downscaling_nShift_local_1[idx] = nShift_output[idx + t_LAYER_i_OFFSET];
    }
}

#endif