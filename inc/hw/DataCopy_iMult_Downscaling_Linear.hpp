#ifndef __COPY_iMult_DS_PW_HEADER__
#define __COPY_iMult_DS_PW_HEADER__

#include <global/net_headers.hpp>

template <
    unsigned t_LAYER_i_LENGTH,
    unsigned t_LAYER_i_OFFSET>
void copy_iMult_downscaling_linear(dType_8u *iMult_output,
                                   dType_8u *accum_iMult_local_1)
{
// Layer i
LOOP_1_ACC_PW:
    for (int idx = 0; idx < t_LAYER_i_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        accum_iMult_local_1[idx] = iMult_output[idx + t_LAYER_i_OFFSET];
    }
}

#endif