#ifndef __COPY_iMult_DS_HEADER__
#define __COPY_iMult_DS_HEADER__

#include <global/net_headers.hpp>

template <
    unsigned t_LAYER_i_LENGTH,
    unsigned t_LAYER_i_OFFSET,
    unsigned t_LAYER_i_1_LENGTH,
    unsigned t_LAYER_i_1_OFFSET,
    unsigned t_LAYER_i_2_LENGTH,
    unsigned t_LAYER_i_2_OFFSET>
void copy_iMult_downscaling(dType_8u *iMult_output,
                            dType_8u *accum_iMult_local_1,
                            dType_8u *accum_iMult_local_2,
                            dType_8u *accum_iMult_local_3)
{
// Layer i
LOOP_1_ACC:
    for (int idx = 0; idx < t_LAYER_i_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        accum_iMult_local_1[idx] = iMult_output[idx + t_LAYER_i_OFFSET];
    }

// Layer i+1
LOOP_2_ACC:
    for (int idx = 0; idx < t_LAYER_i_1_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        accum_iMult_local_2[idx] = iMult_output[idx + t_LAYER_i_1_OFFSET];
    }

// Layer i+2
LOOP_3_ACC:
    for (int idx = 0; idx < t_LAYER_i_2_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        accum_iMult_local_3[idx] = iMult_output[idx + t_LAYER_i_2_OFFSET];
    }
}

#endif