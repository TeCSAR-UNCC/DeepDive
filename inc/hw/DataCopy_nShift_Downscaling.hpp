#ifndef __COPY_nShift_DS_HEADER__
#define __COPY_nShift_DS_HEADER__

#include <global/net_headers.hpp>

template <
    unsigned t_LAYER_i_LENGTH,
    unsigned t_LAYER_i_OFFSET,
    unsigned t_LAYER_i_1_LENGTH,
    unsigned t_LAYER_i_1_OFFSET,
    unsigned t_LAYER_i_2_LENGTH,
    unsigned t_LAYER_i_2_OFFSET>
void copy_nShift_downscaling(dType_8u *nShift_output, 
                             dType_8u *downscaling_nShift_local_1,
                             dType_8u *downscaling_nShift_local_2,
                             dType_8u *downscaling_nShift_local_3)
{
//Layer 1
LOOP_1_nSHIFT_ACC:
    for (int idx = 0; idx < t_LAYER_i_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        downscaling_nShift_local_1[idx] = nShift_output[idx + t_LAYER_i_OFFSET];
    }

//Layer 2
LOOP_2_nSHIFT_ACC:
    for (int idx = 0; idx < t_LAYER_i_1_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        downscaling_nShift_local_2[idx] = nShift_output[idx + t_LAYER_i_1_OFFSET];
    }

//Layer 3
LOOP_3_nSHIFT_ACC:
    for (int idx = 0; idx < t_LAYER_i_2_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        downscaling_nShift_local_3[idx] = nShift_output[idx + t_LAYER_i_2_OFFSET];
    }
}

#endif