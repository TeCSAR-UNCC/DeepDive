#ifndef __COPY_WEIGHT_ZP_PW_HEADER__
#define __COPY_WEIGHT_ZP_PW_HEADER__

#include <global/net_headers.hpp>

template <
    unsigned t_LAYER_i_LENGTH,
    unsigned t_LAYER_i_OFFSET>
void copy_weight_zp_linear(dType_8u *weight_zp,
                           dType_4u *weights_zp_local_1)
{

//Layer 1
LOOP_1_ZP_PW:
    for (int idx = 0; idx < t_LAYER_i_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        weights_zp_local_1[idx] = weight_zp[idx + t_LAYER_i_OFFSET];
    }
}

#endif