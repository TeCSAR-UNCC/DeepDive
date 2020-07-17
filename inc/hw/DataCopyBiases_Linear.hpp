#ifndef __COPY_BIASES_PW_HEADER__
#define __COPY_BIASES_PW_HEADER__

#include <global/net_headers.hpp>

template <
    unsigned t_LAYER_i_LENGTH,
    unsigned t_LAYER_i_OFFSET>
void copy_biases_linear(dType_8u *biases,
                        dType_4u *biases_local_1)
{
//Layer 1
LOOP_0_BQ_PW:
    for (int idx = 0; idx < t_LAYER_i_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
        biases_local_1[idx] = biases[idx + t_LAYER_i_OFFSET];
    }
}
#endif