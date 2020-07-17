#ifndef __COPY_BIASES_HEADER__
#define __COPY_BIASES_HEADER__

#include <global/net_headers.hpp>
#include <utils/debug.hpp>

template <
    unsigned t_LAYER_i_LENGTH,
    unsigned t_LAYER_i_OFFSET,
    unsigned t_LAYER_i_1_LENGTH,
    unsigned t_LAYER_i_1_OFFSET,
    unsigned t_LAYER_i_2_LENGTH,
    unsigned t_LAYER_i_2_OFFSET>
void copy_biases_head(dType_8u *biases,
                      dType_8u *biases_local_1,
                      dType_4u *biases_local_2,
                      dType_4u *biases_local_3)
{

#if __DEBUG__
    int count = 0;
#endif

//Layer 1
LOOP_0_BQ:
    for (int idx = 0; idx < t_LAYER_i_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
#if __DEBUG__
        count++;
#endif
        biases_local_1[idx] = biases[idx + t_LAYER_i_OFFSET];
    }

//Layer 2
LOOP_2_BQ:
    for (int idx = 0; idx < t_LAYER_i_1_LENGTH; idx++)
    {

#pragma HLS PIPELINE II = 1
#if __DEBUG__
        count++;
#endif
        biases_local_2[idx] = biases[idx + t_LAYER_i_1_OFFSET];
    }
//Layer 3
LOOP_3_BQ:
    for (int idx = 0; idx < t_LAYER_i_2_LENGTH; idx++)
    {
#pragma HLS PIPELINE II = 1
#if __DEBUG__
        count++;
#endif

        biases_local_3[idx] = biases[idx + t_LAYER_i_2_OFFSET];
    }

    //C_ASSERT(count == t_LAYER_i_LENGTH +
    //                      t_LAYER_i_1_LENGTH +
    //                      t_LAYER_i_2_LENGTH,
    //         "x> Bias copy length is not same!");
}

#endif