#ifndef __NORMAL_MAC_CONV_HEADER__
#define __NORMAL_MAC_CONV_HEADER__

#include <global/net_headers.hpp>

// A buffered implementation of a 2D filter.
#define KERNAL_SIZE 3
#define HALF_SIZE (((KERNAL_SIZE)-1) / 2)

// Defines the actual calculation for one output value.
template <unsigned t_OUTPUT_CHANNEL, unsigned t_INPUT_CHANNEL, unsigned t_KSIZE>
void normal_single_operation(dType_8u window[t_INPUT_CHANNEL][t_KSIZE][t_KSIZE],
                             dType_8u weight[t_INPUT_CHANNEL * t_KSIZE * t_KSIZE],
                             dType_8u input_zp,
                             dType_8u weight_zp,
                             dType_Reg i_size,
                             dType_Reg y,
                             dType_Reg x, dType_Reg *const sum)
{
#pragma HLS INLINE
// #pragma HLS DATAFLOW

    dType_16t mul_res;
    dType_Reg mac_res = 0;

    int count = 0;

win_i_n:
    for (int i = -HALF_SIZE; i <= HALF_SIZE; i++)
    {
    win_j_n:
        for (int j = -HALF_SIZE; j <= HALF_SIZE; j++)
        {
        acc_depth_ch:
            for (int ch = 0; ch < t_INPUT_CHANNEL; ch++, count++)
            {
                if (bounds_ok(y + i, x + j, i_size))
                {
                    // z = (window[ch][i + HALF_SIZE][j + HALF_SIZE] - input_zp);
                    // printf("z = %d,",z);
                    // if (oc == 0)
                    // {
                    // printf(" %d * %d )+(", (int)(window[ch][i + HALF_SIZE][j + HALF_SIZE]- input_zp), (int)(weight[count]- weight_zp));
                    // fflush(stdout);
                    // }
                    mul_res = (window[ch][i + HALF_SIZE][j + HALF_SIZE] - input_zp) * (weight[count] - weight_zp);
                    mac_res += mul_res;
                }
            }
        }
    }
    *sum = mac_res;
}
#endif
