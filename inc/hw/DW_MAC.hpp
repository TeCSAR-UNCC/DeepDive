#ifndef __DW_MAC_CONV_HEADER__
#define __DW_MAC_CONV_HEADER__

#include <global/net_headers.hpp>

#define I_CH 32
#define I_SIZE_MAX 48
#define DEPTHWISE_KERNAL_SIZE 3
#define HALF_SIZE (((DEPTHWISE_KERNAL_SIZE)-1) / 2)

// Defines the actual calculation for one output value.
template <unsigned t_OUTPUT_CHANNEL, unsigned t_INPUT_CHANNEL, unsigned t_KSIZE>
void single_operation(dType_8u window[t_INPUT_CHANNEL][t_KSIZE][t_KSIZE],
                      dType_4u weight[t_KSIZE * t_KSIZE],
                      dType_8u input_zp, dType_8u weight_zp, dType_Reg i_size,
                      dType_Reg ch, dType_Reg y, dType_Reg x, dType_16t *const sum)
{
#pragma HLS INLINE
    dType_16t mul_res;
    dType_Reg mac_res = 0;
win_i:
    for (dType_Reg i = -HALF_SIZE; i <= HALF_SIZE; i++)
    {
    win_j:
        for (dType_Reg j = -HALF_SIZE; j <= HALF_SIZE; j++)
        {
            if (bounds_ok(y + i, x + j, i_size))
            {
                dType_8t actInp = window[ch][i + HALF_SIZE][j + HALF_SIZE] - input_zp;
                dType_8t actWeight = weight[(i + HALF_SIZE) * t_KSIZE + (j + HALF_SIZE)] - weight_zp;
                mul_res = actInp * actWeight;
                mac_res += mul_res;
                // if (ch == 0)
                // {
                //     printf(" %d * %d )+(", (int)(window[ch][i + HALF_SIZE][j + HALF_SIZE]- input_zp), (int)(weight[(i + HALF_SIZE) * t_KSIZE + (j + HALF_SIZE)]- weight_zp));
                //     fflush(stdout);
                // }
            }
        }
    }

    *sum = mac_res;
}

#endif
