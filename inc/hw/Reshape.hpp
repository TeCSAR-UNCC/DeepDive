#ifndef __RESHAPE__HEADER__
#define __RESHAPE__HEADER__

#include <global/net_headers.hpp>

template <unsigned t_OUTPUT_CHANNEL, unsigned t_OUTPUT_SIZE>
void channelWiseToColumnWise(dType_8u *cin_0, dType_8u *channelwise)
{
#pragma HLS INLINE
    dType_Reg global_idx = 0;
    for (dType_Reg i = 0; i < t_OUTPUT_SIZE; i++)
    {
        for (dType_Reg j = 0; j < t_OUTPUT_SIZE; j++)
        {
            for (dType_Reg c = 0; c < t_OUTPUT_CHANNEL; c++)
            {
#pragma HLS PIPELINE II = 1
                channelwise[(c * (t_OUTPUT_SIZE * t_OUTPUT_SIZE)) + (i * t_OUTPUT_SIZE + j)] = cin_0[global_idx];
                global_idx++;
            }
        }
    }
}


template <unsigned t_OUTPUT_CHANNEL, unsigned t_OUTPUT_SIZE>
void channelWiseToColumnWise(fused_pipes &cin_0, dType_8u *channelwise)
{
#pragma HLS INLINE
    for (dType_Reg i = 0; i < t_OUTPUT_SIZE; i++)
    {
        for (dType_Reg j = 0; j < t_OUTPUT_SIZE; j++)
        {
            for (dType_Reg c = 0; c < t_OUTPUT_CHANNEL; c++)
            {
#pragma HLS PIPELINE II = 1
                channelwise[(c * (t_OUTPUT_SIZE * t_OUTPUT_SIZE)) + (i * t_OUTPUT_SIZE + j)] = cin_0.read();
            }
        }
    }
}


#endif