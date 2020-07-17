#ifndef __COPY_WEIGHT_BODY_HEADER_Q__
#define __COPY_WEIGHT_BODY_HEADER_Q__

#include <global/net_headers.hpp>

template <
    unsigned t_MAX_INPUT_CHANNEL_PW_CONV, unsigned t_MAX_OUTPUT_CHANNEL_PW_CONV,
    unsigned t_MAX_OUTPUT_CHANNEL_DW_STR1_CONV, unsigned t_DW_STR1_KERNEL,
    unsigned t_MAX_OUTPUT_CHANNEL_DW_STR2_CONV, unsigned t_DW_STR2_KERNEL,
    unsigned t_MAX_INPUT_CHANNEL_POJC_CONV, unsigned t_MAX_OUTPUT_CHANNEL_POJC_CONV>
void copy_weights_body(dType_8u *weights_cpu,
                       dType_4u weights_PW_EXPND[t_MAX_OUTPUT_CHANNEL_PW_CONV][t_MAX_INPUT_CHANNEL_PW_CONV],
                       dType_4u weights_DW[t_MAX_OUTPUT_CHANNEL_DW_STR1_CONV][t_DW_STR1_KERNEL * t_DW_STR1_KERNEL],
                       //    dType_4u weights_DW_ST2[t_MAX_OUTPUT_CHANNEL_DW_STR2_CONV][t_DW_STR2_KERNEL * t_DW_STR2_KERNEL],
                       dType_4u weights_PW_PRJ[t_MAX_OUTPUT_CHANNEL_POJC_CONV][t_MAX_INPUT_CHANNEL_POJC_CONV],
                       dType_Reg t_LAYER_i_LENGTH, dType_Reg t_LAYER_i_OFFSET, dType_Reg t_LAYER_i_ip_chan, dType_Reg t_LAYER_i_k_size,
                       dType_Reg t_LAYER_i_1_LENGTH, dType_Reg t_LAYER_i_1_OFFSET, dType_Reg t_LAYER_i_1_ip_chan, dType_Reg t_LAYER_i_1_k_size,
                       dType_Reg t_LAYER_i_2_LENGTH, dType_Reg t_LAYER_i_2_OFFSET, dType_Reg t_LAYER_i_2_ip_chan, dType_Reg t_LAYER_i_2_k_size)

{
//Layer 1
LOOP_1_W_BODY:
    for (int idx = 0, oc = 0, j = 0; idx < t_LAYER_i_LENGTH; idx++, j++)
    {
        if (j == t_LAYER_i_k_size * t_LAYER_i_k_size * t_LAYER_i_ip_chan)
        {
            oc++;
            j = 0;
        }
#pragma HLS PIPELINE II = 1
        weights_PW_EXPND[oc][j] = weights_cpu[idx + t_LAYER_i_OFFSET];
    }

//Layer 2
LOOP_2_W_BODY:
    for (int idx = 0, oc = 0, j = 0; idx < t_LAYER_i_1_LENGTH; idx++, j++)
    {
#pragma HLS PIPELINE II = 1
        if (j == (t_LAYER_i_1_k_size * t_LAYER_i_1_k_size))
        {
            oc++;
            j = 0;
        }
        weights_DW[oc][j] = weights_cpu[idx + t_LAYER_i_1_OFFSET];
    }

//Layer 3
// LOOP_3_W_BODY:
//     for (int idx = 0, oc = 0, j = 0; idx < t_LAYER_i_1_LENGTH; idx++, j++)
//     {
// #pragma HLS PIPELINE II = 1
//         if (j == (t_LAYER_i_1_k_size * t_LAYER_i_1_k_size))
//         {
//             oc++;
//             j = 0;
//         }
//         weights_DW_ST2[oc][j] = weights_cpu[idx + t_LAYER_i_1_OFFSET];
//     }

//Layer 4
LOOP_4_W_BODY:
    for (int idx = 0, oc = 0, j = 0; idx < t_LAYER_i_2_LENGTH; idx++, j++)
    {
#pragma HLS PIPELINE II = 1
        if (j == (t_LAYER_i_2_k_size * t_LAYER_i_2_k_size * t_LAYER_i_2_ip_chan))
        {
            oc++;
            j = 0;
        }
        weights_PW_PRJ[oc][j] = weights_cpu[idx + t_LAYER_i_2_OFFSET];
    }
}

#endif