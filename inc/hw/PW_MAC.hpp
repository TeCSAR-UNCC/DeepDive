#ifndef __PW_MAC_CONV_HEADER__
#define __PW_MAC_CONV_HEADER__

#include <global/net_headers.hpp>

// Defines the actual calculation for one output value.
template <unsigned t_CORE_SIZE, unsigned t_INPUT_MAX_CHANNEL, unsigned t_MAX_OUTPUT_CHANNEL, class t_SUM_TYPE>
void pw_single_operation(const dType_8u localFeature[t_INPUT_MAX_CHANNEL],
                         const dType_4u weights[t_MAX_OUTPUT_CHANNEL][t_INPUT_MAX_CHANNEL],
                         const dType_8u &input_zp,
                         const dType_4u &weight_zp_local,
                         const dType_16u &block_id,
                         const dType_Reg &oc,
                         const dType_Reg &i_c_size, t_SUM_TYPE *const sum)
{

#pragma HLS INLINE

    dType_8t k_weight;
    dType_16t input_recalib;

ADDER_TREE_LOOP:
    for (dType_16u j = 0; j < t_CORE_SIZE; j++)
    {
#pragma HLS UNROLL
        dType_Reg input = block_id * t_CORE_SIZE + j;
        input_recalib = localFeature[input] - input_zp;
        // if (t_MAX_OUTPUT_CHANNEL == 1280)
        // {
        //     if (input < i_c_size)
        //     {
        //         printf(" %d * %d )+(", (int)(localFeature[input] - input_zp), (int)(weights[oc][input] - weight_zp_local));
        //         fflush(stdout);
        //     }
        // }
        // k_weight = weights[weight_idx_offset + input];
        k_weight = weights[oc][input] - weight_zp_local;
        if (input < i_c_size)
        {
            dType_16t weighted_input = (input_recalib * k_weight);
            *sum = *sum + weighted_input;
        }
    }
}
#endif