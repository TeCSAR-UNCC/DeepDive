#ifndef __LINEAR_HEADER__
#define __LINEAR_HEADER__

#include <global/net_headers.hpp>

template <unsigned t_DEPTH_SIZE>
void read_data(
    fused_pipes &in,
    dType_8u input_local[t_DEPTH_SIZE])
{
INPUT_DEPTH_LOOP:
    for (int i = 0; i < t_DEPTH_SIZE; i++)
    {
#pragma HLS PIPELINE
        input_local[i] = in.read();
    }
}

template <unsigned t_DEPTH_SIZE>
void read_weight(
    weight_pipes &in,
    dType_4u weight_local[t_DEPTH_SIZE])
{
#pragma HLS INLINE
WEIGHT_DEPTH_LOOP:
    for (int i = 0; i < t_DEPTH_SIZE / 2; i++)
    {
        weight_local[i * 2] = in.read();
        weight_local[(i * 2) + 1] = in.read();
    }
}

template <unsigned t_ROW_SIZE, unsigned t_DEPTH_SIZE>
void compute_core(
    dType_8u input[t_DEPTH_SIZE],
    dType_4u weight[t_ROW_SIZE][t_DEPTH_SIZE],
    dType_4u weight_zp[t_ROW_SIZE],
    dType_4u biases[t_ROW_SIZE],
    dType_8u bias_zp,
    dType_8u iMult_bias_acc[t_ROW_SIZE],
    dType_8t nShift_bias_acc[t_ROW_SIZE],
    dType_8u iMult_output[t_ROW_SIZE],
    dType_8u nShift_output[t_ROW_SIZE],
    dType_8u input_zp,
    dType_8u output_zp,
    fused_pipes &res)
{
    int y;
    dType_Reg sum[t_ROW_SIZE];

LINEAR_CORE_COMP_ROW_LOOP_SUM:
    for (int i = 0; i < t_ROW_SIZE; i++)
    {
        // if (i == 159)
        // {
        //     y=0;
        // }
        sum[i] = 0;
        dType_4u w_zp = weight_zp[i];
    LINEAR_CORE_COMP_DEPTH_LOOP:
        for (int j = 0; j < t_DEPTH_SIZE; j++)
        {
#pragma HLS UNROLL FACTOR=320
            dType_Reg k_weight = weight[i][j] - w_zp;
            // #pragma HLS RESOURCE variable = k_weight core = AddSub_DSP
            dType_Reg input_recalib = input[j] - input_zp;
            // #pragma HLS RESOURCE variable = input_recalib core = AddSub_DSP
            dType_Reg term = k_weight * input_recalib;
            // #pragma HLS RESOURCE variable = term core = DSP48
            // if (i == 159){
            // printf(" %d * %d )+(", (int)input_recalib, (int)k_weight);
            // printf(" %d ", (int)k_weight);
            // fflush(stdout);
            // }
            sum[i] += term;
        }
    }
LINEAR_CORE_COMP_ROW_LOOP:
    for (int i = 0; i < t_ROW_SIZE; i++)
    {
        // if (i == 159)
        // {
        //     y=0;
        // }
#pragma HLS PIPELINE
        // if (t_MAX_OUTPUT_CHANNEL == 1280)
        // {
        // printf("\n%d\n", (int)sum);
        // }
        dType_8uf out_8bit;
        dType_32f out_i32;
#pragma HLS RESOURCE variable = out_i32 core = AddSub_DSP

        dType_Reg bias_calib = biases[i] - bias_zp;
#pragma HLS RESOURCE variable = bias_calib core = AddSub_DSP
        dType_Reg upScaled_Bias = (bias_calib * iMult_bias_acc[i]);
#pragma HLS RESOURCE variable = upScaled_Bias core = DSP48
        dType_Reg scaled_bias;

        if (nShift_bias_acc[i] > 0)
        {
            scaled_bias = upScaled_Bias >> abs_8bit(nShift_bias_acc[i]);
        }
        else
        {
            scaled_bias = upScaled_Bias << abs_8bit(nShift_bias_acc[i]);
        }
        // if (t_MAX_OUTPUT_CHANNEL == 1280)
        // {
        // printf("\n%d\n", (int)scaled_bias);
        // }
        out_i32 = (sum[i] + scaled_bias);
        // if (t_MAX_OUTPUT_CHANNEL == 1280)
        // {
        // printf("\n%d\n", (int)out_i32);
        // }
        dType_Reg downScaled_out = (out_i32 * iMult_output[i]);
#pragma HLS RESOURCE variable = downScaled_out core = DSP48

        out_i32 = (downScaled_out >> nShift_output[i]) + output_zp;
        out_8bit = out_i32;
        // if (t_MAX_OUTPUT_CHANNEL == 1280)
        // {
        // printf("\n%d\n", (int)out_8bit);
        //     // printf("%d ",(int)out_8bit[oc]);
        // fflush(stdout);
        // }
        res.write(out_8bit);
    }
}

//NOTE: there is not t_COL_SIZE, since it is one.
template <unsigned t_ROW_SIZE, unsigned t_DEPTH_SIZE, unsigned t_CORE_NUMBER>
void Linear(fused_pipes &in,
            dType_4u weight_local[t_ROW_SIZE][t_DEPTH_SIZE],
            fused_pipes &out,
            dType_4u weight_zp[t_ROW_SIZE],
            dType_4u biases[t_ROW_SIZE],
            dType_8u bias_zp,
            dType_8u iMult_bias_acc[t_ROW_SIZE],
            dType_8t nShift_bias_acc[t_ROW_SIZE],
            dType_8u iMult_output[t_ROW_SIZE],
            dType_8u nShift_output[t_ROW_SIZE],
            dType_8u input_zp,
            dType_8u output_zp)
{
#pragma HLS DATAFLOW
    dType_8u input_local[t_DEPTH_SIZE];
    CORE_SIZE(input_local, t_CORE_NUMBER, 1)
    // dType_4u weight_local[t_DEPTH_SIZE];
    // CORE_SIZE(input_local, t_CORE_NUMBER, 1)

    read_data<t_DEPTH_SIZE>(in, input_local);
    compute_core<t_ROW_SIZE, t_DEPTH_SIZE>(input_local, weight_local, weight_zp, biases, bias_zp,
                                           iMult_bias_acc, nShift_bias_acc, iMult_output, nShift_output,
                                           input_zp, output_zp, out);
}

#endif