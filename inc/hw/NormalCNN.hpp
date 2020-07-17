#ifndef __NORMAL_CONV_HEADER__
#define __NORMAL_CONV_HEADER__

#include <global/net_headers.hpp>
#include <hw/math_8bit.hpp>
#include <hw/NORMAL_MAC.hpp>
#include <hw/DataCopyWeights.hpp>

// A buffered implementation of a 2D filter.
#define KERNAL_SIZE 3
#define HALF_SIZE (((KERNAL_SIZE)-1) / 2)

template <typename t_CLIPING_MODE, typename t_ACCUMULATE_TYPE, unsigned t_MAX_INPUT_CHANNEL,
          unsigned t_MAX_INPUT_SIZE, unsigned t_MAX_OUTPUT_CHANNEL, unsigned t_KSIZE, unsigned t_STRIDE>
void NormalConvolution(
    fused_pipes &in_stream,
    fused_pipes &out_stream,
    dType_8u weights[t_MAX_OUTPUT_CHANNEL][t_MAX_INPUT_CHANNEL * t_KSIZE * t_KSIZE],
    dType_8u *iMult_bias_acc,
    dType_8t *nShift_bias_acc,
    dType_8u *iMult_output,
    dType_8u *nShift_output,
    dType_8u *weight_zp,
    dType_8u bias_zp,
    dType_8u input_zp,
    dType_8u output_zp,
    dType_8u *biases_local,
    dType_Reg o_size,
    dType_Reg i_size,
    dType_Reg o_c_size,
    dType_Reg i_c_size,
    t_CLIPING_MODE const &cliping_mode)

{
    dType_8u bias;
    dType_8u line_buf[t_MAX_INPUT_CHANNEL][t_KSIZE - 1][t_MAX_INPUT_SIZE];
#pragma HLS ARRAY_PARTITION variable = line_buf complete dim = 2

    dType_8u window[t_MAX_INPUT_CHANNEL][t_KSIZE][t_KSIZE];
#pragma HLS ARRAY_PARTITION variable = window complete dim = 0
    // dType_8u right[t_MAX_INPUT_CHANNEL][t_KSIZE];

    dType_Reg y_com = 0;
    dType_Reg x_com = 0;
    dType_Reg y_looper = 0;
    dType_Reg x_looper = 0;
    // int count_s1 = 0;
    // int count_s2 = 0;
    // Load initial values into line buffer
    dType_Reg read_count = (i_size * HALF_SIZE + HALF_SIZE + 1) * i_c_size;

buf_x1_n:
    for (dType_Reg x = i_size - HALF_SIZE - 1; x < i_size; x++)
    {
    itr_channel_x1:
        for (dType_Reg ch = 0; ch < i_c_size; ch++)
        {
#pragma HLS PIPELINE
            line_buf[ch][HALF_SIZE - 1][x] = in_stream.read();
        }
    }

buf_y_n:
    for (dType_8u y = HALF_SIZE; y < t_KSIZE - 1; y++)
    {
    buf_x2_n:
        for (dType_Reg x = 0; x < i_size; x++)
        {
        itr_channel_x2:
            for (dType_Reg ch = 0; ch < i_c_size; ch++)
            {
#pragma HLS PIPELINE
                line_buf[ch][y][x] = in_stream.read();
            }
        }
    }

win_y_n:
    for (dType_8u y = HALF_SIZE; y < t_KSIZE; y++)
    {
    win_x_n:
        for (dType_8u x = HALF_SIZE; x < t_KSIZE; x++)
        {
        win_itr_channel:
            for (dType_Reg ch = 0; ch < i_c_size; ch++)
            {
#pragma HLS PIPELINE
                window[ch][y][x] = line_buf[ch][y - 1][x + i_size - t_KSIZE];
            }
        }
    }
    // Start convolution
    dType_Reg yVal, xVal;

    if (t_STRIDE == 2)
    {
        x_looper = y_looper = o_size * 2;
    }
    else
    {
        x_looper = y_looper = o_size;
    }

// Start convolution
for_y_n:
    for (dType_Reg y = 0; y < y_looper; y++)
    {
        y_com = y;
    for_x_n:
        for (dType_Reg x = 0; x < x_looper; x++)
        {
            x_com = x;
        compute_loop_channel_n:
            for (dType_Reg oc = 0; oc < o_c_size; oc++)
            {
#pragma HLS PIPELINE II = 1

                bias = biases_local[oc];
                // Calculate output value
                dType_Reg val_out;
                normal_single_operation<t_MAX_OUTPUT_CHANNEL, t_MAX_INPUT_CHANNEL, t_KSIZE>(window, weights[oc], input_zp, weight_zp[oc], i_size, y, x, &val_out);
                //Quantize the output
                t_ACCUMULATE_TYPE out_i32;
#pragma HLS RESOURCE variable = out_i32 core = AddSub_DSP
                dType_Reg scaled_bias;
#pragma HLS RESOURCE variable = scaled_bias core = AddSub_DSP
                dType_16t bias_calib;
#pragma HLS RESOURCE variable = bias_calib core = AddSub_DSP

                bias_calib = bias - bias_zp;
                if (nShift_bias_acc[oc] > 0)
                {
                    scaled_bias = (bias_calib * iMult_bias_acc[oc]) >> nShift_bias_acc[oc];
                }
                else
                {
                    scaled_bias = (bias_calib * iMult_bias_acc[oc]) << abs_8bit(nShift_bias_acc[oc]);
                }
                // printf("\n%d\n",(int)scaled_bias);
                out_i32 = (val_out + scaled_bias);
                // printf("\n%d\n",(int)out_i32);
                out_i32 = out_i32 * iMult_output[oc];
                out_i32 = (out_i32 >> nShift_output[oc]) + output_zp;
                dType_8u out_8bit = clip<t_ACCUMULATE_TYPE>(out_i32, cliping_mode);
                // if (oc == 17 && y == 7 && x == 20)
                // {
                // printf("\n%d\n",(int)out_8bit);
                // printf("%d ",(int)out_8bit);
                // fflush (stdout);
                // Write output value
                // }
                if (t_STRIDE == 2)
                {
                    if (!(y_com % 2) && !(x_com % 2))
                    {
                        out_stream.write(out_8bit);
                        // count_s2++;
                    }
                }
                else
                {
                    out_stream.write(out_8bit);
                    // count_s1++;
                }
            }

        Data_movement_loop:
            for (dType_Reg ch = 0; ch < i_c_size; ch++)
            {
#pragma HLS PIPELINE
            shift_win_y:
                for (dType_8u y = 0; y < t_KSIZE; y++)
                {
                shift_win_x:
                    for (dType_8u x = 0; x < t_KSIZE - 1; x++)
                    {
                        window[ch][y][x] = window[ch][y][x + 1];
                    }
                }
                window[ch][0][t_KSIZE - 1] = line_buf[ch][0][x];
            update_idx1:
                for (int y = 1; y < t_KSIZE - 1; y++)
                {
                    window[ch][y][t_KSIZE - 1] = line_buf[ch][y - 1][x] = line_buf[ch][y][x];
                }

                int val_in = 0;
                if (read_count < i_size * i_size * i_c_size)
                {
                    val_in = in_stream.read();
                    read_count++;
                }
                window[ch][t_KSIZE - 1][t_KSIZE - 1] = line_buf[ch][t_KSIZE - 2][x] = val_in;
            }
        }
    }
    // std::cout << "count_s1: " << count_s1 << std::endl;
    // std::cout << "count_s2: " << count_s2 << std::endl;
}

#endif
