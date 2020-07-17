#ifndef __AUXILIARY_FUNCTION__
#define __AUXILIARY_FUNCTION__

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <string>

#include <global/net_headers.hpp>
#include <hw/math_8bit.hpp>
#include <utils/npy.hpp>
 
bool read_imagenet_classes(const std::string &fileName, std::vector<std::string> & classes);



#define PRINT(info, var) std::cout << info << var << std::endl;

#define IS_MEM_NOT_ALLOCATED (image_0 == NULL) ||          \
                                 (net_biases_0 == NULL) || \
                                 (output_0 == NULL)

void concat(dType_8u *image_0, dType_8u *output_0, int const &outChan, int const &out_size);

void print_qnt_results(dType_8u *image, int out_ch, int out_size);

void saveNPYFile(dType_8u *out, const std::string &filename, dType_Reg out_ch, dType_Reg im_Size);

template <typename T>
void softmax(const std::vector<T> &inp, std::vector<T> &conf)
{
    float sum = 0;
    std::vector<T> tmp(inp.size());
    for (int i = 0; i < inp.size(); i++)
    {
        tmp[i] = exp(inp[i]);
        sum += tmp[i];
    }
    for (int i = 0; i < inp.size(); i++)
    {
        conf[i] = tmp[i] / sum;
    }
}

template <typename t_INPUT_TYPE, unsigned t_TRIP_COUNT>
void vector_round_clip(t_INPUT_TYPE const *in, dType_8uf *tmp, dType_8u *out)
{
#pragma HLS INLINE
    for (int i = 0; i < t_TRIP_COUNT; i++)
    {
        tmp[i] = in[i];
        out[i] = tmp[i];
    }
}

template <typename t_INPUT_TYPE, unsigned t_TRIP_COUNT>
void vector_trunc_clip(t_INPUT_TYPE const *in, dType_8u *out)
{
    for (int i = 0; i < t_TRIP_COUNT; i++)
    {
        dType_16t tmp = in[i];
        if (in[i] < 0)
        {
            tmp = 0;
        }
        else if (in[i] > 255)
        {
            tmp = 255;
        }
        out[i] = tmp;
    }
}

#endif