/*
Copyright (c) 2019, University of North Carolina at Charlotte All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Reza Baharani    - Transformative Computer Systems Architecture Research (TeCSAR) at UNC Charlotte
Steven Furgurson - Transformative Computer Systems Architecture Research (TeCSAR) at UNC Charlotte

This file has been generated automatically. DO NOT MODIFY IT.

*/

#ifndef __GLOBAL__HEADER__
#define __GLOBAL__HEADER__

#include <params/indx_length.hpp>
#include <params/typedef_fixedpt.hpp>
#include <params/net_params_offset.hpp>
#include <params/buf_size.hpp>
#include <params/net_spect_size.hpp>

enum COMMAND__
{
  __COMMAND__COMPUTE_HEAD__,
  __COMMAND__COMPUTE_IRB_SKIP_LINE_STR2__,
  __COMMAND__COMPUTE_IRB_SKIP_LINE_STR1__,
  __COMMAND__COMPUTE_IRB_STR2__,
  __COMMAND__COMPUTE_IRB_STR1__,
  __COMMAND__AVG__
};

enum OP_TYPE
{
  __COLUMN_WISE__,
  __CHANNEL_WISE__
};

#define IS_COMPUTE_HEAD(cmd_) (cmd_ == __COMMAND__COMPUTE_HEAD__)
#define IS_COMPUTE_BODY(cmd_) (cmd_ == __COMMAND__COMPUTE_IRB_SKIP_LINE_STR2__ || \
                               cmd_ == __COMMAND__COMPUTE_IRB_SKIP_LINE_STR1__ || \
                               cmd_ == __COMMAND__COMPUTE_IRB_STR2__ ||           \
                               cmd_ == __COMMAND__COMPUTE_IRB_STR1__)

#define IS_COMPUTE_STRIDE1(cmd_) (cmd_ == __COMMAND__COMPUTE_IRB_STR1__)

#define IS_COMPUTE_STRIDE2(cmd_) (cmd_ == __COMMAND__COMPUTE_IRB_STR2__)

#define IS_COMPUTE__STRIDE1_SKIPLINE(cmd_) (cmd_ == __COMMAND__COMPUTE_IRB_SKIP_LINE_STR1__)

#define IS_DW_STR1(cmd_) (cmd_ == __COMMAND__COMPUTE_IRB_STR1__)
#define IS_DW_STR2(cmd_) (cmd_ == __COMMAND__COMPUTE_IRB_STR2__)
#define IS_CONV(cmd_) (cmd_ == __COMMAND__CONV__)
#define IS_DW_CONV(cmd_) (cmd_ == __COMMAND__CONV_DW__)
#define IS_PW_NORM_CONV(cmd_) (cmd_ == __COMMAND__CONV__)
#define IS_POOLING(cmd_) (cmd == __COMMAND__AVG__)

#define IS_IDENTITY(_i_chan_, _o_chan_) (_i_chan_ == _o_chan_)

#define __GLOBAL_WORKER_NUMBER__ 6
#define __CAPSULE_SIZE__ 2
#define __MASK_HIGHER__ 0xF0
#define __MASK_LOWER__ 0x0F
#define ALIGNED_KERNEL_SIZE(KERNEL) ((KERNEL * KERNEL) + 1)

#define STRINGIFY(a) #a
#define CORE_SIZE(ARRAY_NAME, FACTOR_SIZE, DIM_IDX) \
  _Pragma(STRINGIFY(HLS ARRAY_PARTITION variable = ARRAY_NAME CYCLIC FACTOR = FACTOR_SIZE DIM = DIM_IDX))


//Classes are for different rounding.
class ap_accuracy_round_clip
{
};

class ap_accuracy_truc_clip
{
};

class ap_accuracy_none
{
};

//Classes are for resource allocation.
class ap_resource_lut
{
};

class ap_resource_dsp
{
};

class ap_resource_dflt
{
};

#endif