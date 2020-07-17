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

#ifndef __TYPEDEF__HEADER__
#define __TYPEDEF__HEADER__

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>

using namespace hls;

typedef ap_fixed<33, 32, AP_RND, AP_SAT_SYM> dType_32f;
typedef ap_int<33> dType_33t;
typedef ap_ufixed<8, 8, AP_RND, AP_SAT_SYM> dType_8uf;
typedef ap_uint<4> dType_4u;
typedef int dType_Reg;
typedef unsigned int dType_32u;
typedef short int dType_16t;
typedef unsigned short int dType_16u;
typedef signed char dType_8t;
typedef unsigned char dType_8u;
typedef hls::stream<dType_8u> fused_pipes;
typedef hls::stream<dType_4u> weight_pipes;
typedef hls::stream<dType_16t> b16t_pipes; 

#endif
