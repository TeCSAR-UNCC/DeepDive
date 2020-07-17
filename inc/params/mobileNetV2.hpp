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



#ifndef __MOBILE_NET_V2__HEADER__
#define __MOBILE_NET_V2__HEADER__


#include "indx_length.hpp"
#include "typedef_fixedpt.hpp"
#include "bit_ptr_rules.hpp"
#include "net_params_offset.hpp"
#include "buf_size.hpp"


#pragma SDS data zero_copy( \
                           image [0: __IMAGE_INPUT_LENGTH__], \
                           net_weights [0: __TOTAL_WIGHTS__], \
                           net_biases [0: __TOTAL_BIASES_LENGTH__])

#pragma SDS data access_pattern( \
                                image : SEQUENTIAL, \
                                net_weights : SEQUENTIAL, \
                                net_biases : SEQUENTIAL)

void MobileNetV2(
                dType_16t   * image,
                dType_16t   * net_weights,
                dType_16t   * net_biases,
                dType_Reg init_net_params);

#endif
