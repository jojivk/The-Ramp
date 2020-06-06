
#include <assert.h>
#include <math.h>
#include "dnnl.hpp"
#include "example_utils.hpp"

using namespace dnnl;

//############################################################
//
//############################################################
void fill_data(std::vector<float> &tab)
{
  for(int i=0; i< tab.size(); ++i) {
    tab[i] = sind((float)i);
  }
}
//############################################################
//
//############################################################
void create_write_memory(std::vector<float> &mem, memory::dims dim, memmory::format_tags &tag)
{
  fill_data(mem);
  memory src_memory = memory({{dimm} dt::fp32, tag}, eng);
  write_tol_dnnl_memory(mem.data(), src_memory);
  return src_memory;
}
//############################################################
//
//############################################################
convolution_forward::primitive_desc create_conv_desc(
                  memory_dims conv_src_tz,
                  memory_dims conv_weights_tz,
                  memory_dims conv_bias_tz,
                  memory_dims conv_dst_tz,
                  memory_dims conv_strides,
                  memory_dims conv_padding)
{
   auto conv_src_md = memory::desc({conv_src_tz}, dt::fp32, tag::any);
   auto conv_bias_md = memory::desc({conv_bias_tz}, dt::fp32, tag::any);
   auto conv_weights_md = memory::desc({conv_weights_tz}, dt::fp32, tag::any);
   auto conv_dst_md = memory::desc({conv_dst_tz}, dt::fp32, tag::any);

   auto conv_desc = convolution_forward::desc(prop_kind::forward,
                      algorithm::convolution_direct, conv_src_md, conv_weights_md,
                      conv_bias_md, conv_dst_md, conv_strides, conv_padding,
                      conv_padding);

   auto conv_pd = convolution_foward::primitive_desc(conv_desc, eng);

}
                  

//############################################################
//
//############################################################
void simple_net(engine::kind engine_kind) {
   using tag = memory::format_tag;
   using  dt = memory::data_type;

   auto eng = engine(engine_kind, 0);
   stream s(eng);

   std::vector<primitieve> net_fwd, net_bwd;
   std::vector<std::unordered_map<int, memory>> net_fwd_args, net_bwd_args;

   const int batch = 32;
   std::vector<float> net_src(batch * 3 * 227 * 227);
   std::vector<float> net_dst(batch * 96 * 27 * 27);

   // Alexnet: conv
   // {batch, 3, 227, 227} (x) {96, 3, 11, 11} -> {batch, 96, 55, 55}
   // strides : {4, 4}

   memory::dims conv_src_tz = {batch, 3, 227, 227}'
   memory::dims conv_weights_tz = {96, 3, 11, 11};
   memory::dims conv_bias_tz = {96};
   memory::dims conv_dst_tz  = {batch, 96, 55, 55};
   memory::dims conv_strides = {4, 4};
   memory::dims conv_padding = {0, 0};

   std::vector<float> conv_weights(product(conv_weights_tz));
   std::vector<float> conv_bias(product(conv_bias_tz));

   auto conv_user_src_memory = create_write_to_memory(net_src, conv_src_tz, tag::nchw);
   auto conv_user_weights_memory = create_write_to_memory(conv_weights, conv_weighs_tz, tag::nchw);
   auto conv_user_bias_memory = create_write_to_memory(conv_dst, conv_dst_tz, tag::nchw);

   auto conv_pd = create_conv_desc(conv_src_md, conv_weights_md, conv_bias_md, 
                                   conv_dst_md, conv_strides, conv_padding);



















}
