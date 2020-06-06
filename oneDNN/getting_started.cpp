

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "dnnl.hpp"
#include "dnnl_debug.h"

#include "example_utils.hpp"

using namespace dnnl;

void getting_started_tutorial(engine::kind engine_kind) {
  engine eng(engine_kind, 0);
  stream engine_stream(eng);

  const int N =1, H =13, W = 13, C = 3;

  const int stride_N = H * W * C;
  const int stride_H = W * C; 
  const int stride_W = C;
  const int stride_C = 1;

  auto offset = [=] (int n, int h, int w, int c) {
    return n * stride_N + h * stride_H + w * stride_W + c * stride_C;
  };

  const int image_size = N * H * W * C;
  std::vector<float> image(image_size);
  
  for (int n=0; n<N; ++n) 
    for (int h=0; h<H; ++h)
      for (int w=0; w<W; ++w)
        for (int c =0; c<C; ++c) {
           int off = offset(n, h, w, c);
           image[off] = -std::cos(off/10.f);
        }

  auto src_md = memory_desc( {N, C, H, W}, memory::data_type::fp32, memory::format_tag::nhwc );
  auto alt_src_md = memory::desc( {N, C, H, W}, memory::data_type::fp32,
                                  {stride_N, stride_C, stride_H, stride_W});                                   

  if (src_md != alt_src_md)
     throw std::logic_error("Memory descriptor intialization mismatch.");

  auto src_mem = memory(src_md, eng);
  write_to_dnnl_memory(image_data(), src_mem);

  auto dst_mem = memory(src_md, eng);
  auto relu_d = eltwise_forward::desc(prop_kind::forward_inference,
                                      algorithm::eltwise_relu, src_md,
                                      0.f, 0.f);
  
  auto relu_pd = eltwise_forward::primitive_desc(relu_d, eng);
  auto relu = eltwise_forward(relu_pd);

  relu.execute(engine_stream, 
               {
                   { DNNL_ARG_SRC, src_mem},
                   { DNNL_ARG_DST, dst_mem},
               });
  engine_stream.wait();
  std::vector<float> relu_image(image_size);
  read_from_dnnl_memory(relu_image.data(), dst_mem);

  for (int n=0; n<N; ++n) 
    for (int h=0; h<H; ++h)
      for (int w=0; w<W; ++w)
        for (int c =0; c<C; ++c) {
           int off = offset(n, h, w, c);
           float expected = image[off] <0 ? 0.f : image[off];
           if (relu_image[off] != expected){
             std::cout<<" Bad res...!"<<std::endl;
             throw std::logic_error("Accuracy check failed.") 
           }
        }
}

int main(int argc, char **argv) {
  int exit_code = 0;
  engine::kind engine_kind = parse_engine_kind(argc, argv);
  try {
    getting_started_tutorial(engine_kind);
  } catch (dnnl::error &e){
    std::cout<<"oneDNN error caught: "<<std::endl;
    exit(0)
  }

  std::cout<<"Run successful..!"<<std::endl;
}
  


































  
}
