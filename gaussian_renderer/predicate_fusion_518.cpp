#include <torch/extension.h>
std::vector<torch::Tensor> predicate_cuda(std::vector<torch::Tensor>,double,bool);
std::vector<torch::Tensor> run(std::vector<torch::Tensor> a,double gamma,bool debug) {
 TORCH_CHECK(a.size()==13,"13 full-array/index inputs required");
 for(int k=0;k<13;k++) {
  TORCH_CHECK(a[k].is_cuda() && (k>=11 || a[k].is_contiguous()),"CUDA contiguous input required: ",k);
  TORCH_CHECK(a[k].device()==a[0].device(),"one device required");
  TORCH_CHECK(a[k].scalar_type()==(k<11?torch::kFloat64:torch::kInt64),"dtype mismatch ",k);
 }
 TORCH_CHECK(a[8].dim()==3 && a[8].size(1)==256 && a[8].size(2)==3,"points shape");
 TORCH_CHECK(a[9].sizes()==a[8].sizes(),"penv shape");
 TORCH_CHECK(a[11].dim()==1 && a[12].sizes()==a[11].sizes(),"ordered indices shape");
 return predicate_cuda(a,gamma,debug);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("run",&run);}
