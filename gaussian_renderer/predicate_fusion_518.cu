#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
__device__ __forceinline__ double ad(double a,double b){return __dadd_rn(a,b);}
__device__ __forceinline__ double su(double a,double b){return __dsub_rn(a,b);}
__device__ __forceinline__ double mu(double a,double b){return __dmul_rn(a,b);}
// Measured Torch CUDA Reduce.cuh 3-coordinate order; each partial starts at +0.
__device__ __forceinline__ double sum3(double a,double b,double c){return ad(ad(ad(0.,a),ad(0.,c)),ad(0.,b));}
__global__ void kernel(const double*c,const double*u,const double*v,const double*n,const double*nab,const double*h,const double*aa,const double*ae,const double*p,const double*pe,const double*rho,const int64_t*ii,const int64_t*jj,int64_t total,int64_t istride,int64_t jstride,double gamma,bool*coverout,bool*nearout,bool*uncout,bool*ambout,double*deout,double*dbg,float*sgout){
 int64_t z=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;if(z>=total)return;
 int64_t row=z/256,k=z%256,i=ii[row*istride],j=jj[row*jstride];
 double C[3],Uj[3],Vj[3],N[3],Na[3],P[3],Pe[3],D[3],Q[3];
 #pragma unroll
 for(int l=0;l<3;l++){C[l]=c[j*3+l];Uj[l]=u[j*3+l];Vj[l]=v[j*3+l];N[l]=n[j*3+l];Na[l]=nab[j*3+l];P[l]=p[(i*256+k)*3+l];Pe[l]=pe[(i*256+k)*3+l];D[l]=su(P[l],C[l]);}
 double B=sum3(mu(P[0],N[0]),mu(P[1],N[1]),mu(P[2],N[2]));
 double be=sum3(mu(Pe[0],Na[0]),mu(Pe[1],Na[1]),mu(Pe[2],Na[2]));
 double nv=sum3(mu(D[0],N[0]),mu(D[1],N[1]),mu(D[2],N[2]));
 double de=mu(gamma,ad(be,ae[j]));float sg=B>=0?1.f:-1.f;
 #pragma unroll
 for(int l=0;l<3;l++)Q[l]=su(mu(aa[j],D[l]),mu(nv,C[l]));
 double U=mu(sum3(mu(Q[0],Uj[0]),mu(Q[1],Uj[1]),mu(Q[2],Uj[2])),(double)sg);
 double V=mu(sum3(mu(Q[0],Vj[0]),mu(Q[1],Vj[1]),mu(Q[2],Vj[2])),(double)sg);
 double HB0=mu(h[j*2],fabs(B)),HB1=mu(h[j*2+1],fabs(B));
 double ue=mu(gamma,ad(ad(mu(ae[j],sum3(mu(Pe[0],fabs(Uj[0])),mu(Pe[1],fabs(Uj[1])),mu(Pe[2],fabs(Uj[2])))),mu(ad(be,mu(2.,ae[j])),sum3(mu(fabs(C[0]),fabs(Uj[0])),mu(fabs(C[1]),fabs(Uj[1])),mu(fabs(C[2]),fabs(Uj[2]))))),mu(h[j*2],be)));
 double ve=mu(gamma,ad(ad(mu(ae[j],sum3(mu(Pe[0],fabs(Vj[0])),mu(Pe[1],fabs(Vj[1])),mu(Pe[2],fabs(Vj[2])))),mu(ad(be,mu(2.,ae[j])),sum3(mu(fabs(C[0]),fabs(Vj[0])),mu(fabs(C[1]),fabs(Vj[1])),mu(fabs(C[2]),fabs(Vj[2]))))),mu(h[j*2+1],be)));
 double pos=mu(aa[j],(double)sg),dep=mu(nv,(double)sg);
 bool cov=(B!=0.)&&(pos>0.)&&(U>=-HB0)&&(U<HB0)&&(V>=-HB1)&&(V<HB1)&&(rho[j]>0.);
 bool possible=(pos>=mu(-gamma,ae[j]))&&(ad(U,HB0)>=-ue)&&(su(HB0,U)>=-ue)&&(ad(V,HB1)>=-ve)&&(su(HB1,V)>=-ve)&&(rho[j]>0.);
 bool unc=possible&&((fabs(B)<=mu(gamma,be))||(fabs(pos)<=mu(gamma,ae[j]))||(fabs(ad(U,HB0))<=ue)||(fabs(su(HB0,U))<=ue)||(fabs(ad(V,HB1))<=ve)||(fabs(su(HB1,V))<=ve));
 coverout[z]=cov;nearout[z]=cov&&(dep>de);uncout[z]=unc;ambout[z]=cov&&(fabs(dep)<=de);deout[z]=de;
 if(dbg){ // 23 binary64 channels, separately typed float32 sg.
  double vals[23]={B,be,D[0],D[1],D[2],nv,de,Q[0],Q[1],Q[2],U,V,HB0,HB1,ue,ve,pos,dep,0.,0.,0.,0.,0.};
  #pragma unroll
  for(int l=0;l<18;l++)dbg[z*18+l]=vals[l];sgout[z]=sg;
 }
}
std::vector<torch::Tensor> predicate_cuda(std::vector<torch::Tensor>a,double gamma,bool debug){
 c10::cuda::CUDAGuard guard(a[0].device());int64_t pairs=a[11].numel(),total=pairs*256;
 auto bo=a[0].options().dtype(torch::kBool),dob=a[0].options();
 std::vector<torch::Tensor> o={torch::empty({pairs,256},bo),torch::empty({pairs,256},bo),torch::empty({pairs,256},bo),torch::empty({pairs,256},bo),torch::empty({pairs,256},dob),torch::empty({debug?pairs:0,256,18},dob),torch::empty({debug?pairs:0,256},dob.dtype(torch::kFloat32))};
 if(total)kernel<<<(total+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(a[0].data_ptr<double>(),a[1].data_ptr<double>(),a[2].data_ptr<double>(),a[3].data_ptr<double>(),a[4].data_ptr<double>(),a[5].data_ptr<double>(),a[6].data_ptr<double>(),a[7].data_ptr<double>(),a[8].data_ptr<double>(),a[9].data_ptr<double>(),a[10].data_ptr<double>(),a[11].data_ptr<int64_t>(),a[12].data_ptr<int64_t>(),total,a[11].stride(0),a[12].stride(0),gamma,o[0].data_ptr<bool>(),o[1].data_ptr<bool>(),o[2].data_ptr<bool>(),o[3].data_ptr<bool>(),o[4].data_ptr<double>(),debug?o[5].data_ptr<double>():nullptr,debug?o[6].data_ptr<float>():nullptr);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return o;
}
