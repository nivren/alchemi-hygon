#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CHECK(x) do { hipError_t e=(x); if(e!=hipSuccess){fprintf(stderr,"%s\n",hipGetErrorString(e));return 1;} } while(0)
__global__ void fill(int *x) { int i=threadIdx.x; x[i]=i*2; return; }
int main() {
  int *d; int h[32]; CHECK(hipMalloc(&d,sizeof(h)));
  hipLaunchKernelGGL(fill,dim3(1),dim3(32),0,0,d);
  CHECK(hipGetLastError()); CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h,d,sizeof(h),hipMemcpyDeviceToHost)); CHECK(hipFree(d));
  for(int i=0;i<32;i++) if(h[i]!=i*2) return 2;
  puts("{\"status\":\"passed\",\"elements\":32,\"max_error\":0}");
}
