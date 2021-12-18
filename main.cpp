#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>  
#include <ATen/ATen.h>  
#include <ATen/DeviceGuard.h>
#include <memory>
#include <dlfcn.h>
// #include <libraries/dcn.cpython-38-x86_64-linux-gnu.so>
#include "models/edvr.cpp"  // load VSR model
using namespace std;

int main() {
  // Check CUDA NVIDIA Cudnn
  cout<<"Cuda is available: "<<torch::cuda::is_available()<<endl;
  cout<<"Cudnn is available: "<<torch::cuda::cudnn_is_available()<<endl;
  cout<<"GPU count: "<<torch::cuda::device_count()<<endl;

  // Set device
  torch::DeviceType device_type = at::kCPU;
  if(torch::cuda::is_available())
    device_type = torch::kCUDA;

  // Set a input img
  torch::Tensor input = torch::randn({1,5,3,63,63}).to(device_type);
  cout<<"Input size: "<<input.sizes()<<endl;

  EDVRNet net(3, 64);
  net.to(device_type);
  auto output = net.forward(input);
}