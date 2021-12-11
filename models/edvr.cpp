#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <iomanip>
#include <stdio.h>  

// #include "dcn/deform_conv_ext.cpp"
using namespace std;

// 
// class 


class ResidualBlockNoBN: public torch::nn::Module {
  // init module
  public:
    ResidualBlockNoBN(int mid_channels);
    torch::Tensor forward(torch::Tensor x);
  
  private:
    // register modules
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::ReLU relu{nullptr};
};


ResidualBlockNoBN::ResidualBlockNoBN(int mid_channels) {
  conv1 = register_module("conv1",torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels, mid_channels, {3,3}).stride(1).padding(1).bias(true)));
  conv2 = register_module("conv2",torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels, mid_channels, {3,3}).stride(1).padding(1).bias(true)));
  relu = register_module("relu", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
}


torch::Tensor ResidualBlockNoBN::forward(torch::Tensor x) {
  torch::Tensor identity = x;
  auto out = conv2->forward(relu->forward(conv1->forward(x)));
  return out;
}

// make residual blocks
torch::nn::Sequential make_layer(int num_blocks, int mid_channels) {
  torch::nn::Sequential features;
  
  for(int i = 0; i < num_blocks; i++) {
    auto conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels, mid_channels, {3,3}).stride(1).padding(1).bias(true));
    auto conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels, mid_channels, {3,3}).stride(1).padding(1).bias(true));
    auto relu = torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)); 

    features->push_back(conv1);
    features->push_back(relu);
    features->push_back(conv2);
  }

  return features;
};

// make TSA fusion module
torch::nn::Sequential tsa_fusion(int mid_channels, int num_frame, int center_frame_idx) {
  torch::nn::Sequential features;
  features->push_back(ResidualBlockNoBN(mid_channels));
  return features;
}

// EDVR Net
class EDVRNet: public torch::nn::Module{
  public:
    EDVRNet(int in_channels, int mid_channels);
    torch::Tensor spatial_padding(torch::Tensor lrs);  // Apply padding satially
    torch::Tensor forward(torch::Tensor x);   // forward function
  
  private:
    torch::nn::Conv2d conv_first{nullptr};
    torch::nn::Sequential feature_extraction{nullptr};

    // extract pyramid features
    torch::nn::Conv2d conv_l2_1{nullptr};
    torch::nn::Conv2d conv_l2_2{nullptr};
    torch::nn::Conv2d conv_l3_1{nullptr};
    torch::nn::Conv2d conv_l3_2{nullptr};

    // reconstruction
    torch::nn::Sequential reconstruction{nullptr};

    // TSA module
    torch::nn::Sequential TSAFusion{nullptr};

    // upsample
    torch::nn::Conv2d upconv1{nullptr}, upconv2{nullptr}, conv_hr{nullptr}, conv_last{nullptr};
    torch::nn::PixelShuffle pixel_shuffle{nullptr};

    // activate functon
    torch::nn::LeakyReLU lrelu{nullptr};
};

// EDVR init function
EDVRNet::EDVRNet(int in_channels, int  mid_channels) {
  conv_first = register_module("conv_first", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, mid_channels, {3,3}).stride(1).padding(1).bias(true)));

  feature_extraction = make_layer(5, mid_channels); // num_blocks = 5, mid_channels = mid_channels
  feature_extraction = register_module("feature_extraction", feature_extraction);

  // extract pyramid features
  conv_l2_1 = register_module("conv_l2_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels, mid_channels, {3,3}).stride(2).padding(1).bias(true)));
  conv_l2_2 = register_module("conv_l2_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels, mid_channels, {3,3}).stride(1).padding(1).bias(true)));
  conv_l3_1 = register_module("conv_l3_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels, mid_channels, {3,3}).stride(2).padding(1).bias(true)));
  conv_l3_2 = register_module("conv_l3_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels, mid_channels, {3,3}).stride(1).padding(1).bias(true)));

  // TSA module
  TSAFusion = tsa_fusion(mid_channels, 5, 2);
  TSAFusion = register_module("TSAFusion", TSAFusion);

  // reconstruction
  reconstruction = make_layer(10, mid_channels);  // num_blocks = 5, mid_channels = mid_channels
  pixel_shuffle = register_module("pixel_shuffle", torch::nn::PixelShuffle(torch::nn::PixelShuffleOptions(2)));

  // activate function
  lrelu = register_module("lrelu", torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
}

// define spatial padding functon
torch::Tensor EDVRNet::spatial_padding(torch::Tensor lrs) {
  auto b = lrs.sizes()[0]; auto t = lrs.sizes()[1]; auto c = lrs.sizes()[2]; auto h = lrs.sizes()[3]; auto w = lrs.sizes()[4];
  auto pad_h = (4 - h % 4) % 4;
  auto pad_w = (4 - w % 4) % 4;

  lrs = lrs.view({-1, c, h, w});
  lrs = torch::nn::functional::pad(lrs, torch::nn::functional::PadFuncOptions({0, pad_h, 0, pad_w}).mode(torch::kReflect));

  return lrs.view({b, t, c, h+pad_h, w+pad_w});
}

// forward function
torch::Tensor EDVRNet::forward(torch::Tensor x) {
  auto b = x.sizes()[0]; auto t = x.sizes()[1]; auto c = x.sizes()[2]; auto h_input = x.sizes()[3]; auto w_input = x.sizes()[4];
  
  // padding lrs
  x = spatial_padding(x);

  auto h = x.sizes()[3]; auto w = x.sizes()[4];

  int x_center_frame_index = 2;

  auto x_center = x.select(1, x_center_frame_index).contiguous();

  auto feat_l1 = lrelu->forward(conv_first->forward(x.view({-1, c, h, w})));
  feat_l1 = feature_extraction->forward(feat_l1);

  // L2
  auto feat_l2 = lrelu->forward(conv_l2_1->forward(feat_l1));
  feat_l2 = lrelu->forward(conv_l2_2->forward(feat_l2));

  // L3
  auto feat_l3 = lrelu->forward(conv_l3_1->forward(feat_l2));
  feat_l3 = lrelu->forward(conv_l3_2->forward(feat_l3));

  feat_l1 = feat_l1.view({b, t, -1, h, w});
  feat_l2 = feat_l2.view({b, t, -1, h / 2, w / 2});
  feat_l3 = feat_l3.view({b, t, -1, h / 4, w / 4});

  cout << "feat l1 size: " << feat_l1.sizes() << endl;
  cout << "feat l2 size: " << feat_l2.sizes() << endl;
  cout << "feat l3 size: " << feat_l3.sizes() << endl;

  return x;
}