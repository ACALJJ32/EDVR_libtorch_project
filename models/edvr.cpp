#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <iomanip>
#include <stdio.h> 
#include <vector>
// #include <src/deform_conv_ext.cpp> // error occured!
#include "dcn/src/deform_conv_ext.cpp"
using namespace std;


class ResidualBlockNoBN: public torch::nn::Module {
  public:
    ResidualBlockNoBN(int mid_channels);
    torch::Tensor forward(torch::Tensor x);
  
  private:
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


// DCN block
class ModulatedDeformConvPack: public torch::nn::Module {
  public:
    ModulatedDeformConvPack(int in_channels, int out_channels, int kernel_size, int stride, int padding, int dilation, int groups, int deformable_groups);
    torch::Tensor forward(torch::Tensor x, torch::Tensor feat);
  private:
    torch::nn::Conv2d conv_offset{nullptr};
};


ModulatedDeformConvPack::ModulatedDeformConvPack(int in_channels, int out_channels, int kernel_size, int stride, int padding, int dilation, int groups, int deformable_groups){
  // init parameters
  conv_offset = register_module("conv_offset", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, deformable_groups * 3 * kernel_size * kernel_size, {kernel_size, kernel_size}).stride(stride).padding(padding).dilation(dilation).bias(true)));
};


torch::Tensor ModulatedDeformConvPack::forward(torch::Tensor x, torch::Tensor feat) {
  // auto out = conv_offset->forward(x);
  // auto o1, o2, mask = torch::chunk(out, 3, 1);

  cout << "ModulatedDeformConvPack testing..." << endl;
  cout << x.sizes() << endl;
  return x;
}


torch::nn::Sequential DCNv2Pack(
  int in_channels, int out_channels, int kernel_size, int stride, int padding, int dilation, int groups, int deformable_groups) {

  torch::nn::Sequential features;
  features->push_back(ModulatedDeformConvPack(
    in_channels, out_channels, kernel_size, stride, padding, dilation, groups, deformable_groups
  ));
  return features;
}


// make residual blocks
torch::nn::Sequential make_layer(int num_blocks, int mid_channels) {
  torch::nn::Sequential features;
  
  for(int i = 0; i < num_blocks; i++) {
    features->push_back(ResidualBlockNoBN(mid_channels));
  }

  return features;
};


// TSA Fusion module
class TSAFusion: public torch::nn::Module {
  public:
    TSAFusion(int mid_channels, int num_frame, int center_frame_idx);
    torch::Tensor forward(torch::Tensor aligned_feat);
  
  private:
    int64_t center_frame_idx; 
    // temporal attention (before fusion conv)
    torch::nn::Conv2d temporal_attn1{nullptr}, temporal_attn2{nullptr};
    torch::nn::Conv2d feat_fusion{nullptr};
  
    // spatial attention (after fusion conv)
    torch::nn::MaxPool2d max_pool{nullptr};  
    torch::nn::AvgPool2d avg_pool{nullptr};
    torch::nn::Conv2d spatial_attn1{nullptr}, spatial_attn2{nullptr}, spatial_attn3{nullptr}, spatial_attn4{nullptr}, spatial_attn5{nullptr};
    torch::nn::Conv2d spatial_attn_l1{nullptr}, spatial_attn_l2{nullptr}, spatial_attn_l3{nullptr};
    torch::nn::Conv2d spatial_attn_add1{nullptr}, spatial_attn_add2{nullptr};

    // activate function
    torch::nn::LeakyReLU lrelu{nullptr};

    // upsample
    torch::nn::Upsample upsample{nullptr};
};


TSAFusion::TSAFusion(int mid_channels, int num_frame, int center_frame_idx){
  center_frame_idx = center_frame_idx;

  // temporal attention (before fusion conv)
  temporal_attn1 = register_module("temporal_attn1",\
   torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels, mid_channels, {3,3}).stride(1).padding(1).bias(true)));
  temporal_attn2 = register_module("temporal_attn2",\
   torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels, mid_channels, {3,3}).stride(1).padding(1).bias(true)));
  feat_fusion = register_module("feat_fusion",\
   torch::nn::Conv2d(torch::nn::Conv2dOptions(num_frame * mid_channels, mid_channels, {1,1}).stride(1).bias(true)));

  // spatial attention (after fusion conv)
  max_pool = register_module("max_pool", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({3,3}).stride(2).padding(1)));
  avg_pool = register_module("avg_pool",torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({3,3}).stride(2).padding(1)));
  spatial_attn1 = register_module("spatial_attn1",\
   torch::nn::Conv2d(torch::nn::Conv2dOptions(num_frame * mid_channels, mid_channels, {1,1})));
  spatial_attn2 = register_module("spatial_attn2",\
   torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels * 2, mid_channels, {1,1})));
  spatial_attn3 = register_module("spatial_attn3",\
   torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels, mid_channels, {3,3}).stride(1).padding(1)));
  spatial_attn4 = register_module("spatial_attn4",\
   torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels, mid_channels, {1,1})));
  spatial_attn5 = register_module("spatial_attn5",\
   torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels, mid_channels, {3,3}).stride(1).padding(1)));
  
  spatial_attn_l1 = register_module("spatial_attn_l1",\
   torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels, mid_channels, {1,1})));
  spatial_attn_l2 = register_module("spatial_attn_l2",\
   torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels * 2, mid_channels, {3,3}).stride(1).padding(1)));
  spatial_attn_l3 = register_module("spatial_attn_l3",\
   torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels, mid_channels, {3,3}).stride(1).padding(1)));
  spatial_attn_add1 = register_module("spatial_attn_add1",\
   torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels, mid_channels, {1,1})));
  spatial_attn_add2 = register_module("spatial_attn_add2",\
   torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels, mid_channels, {1,1})));

  // activate function
  lrelu = register_module("lrelu", torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1).inplace(true)));

  // umsample function
  upsample = register_module("upsample",\
   torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double> ({2,2})).mode(torch::kBilinear).align_corners(false)));
};


torch::Tensor TSAFusion::forward(torch::Tensor aligned_feat) {
  cout << "TSA testing..." <<endl;

  auto b = aligned_feat.sizes()[0]; auto t = aligned_feat.sizes()[1]; auto c = aligned_feat.sizes()[2]; 
  auto h = aligned_feat.sizes()[3]; auto w = aligned_feat.sizes()[4];

  // temporal attention 
  auto embedding_ref = temporal_attn1->forward(aligned_feat.select(1, 2).clone());  // center_frame_index: default 2
  auto embedding = temporal_attn2->forward(aligned_feat.view({{-1, c, h, w}}));
  embedding = embedding.view({b, t, -1, h, w});

  vector<torch::Tensor> corr_l;
  for(int i = 0; i < t; i++) {
    auto emb_neighbor = embedding.select(1,i);  // (b, mid_channels, h, w)
    auto corr = torch::sum(emb_neighbor * embedding_ref, 1);
    corr_l.push_back(corr.unsqueeze(1));
  }

  auto corr_prob = torch::sigmoid(torch::cat(corr_l, 1));  // (b, t, h, w)
  corr_prob = corr_prob.unsqueeze(2).expand({b, t, c, h, w});
  corr_prob = corr_prob.contiguous().view({b, -1, h, w});

  aligned_feat = aligned_feat.view({b, -1, h, w}) * corr_prob;

  // fusion
  auto feat = lrelu->forward(feat_fusion->forward(aligned_feat));

  // spatial attention
  auto attn = lrelu->forward(spatial_attn1->forward(aligned_feat));
  auto attn_max = max_pool->forward(attn);
  auto attn_avg = avg_pool->forward(attn);
  attn = lrelu->forward(spatial_attn2->forward(torch::cat({attn_max, attn_avg}, 1)));

  // pyramid levels TODO
  auto attn_level = lrelu->forward(spatial_attn_l1->forward(attn));
  attn_max = max_pool->forward(attn_level);
  attn_avg = avg_pool->forward(attn_level);
  attn_level = lrelu->forward(spatial_attn_l2->forward(torch::cat({attn_max, attn_avg}, 1)));
  attn_level = lrelu->forward(spatial_attn_l3->forward(attn_level));
  cout << attn_level.sizes() << endl;
  attn_level = upsample->forward(attn_level);  // bug occured

  attn = lrelu->forward(spatial_attn3->forward(attn)) + attn_level;
  attn = lrelu->forward(spatial_attn4->forward(attn));
  attn = upsample->forward(attn);
  auto attn_add = spatial_attn_add2(lrelu->forward(spatial_attn_add1->forward(attn)));
  attn = torch::sigmoid(attn);

  // after initialization, *2 makes (attn * 2) to be close to 1.
  feat = feat * attn * 2 + attn_add;
  return feat;
}


torch::nn::Sequential TSAFusionSequential(int mid_channels, int num_frame, int center_frame_idx) {
  torch::nn::Sequential features;
  features->push_back(TSAFusion(mid_channels,num_frame,center_frame_idx));
  return features;
};


// EDVRNet
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

    // PCD module  
    // TODO
    torch::nn::Sequential PCDAlignmentModule{nullptr};

    // TSA module
    torch::nn::Sequential TSAFusionModule{nullptr};

    // upsample
    torch::nn::Conv2d upconv1{nullptr}, upconv2{nullptr}, conv_hr{nullptr}, conv_last{nullptr};
    torch::nn::PixelShuffle pixel_shuffle{nullptr};

    // activate functon
    torch::nn::LeakyReLU lrelu{nullptr};
};


// EDVR init function
EDVRNet::EDVRNet(int in_channels, int  mid_channels) {
  conv_first = register_module("conv_first",\
   torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, mid_channels, {3,3}).stride(1).padding(1).bias(true)));

  feature_extraction = make_layer(5, mid_channels); // num_blocks = 5, mid_channels = mid_channels
  feature_extraction = register_module("feature_extraction", feature_extraction);

  // extract pyramid features
  conv_l2_1 = register_module("conv_l2_1",\
   torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels, mid_channels, {3,3}).stride(2).padding(1).bias(true)));
  conv_l2_2 = register_module("conv_l2_2",\
   torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels, mid_channels, {3,3}).stride(1).padding(1).bias(true)));
  conv_l3_1 = register_module("conv_l3_1",\
   torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels, mid_channels, {3,3}).stride(2).padding(1).bias(true)));
  conv_l3_2 = register_module("conv_l3_2",\
   torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels, mid_channels, {3,3}).stride(1).padding(1).bias(true)));

  //PCD module
  PCDAlignmentModule = DCNv2Pack(
    mid_channels, mid_channels, 3, 1, 1, 1, 1, 8);

  PCDAlignmentModule = register_module("PCDAlignmentModule", PCDAlignmentModule);  

  // TSA module
  TSAFusionModule = TSAFusionSequential(mid_channels, 5, 2);
  TSAFusionModule = register_module("TSAFusionModule", TSAFusionModule);

  // reconstruction
  reconstruction = make_layer(10, mid_channels);  // num_blocks = 5, mid_channels = mid_channels
  reconstruction = register_module("reconstruction", reconstruction);
  pixel_shuffle = register_module("pixel_shuffle", torch::nn::PixelShuffle(torch::nn::PixelShuffleOptions(2)));

  // upsample
  upconv1 = register_module("upconv1",\
   torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels, mid_channels * 4, {3,3}).stride(1).padding(1).bias(true)));

  upconv2 = register_module("upconv2",\
   torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels, 64 * 4, {3,3}).stride(1).padding(1).bias(true)));

  conv_hr = register_module("conv_hr",\
   torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, {3,3}).stride(1).padding(1).bias(true)));

  conv_last = register_module("conv_last", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 3, {3,3}).stride(1).padding(1).bias(true)));

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

  // PCD alignment
  // TODO

  // auto out = PCDAli->forward(feat_l1, feat_l1);

  // TSA module 
  auto feat = TSAFusionModule->forward(feat_l1);

  // Reconstruction module
  auto out = reconstruction->forward(feat);
  out = lrelu->forward(pixel_shuffle->forward(upconv1->forward(out)));
  out = lrelu->forward(pixel_shuffle->forward(upconv2->forward(out)));
  out = lrelu->forward(conv_hr->forward(out));
  out = conv_last->forward(out);

  auto base = torch::nn::functional::interpolate(x_center,\
   torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({h * 4, w * 4})).mode(torch::kBilinear).align_corners(false));
  out += base;
  
  return out;
}