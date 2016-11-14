/* Author: Lingni Ma
 * Created Date: 12 Jan 2016
 * Last Edit: 12 Jan 2016
 * Function: memorized unpooling
 * Note: implementation is based on the UpsampleLayer from SegNet (https://github.com/alexgkendall/caffe-segnet)
 */

#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>

#include "caffe/layers/unpooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void UnpoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  UnpoolingParameter unpooling_param = this->layer_param_.unpooling_param();
  if (unpooling_param.has_output_h() && unpooling_param.has_output_w()){
    output_h_ = unpooling_param.output_h();
    output_w_ = unpooling_param.output_w();
    CHECK_GT(output_h_, 0);
    CHECK_GT(output_w_, 0);
  }
  else if (unpooling_param.has_scale())
  {
    CHECK_GT(scale_, 0);
    scale_ = unpooling_param.scale();
    output_h_ = bottom[DataBlobIndex]->height() * scale_;
    output_w_ = bottom[DataBlobIndex]->width() * scale_;
  }
  else{
    LOG(FATAL) << "require specify either (output_h, output_w) or scale";
  }
  CHECK_EQ(4, bottom[DataBlobIndex]->num_axes()) << "input must have 4 axes";
  CHECK_EQ(4, bottom[MaskBlobIndex]->num_axes()) << "input mask must have 4 axes";
  CHECK_EQ(bottom[DataBlobIndex]->num(), bottom[MaskBlobIndex]->num());
  CHECK_EQ(bottom[DataBlobIndex]->channels(), bottom[MaskBlobIndex]->channels());
  CHECK_EQ(bottom[DataBlobIndex]->height(), bottom[MaskBlobIndex]->height());
  CHECK_EQ(bottom[DataBlobIndex]->width(), bottom[MaskBlobIndex]->width());

}

template <typename Dtype>
void UnpoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[DataBlobIndex]->num(), bottom[DataBlobIndex]->channels(), output_h_, output_w_);
  channels_ = bottom[DataBlobIndex]->channels();
  height_ = bottom[DataBlobIndex]->height();
  width_ = bottom[DataBlobIndex]->width();
}

template <typename Dtype>
void UnpoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[DataBlobIndex]->cpu_data();
  const Dtype* bottom_mask_data = bottom[MaskBlobIndex]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  // Initialize
  const int top_count = top[0]->count();
  caffe_set(top_count, Dtype(0), top_data);
  // The main loop
  for (int n = 0; n < bottom[DataBlobIndex]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int i = 0; i < height_ * width_; ++i) {
        const int idx = static_cast<int>(bottom_mask_data[i]);
        if (idx >= output_h_ * output_w_) {
          // this can happen if the pooling layer that created the input mask
          // had an input with different size to top[0]
          LOG(FATAL) << "unpool top index " << idx << " out of range - "
            << "check scale settings match input pooling layer's "
            << "downsample setup";
        }
        top_data[idx] = bottom_data[i];
      }
      // compute offset
      bottom_data += bottom[DataBlobIndex]->offset(0, 1);
      bottom_mask_data += bottom[MaskBlobIndex]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
    }
  }
}

template <typename Dtype>
void UnpoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_mask_data = bottom[MaskBlobIndex]->cpu_data();
    Dtype* bottom_diff = bottom[DataBlobIndex]->mutable_cpu_diff();

    const int bottom_count = bottom[DataBlobIndex]->count();
    caffe_set(bottom_count, Dtype(0), bottom_diff);
    // The main loop
    for (int n = 0; n < bottom[DataBlobIndex]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int i = 0; i < height_ * width_; ++i) {
          const int idx = static_cast<int>(bottom_mask_data[i]);
          if (idx >= output_h_ * output_w_){
            // this can happen if the pooling layer that created
            // the input mask had an input with different size to top[0]
            LOG(FATAL) << "unpool top index " << idx << " out of range - "
              << "check scale settings match input pooling layer's downsample setup";
          }
          bottom_diff[i] = top_diff[idx];
        }
        // compute offset
        bottom_diff += bottom[DataBlobIndex]->offset(0, 1);
        bottom_mask_data += bottom[MaskBlobIndex]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(UnpoolingLayer);
#endif

INSTANTIATE_CLASS(UnpoolingLayer);
REGISTER_LAYER_CLASS(Unpooling);

}  // namespace caffe
