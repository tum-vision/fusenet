/* Author: Lingni Ma
 * Created Date: 12 Jan 2016
 * Last Edit: 12 Jan 2016
 * Function: memorized unpooling
 * Note: implementation is based on the UpsampleLayer from SegNet (https://github.com/alexgkendall/caffe-segnet)
 */

#ifndef CAFFE_UNPOOLING_LAYER_HPP_
#define CAFFE_UNPOOLING_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{

template <typename Dtype>
class UnpoolingLayer : public Layer<Dtype> {
 public:
  explicit UnpoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Unpooling"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  static const int DataBlobIndex = 0;
  static const int MaskBlobIndex = 1;

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int channels_;
  int height_;
  int width_;
  int output_h_, output_w_;
  int scale_;
};


}




#endif
