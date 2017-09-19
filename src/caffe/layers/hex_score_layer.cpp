#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/hex_score_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void HexScoreLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  HexScoreParameter param = this->layer_param_.hex_score_param();
  G_file_ = param.g_file();
  label_name_file_ = param.label_name_file();
  node_nums_ = param.node_nums();
  hex_ = HexGraph<Dtype>(G_file_, label_name_file_, node_nums_);
  hex_.Init();
  hex_.ProduceG();
}

template <typename Dtype>
void HexScoreLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);  
  CHECK_EQ(node_nums_, bottom[0]->channels());
}

template <typename Dtype>
void HexScoreLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  hex_.CalProbs(bottom, top); 
}

INSTANTIATE_CLASS(HexScoreLayer);
REGISTER_LAYER_CLASS(HexScore);

}  // namespace caffe
