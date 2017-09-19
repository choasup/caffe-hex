//
//  HEXGraph.hpp
//  HEXGraph
//
//  Created by LY on 17/7/7.
//  Copyright © 2017 LY. All rights reserved.
//

#ifndef HEXGraph_hpp
#define HEXGraph_hpp

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <set>
#include <math.h>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class HexGraph {

public:
    string GFile;
    string labelNameFile;
    int nums;
    vector<vector<int> > G;
    vector<string> labelName;
    set<vector<int> > states; //States
    
    vector<Dtype> pNode;
    vector<Dtype> score;

    vector<vector<Dtype> > pStates;
    int leaf_nums; 
    vector<int> leaf;

public:
    HexGraph();
    HexGraph(string file, string labelfile, int n);
    void Init();
    void ProduceG();
    void ListStatesSpace();
    void Search(int index, int value, vector<int>& res);
    void Forward();
    Dtype CalNode(int k);
    //Blob<Dtype> CalLabels(const vector<Blob<Dtype>*>& bottom);
    void CalLabels(const vector<Blob<Dtype>*>& bottom, Blob<Dtype> &node);	
    void CalProbs(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);  
    void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    void Backward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    //void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
};
}
#endif /* HEXGraph_hpp */
