//
//  HEXGraph.cpp
//  HEXGraph
//
//  Created by LY on 17/7/7.
//  Copyright © 2017年 LY. All rights reserved.
//

#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>

//#include "caffe/layers/region_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/util/HEXGraph.hpp"
//#include <boost/regex.hpp>

namespace caffe {

using std::ifstream;
using std::cout;
using std::endl;
using std::max;

template <typename Dtype>
HexGraph<Dtype>::HexGraph() {
}

template <typename Dtype>
HexGraph<Dtype>::HexGraph(string file, string labelfile, int n) {
    GFile = file;
    labelNameFile = labelfile;
    nums = n;
}

template <typename Dtype>
void HexGraph<Dtype>::Init() {
    
    ifstream fin(labelNameFile.data());
    
    if (!fin) {
	LOG(ERROR) << labelNameFile << " :Label File Not Exists.";
	return;
    }

    string str;
    
    //const regex pattern("(\\d+) (\\w+)");
    //match_results<string::const_iterator> result;
    
    while (getline(fin, str)) {   //init G.
        //bool valid = regex_match(s, result,pattern);
        
        //if(valid&&(result.length()>0))
        //{
            //int index = (int)atoi(((string)result[1]).data());
        //    string label = (string)result[2];
        //    labelName.push_back(label);
        //}
        if (str == "")
	    continue;
	int pos = (int)str.find(" ", 0);
	labelName.push_back(str.substr(pos + 1, str.length()));
    }
    
    for (int i = 0; i < labelName.size(); i ++)
	LOG(INFO) << labelName[i]; 
    //for (auto x: labelName)
    //    cout << x << endl;
    
    for (int i = 0; i < nums; i ++)
        pNode.push_back(-1);
}

template <typename Dtype>
void HexGraph<Dtype>::ProduceG() {
    ifstream fin(GFile.data());
    
    if (!fin) {
	LOG(ERROR) << GFile << " :Graph File Not Exists.";
   	return;
    }

    string str;
    
    //const regex pattern("(\\d+) (\\d+)");
    //match_results<string::const_iterator> result;
    
    for (int i = 0; i < nums; i ++) {
        vector<int> p(nums, 0);
        G.push_back(p);
    }
    
    int maxN = 0;
    
    cout << "Produce HEXGraph:"<<endl;
    while (getline(fin, str)) {   //init G.
        //bool valid = regex_match(s, result,pattern);
        
        //if(valid&&(result.length()>0))
        //{
        //    int s = (int)atoi(((string)result[1]).data());
        //    int e = (int)atoi(((string)result[2]).data());
        //    G[s][e] = 1;
            
        //    if (maxN < s || maxN < e)
        //        maxN = max(s, e);
        //}
        if (str == "") 
	    continue;
	int pos = (int)str.find(" ", 0);
	int s = (int)atoi((str.substr(0, pos)).data());
	int e = (int)atoi((str.substr(pos + 1, str.length())).data());
	G[s][e] = 1;
	if (maxN < s || maxN < e)
	    maxN = max(s, e);    
    }
    
    LOG(INFO) << "Init G:";
    for (int i = 0; i < nums; i ++) {
	for (int j = 0; j < nums; j ++)
		cout << G[i][j] << " ";
	cout << endl;
    }

    if (maxN + 1 != nums)
        cout << "Nums Error!" << endl;
    
    vector<vector<int> > G_next = G;
    int rowA = (int)G.size();
    int colA = (int)G[0].size();
    //int rowB = (int)G.size();
    int colB = (int)G[0].size();
    
    while (1) {                         //nodes are arrived. G
        for (int i = 0; i < rowA; ++i) {
            for (int j = 0; j < colB; ++j) {
                for (int k = 0; k < colA; ++k) {
                    G_next[i][j] += (G[i][k] * G[k][j]);
                }
                if (G_next[i][j] > 1)
                    G_next[i][j] = 1;
            }
        }
        
        if (G == G_next)
            break;
        
        G = G_next;
    }
	
    LOG(INFO) << "Arrived G:";
    for (int i = 0; i < nums; i ++) {
	for (int j = 0; j < nums; j ++)
		cout << G[i][j] << " ";
	cout << endl;
    }
        
    for (int i = 0; i < nums; i ++)
	for (int j = 0; j < nums; j ++)
	    if (G[i][j] == 1)
		G[j][i] = 3;	//father -> child
 
    for (int j = 0; j < nums; j ++) {   //visiting by col is slow.
        vector<int> col;
        col.clear();
        for (int i = 0; i < nums; i ++)
            if (G[i][j] == 1)
                col.push_back(i);
        //LOG(INFO) << j << ":";
	//for (int i = 0; i < col.size(); i ++)
	//	cout << col[i] << " ";
	//cout << endl;
	
	if (col.size() <= 1)
		continue;

        for (int l = 0; l < col.size() - 1; l ++)
            for (int r = l + 1; r < col.size(); r ++) {
                int s = col[l];
                int e = col[r];
                
                if (G[s][e] == 0 && G[s][e] == 0) {     //overlap
                    G[s][e] = 2;
                    G[e][s] = 2;
                }
            }
    }

    //LOG(INFO) << "Overlap G:";
    //for (int i = 0; i < nums; i ++) {
    //	for (int j = 0; j < nums; j ++)
//		cout << G[i][j] << " ";
//	cout << endl;
//    }
    

    for (int i = 0; i < nums; i ++)             //exclusion
        for (int j = i + 1; j < nums; j ++)
            if (G[i][j] == 0 && G[j][i] == 0) {
                G[i][j] = -1;
                G[j][i] = -1;
            }
	
    LOG(INFO) << "Complete G:";
    for (int i = 0; i < nums; i ++) {
	for (int j = 0; j < nums; j ++)
		cout << G[i][j] << " ";
	cout << endl;
    }
    //for (auto x: G) {
    //   for (auto y: x)
    //        cout << y << " ";
    //    cout << endl;
    //}
   leaf_nums = 0;
   for (int i = 0; i < nums; i ++) {
	int flag = 0;
	for (int j = 0; j < nums; j ++)
		if (G[i][j] == 1) {
			flag = 1;			     
			break;
		}
	if (flag == 0) {
	    leaf.push_back(i);
	    leaf_nums += 1;
	}
   }
}

template <typename Dtype>
void HexGraph<Dtype>::ListStatesSpace() {
    cout << "Search States:" << endl;
    
    vector<int> p(nums, 0);
    for (int k = 0; k < nums; k ++) {
        Search(0, k, p);
    }
    
    //for (auto x: states) {
    set<vector<int> >::iterator iter;
    for (iter = states.begin(); iter != states.end(); iter ++) {
        vector<int> x = *iter;
        for (int i = 0; i < x.size(); i ++)
            if (x[i] == 1)
                cout << labelName[i] << " ";
        
        cout << endl;
    }
            
}

template <typename Dtype>
void HexGraph<Dtype>::Search(int index, int k, vector<int>& res) {
    if (k == 0) {
        for (int i = 0; i < nums; i ++) {
            if (res[i] == 1) {
                for (int j = 0; j < nums; j ++) {
                    if (G[i][j] == 3) // i -> j
                        res[j] = 1;
                }
            }
        }
        
        states.insert(res);
        
        return;
    }
    
    for (int i = index; i < nums; i ++) {
        if (res[i] != 1) {
            vector<int> p = res;
            res[i] = 1;
            
            int end = 0;
            for (int j = 0; j < nums; j ++) {
                if (res[j] == 1 && G[i][j] == -1) {
                    end = 1;
                    break;
                }
            }
            
            if (end == 0)
                Search(i + 1, k - 1, res);
            
            res = p;
        }
    }
}

template <typename Dtype>
void HexGraph<Dtype>::Forward() {
    for (int i = 0; i < nums; i ++)
        score.push_back(1.0);
    
    vector<float> ps;
    
    float sum = 0;
    int index = -1;
    //for (auto x: states){
    set<vector<int> >::iterator iter;
    for (iter = states.begin(); iter != states.end(); iter ++) {
        vector<int> x = *iter;
        float py = 1;
        index ++;
        if (index == 0)
            continue;
        
        for (int i = 0; i < x.size(); i ++) {
            if (x[i] == 1)
                py *= exp(score[i]);
        }
        sum += py;
        ps.push_back(py);
    }
    
    //for (auto z: ps)
    //    cout << z << " ";
    //cout << endl;
    
    //cout << endl;
    //cout << "results:" << ps.size() << endl;
    //index = -1;
    //for (auto x: states) {
    //    index ++;
    //    if (index == 0)
    //        continue;
        
    //    for (int i = 0; i < x.size(); i ++)
    //        if (x[i] == 1)
    //            cout << labelName[i] << " ";
    //    
    //    cout << ps[index - 1] / sum << endl;
    //}    
}

template <typename Dtype>
void HexGraph<Dtype>::CalProbs(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    int num = bottom[0]->num();
    int channels = bottom[0]->channels();    
    vector<int> shape(2, 1);
    //leaf_nums = nums;

    top[0]->Reshape(shape); // n, c    
    
    Dtype* top_data = top[0]->mutable_cpu_data();
    caffe_set(top[0]->count(), Dtype(0.0), top_data);
    
    //score.clear();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    //LOG(INFO) << "num" << num;

    for (int n = 0; n < num; n ++) {
	score.clear();
	pNode.clear();	
	for (int c = 0; c < channels; c ++) {
	    score.push_back(bottom_data[n * channels + c]);
	    pNode.push_back(-1);
	}
	for (int i = 0; i < nums; i ++) {
	    if (pNode[i] == -1)
		pNode[i] = CalNode(i);
	}

	for (int i = 0; i < nums; i ++)
	    top_data[n * nums + i] = pNode[i];	
    }
    return;
}

template <typename Dtype>
void HexGraph<Dtype>::CalLabels(const vector<Blob<Dtype>*>& bottom, Blob<Dtype> &node) {
    int num = bottom[0]->num();
    int channels = bottom[0]->channels();    
    vector<int> shape(2, 1);
    //leaf_nums = nums;

    shape[0] = num;
    shape[1] = leaf_nums;
    //cout << "leafs:" << leaf_nums << endl;;
    //_hex_Graph
    node.Reshape(shape); // n, c    
    
    Dtype* node_data = node.mutable_cpu_data();
    caffe_set(node.count(), Dtype(0.0), node_data);
    
    //score.clear();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    //const Dtype* label_data = bottom[1]->cpu_data();
 	
    //LOG(INFO) << "num" << num;
    for (int n = 0; n < num; n ++) {
	score.clear();
	pNode.clear();
        //cout << "n:" << n << endl;   
        //cout << "score:";	
	for (int c = 0; c < channels; c ++) {
	    score.push_back(bottom_data[n * channels + c]);
	    pNode.push_back(-1);
	    //cout << score[c] << " ";
	}
	//cout << endl;
	//cout << "Label:" << label_data[n] << endl;
	//cout << "Nodes:";
	for (int i = 0; i < nums; i ++) {
	    if (pNode[i] == -1)
		pNode[i] = CalNode(i);
	    //cout << pNode[i] << " ";
	}

	for (int i = 0; i < leaf_nums; i ++)
	    node_data[n * leaf_nums + i] = pNode[i];	
	//cout << endl;	
    }

   //LOG(INFO) << "Done.";
   // float sum = 0;
   // for (int i = 0; i < nums; i ++) {
   //     if (pNode[i] == -1)
   //         pNode[i] = CalNode(i);
   //	node_data[i] = pNode[i];        
   // 	sum += pNode[i];	
   // }
    //for (int i = 0; i < nums; i ++)
    //    cout << labelName[i] << " " << pNode[i] << endl;
    return;
}

template <typename Dtype> 
Dtype HexGraph<Dtype>::CalNode(int k) {
    vector<int> child;
    vector<int> father;
    Dtype p = 0;

    for (int j = 0; j < nums; j ++)
        if (G[k][j] == 1)
            child.push_back(j);
    
    for (int j = 0; j < nums; j ++)
        if (G[j][k] == 1)
            father.push_back(j);
    
    Dtype s = score[k];
    for (int i = 0; i < father.size(); i ++)
        s += score[father[i]];
    
    p += exp(s);

    for (int i = 0; i < child.size(); i ++)
        p += CalNode(child[i]);
    
    pNode[k] = p;
    return p;
}

template <typename Dtype>
void HexGraph<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    //LOG(INFO) << "Forward Hex";
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* label_data = bottom[1]->cpu_data();
    //LOG(INFO) << "?????";    
    //Dtype *top_data = top[0]->mutable_cpu_data();
    //caffe_copy(1, bottom_data, top_data);
    //LOG(INFO) << "TOP?"; 
    //LOG(INFO) << "data:" << bottom[0]->shape_string() << endl;
    //LOG(INFO) << "label:" << bottom[1]->shape_string() << endl;
    //LOG(INFO) << "1";
    int num = bottom[0]->num();
    int channels = bottom[0]->channels();
    //LOG(INFO) << "2";
    //Dtype loss(1.0);
    //LOG(INFO) << "1";
    //Dtype *data = bottom[0]->mutable_cpu_data();
    //for (int n = 0; n < num; n ++)
    //	for (int c = 0; c < channels; c ++)
    //	    data[n * channels + c] = 0.01 * (c + 1);
    
/*    cout << "data:";

    for (int n = 0; n < 1; n ++)
    	for (int c = 0; c < channels; c ++)
    	    cout << bottom[0]->data_at(n, c, 0, 0) << " ";
    cout << endl; 
    LOG(INFO) << "label:"<< bottom[1]->data_at(0,0,0,0); 
*/    
    pStates.clear();
    for (int n = 0; n < num; n ++) { //batch_size
	score.clear();
        Dtype sum(0.0);
        vector<Dtype> ps;
	ps.clear();

	for (int c = 0; c < channels; c ++) {
	    score.push_back(bottom_data[n * channels + c]);
	}

    	set<vector<int> >::iterator iter;
    	
	for (iter = states.begin(); iter != states.end(); iter ++) {  //all states
            vector<int> x = *iter;
            Dtype py = 0.0;
	    	
            for (int i = 0; i < x.size(); i ++) {
                if (x[i] == 1)
                    py += (score[i]);
            }
            	
            ps.push_back(py); //p of every states.
        }

	Dtype max_p = ps[0];	
	for (int i = 0; i < ps.size(); i ++)
	    if (max_p < ps[i])
		max_p = ps[i];
	
	for (int i = 0; i < ps.size(); i ++) {
	    ps[i] = exp(ps[i] - max_p);	
	    sum += ps[i];
	}
	
	for (int i = 0; i < ps.size(); i ++)
	    ps[i] = ps[i] / sum;
        
/*	if (n == 0) {
	    cout << "f ps:";
	    for (int i = 0; i < ps.size(); i ++)
		cout << ps[i] << " ";
	    cout << endl;
	}
*/	   
	//for (int i = 0; i < ps.size(); i ++)
	//    ps[i] = ps[i] / sum;   // normalization
	pStates.push_back(ps);
    }
   	
   	
    //LOG(INFO) << "LOSS:";
    Dtype loss(0.0);

    for (int n = 0; n < num; n ++) {
	Dtype single_loss(0.0);
	vector<Dtype> ps = pStates[n];
	
	Dtype gt = label_data[n];	
	    
	set<vector<int> >::iterator iter;
	
	int index = 0;
        for (iter = states.begin(); iter != states.end(); iter ++) {  //all states
	    vector<int> x = *iter;
	
	    if (x[gt] == 1)
	        single_loss += ps[index];

	    index += 1;
	}
	
	single_loss = -1 * log(single_loss);
	loss += single_loss;
    }  
    
    //LOG(INFO) << "foward loss:" << loss / num;
    top[0]->mutable_cpu_data()[0] = loss / num;
    
    return;
}

template <typename Dtype>
void HexGraph<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
//    LOG(INFO) << "backward";
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* label_data = bottom[1]->cpu_data();
	
    caffe_copy(bottom[0]->count(), bottom_data, bottom_diff);
    
    //vector<Dtype> all_p(nums, 0.0);
    //vector<Dtype> gts_p(nums, 0.0);

    int num = bottom[0]->num();
    int channels = bottom[0]->channels();

/*    cout << "data:";

    for (int n = 0; n < 1; n ++)
        for (int c = 0; c < channels; c ++)
            cout << bottom[0]->data_at(n, c, 0, 0) << " ";
    cout << endl;
    LOG(INFO) << "label:"<< bottom[1]->data_at(0,0,0,0);
*/

    for (int n = 0; n < num; n ++) { //batch_size
  	vector<Dtype> all_p(nums, 0.0);
        vector<Dtype> gts_p(nums, 0.0);
	
	score.clear();
        Dtype sum(0.0);
        vector<Dtype> ps;
	ps.clear();

	Dtype gt = label_data[n];
	Dtype gts_sum(0.0);


	//========================================================================//
	for (int c = 0; c < channels; c ++) {
            score.push_back(bottom_data[n * channels + c]);
        }

        set<vector<int> >::iterator iter;

        for (iter = states.begin(); iter != states.end(); iter ++) {  //all states
            vector<int> x = *iter;
            Dtype py = 0.0;

            for (int i = 0; i < x.size(); i ++) {
                if (x[i] == 1)
                    py += (score[i]);
            }

            ps.push_back(py); //p of every states.
        }

        Dtype max_p = ps[0];
        for (int i = 0; i < ps.size(); i ++)
            if (max_p < ps[i])
                max_p = ps[i];

        for (int i = 0; i < ps.size(); i ++) {
            ps[i] = exp(ps[i] - max_p);
            sum += ps[i];
        }

        for (int i = 0; i < ps.size(); i ++)
            ps[i] = ps[i] / sum;


	//========================================================================//

        int index = 0;
	for (iter = states.begin(); iter != states.end(); iter ++) {  //all states
            vector<int> x = *iter;
		
            if (x[gt] == 1)
            	gts_sum += ps[index];
            
	    index ++;
            //sum += py;
        }
       
/*        if (n == 0) {	
           //cout << "gts_sum:"<< gts_sum << endl;	
	   cout << "b ps:";
  	   for (int i = 0; i < ps.size(); i ++)
	      cout << ps[i] << " ";
	   cout << endl;
	   cout << "gts_sum:" << gts_sum << endl;
	   cout << "diff:";
	}
*/
 	for (int i = 0; i < channels; i ++) {  // classes
	    set<vector<int> >::iterator iter;
	    int index = 0;
       	    for (iter = states.begin(); iter != states.end(); iter ++) {  //all states
                vector<int> x = *iter;
		
                if (x[i] == 1)
		    all_p[i] += ps[index];
		
		if (x[i] == 1 && x[gt] == 1)
		    gts_p[i] += ps[index];
	
		index ++;
            }
            
	    //if (n == 0) cout << (all_p[i] - gts_p[i] / gts_sum) << " ";	
	    bottom_diff[n * channels + i] = -1 * (gts_p[i] / gts_sum - all_p[i]); 	    
	}

/*	if (n == 0) {
           cout << endl;
           cout << "all_p:";
           for (int i = 0; i < channels; i ++)
	      cout << all_p[i] << " ";
    	
	   cout << endl;
 	   cout << "gts_p:";
	   for (int i = 0; i < channels; i ++)
	      cout << gts_p[i] / gts_sum << " ";
	   cout << endl;
           cout << endl;
	}
*/ 
    }
    Dtype loss_weight = top[0]->cpu_diff()[0] / num;
    caffe_scal(bottom[0]->count(), loss_weight, bottom_diff);
}

INSTANTIATE_CLASS(HexGraph);

}
