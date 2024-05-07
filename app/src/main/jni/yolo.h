#ifndef YOLO_H
#define YOLO_H

#include <opencv2/core/core.hpp>

#include <net.h>
#include "benchmark.h"


#include "gpu.h"
#include "net.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#define num_featuremap 4
#define hard_nms 1
#define blending_nms 2 /* mix nms was been proposaled in paper blaze face, aims to minimize the temporal jitter*/

struct Object
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
};


struct  return_d_m //关键点前处理
{
    cv::Mat dst;
    cv::Mat matri;
};


class SSD
{
public:
    SSD();

    int load(const char* modeltype, int target_size,  const float* norm_vals, bool use_gpu = false);

    int load(AAssetManager* mgr, const char* modeltype, int target_size, const float* norm_vals, bool use_gpu = false);

    int detect(const cv::Mat& rgb, std::vector<Object>& objects, std::vector<Object>& face_list, std::vector<cv::Mat>& facelandmarks,float prob_threshold = 0.7f, float nms_threshold = 0.3f);

    int draw(cv::Mat& rgb, const std::vector<Object>& face_list,std::vector<cv::Mat>& facelandmarks);


private:
    void generateBBox(std::vector<Object> &bbox_collection, ncnn::Mat scores, ncnn::Mat boxes, float score_threshold, int num_anchors);

    void nms(std::vector<Object> &input, std::vector<Object> &output, float nms_threshold,int type = blending_nms);

private:

    ncnn::Net facedetector;
    ncnn::Net landmarks; //声明关键点模型

    int image_w;
    int image_h;

    int in_w =320;
    int in_h =240;
    int num_anchors;

 //   int topk = -1;
//    float score_threshold = 0.7;
//    float iou_threshold = 0.3;


    float mean_vals[3];
    float norm_vals[3];

    const float center_variance = 0.1;
    const float size_variance = 0.2;
    const std::vector<std::vector<float>> min_boxes = {
            {10.0f,  16.0f,  24.0f},
            {32.0f,  48.0f},
            {64.0f,  96.0f},
            {128.0f, 192.0f, 256.0f}};
    const std::vector<float> strides = {8.0, 16.0, 32.0, 64.0};
    std::vector<std::vector<float>> featuremap_size;
    std::vector<std::vector<float>> shrinkage_size;
    std::vector<int> w_h_list;

    std::vector<std::vector<float>> priors = {};

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif // YOLO_H
