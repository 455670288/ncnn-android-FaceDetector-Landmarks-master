#include "yolo.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cpu.h"
#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))


SSD::SSD() {
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

//int SSD::load(const char *modeltype, int _target_size, const float *_norm_vals, bool use_gpu) {
//    mobilenetssd.clear();
//    blob_pool_allocator.clear();
//    workspace_pool_allocator.clear();
//
//    ncnn::set_cpu_powersave(2);
//    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());
//
//    mobilenetssd.opt = ncnn::Option();
//
//#if NCNN_VULKAN
//    mobilenetssd.opt.use_vulkan_compute = use_gpu;
//#endif
//    mobilenetssd.opt.num_threads = ncnn::get_big_cpu_count();
//    mobilenetssd.opt.blob_allocator = &blob_pool_allocator;
//    mobilenetssd.opt.workspace_allocator = &workspace_pool_allocator;
//
//    char parampath[256];
//    char modelpath[256];
//    sprintf(parampath, "%s.param", modeltype);
//    sprintf(modelpath, "%s.bin", modeltype);
//
//    mobilenetssd.load_param(parampath);
//    mobilenetssd.load_model(modelpath);
//
////    target_size = _target_size;
//    norm_vals[0] = _norm_vals[0];
//    norm_vals[1] = _norm_vals[1];
//    norm_vals[2] = _norm_vals[2];
//
//    return 0;
//}

int SSD::load(AAssetManager *mgr, const char *modeltype, int _target_size, const float *_norm_vals,
               bool use_gpu) {
    facedetector.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    facedetector.opt = ncnn::Option();
#if NCNN_VULKAN
    facedetector.opt.use_vulkan_compute = use_gpu;
#endif
//    facedetector.opt.use_vulkan_compute = false; //gpu推理
    facedetector.opt.num_threads = 1;
    facedetector.opt.blob_allocator = &blob_pool_allocator;
    facedetector.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    facedetector.load_param(mgr, parampath);
    facedetector.load_model(mgr, modelpath);

    // 加载关键点设置
    landmarks.opt = ncnn::Option();
    landmarks.opt.use_vulkan_compute = false; //gpu推理
    landmarks.opt.num_threads = 1;
    landmarks.load_param(mgr,"2d106det_change_fp16.param"); //加载关键点模型权重
    landmarks.load_model(mgr,"2d106det_change_fp16.bin");

    //初始化先验框
    w_h_list = {in_w, in_h};

    for (auto size : w_h_list) {
        std::vector<float> fm_item;
        for (float stride : strides) {
            fm_item.push_back(ceil(size / stride));
        }
        featuremap_size.push_back(fm_item);
    }

    for (auto size : w_h_list) {
        shrinkage_size.push_back(strides);
    }

    /* generate prior anchors */
    for (int index = 0; index < num_featuremap; index++) {
        float scale_w = in_w / shrinkage_size[0][index];
        float scale_h = in_h / shrinkage_size[1][index];
        for (int j = 0; j < featuremap_size[1][index]; j++) {
            for (int i = 0; i < featuremap_size[0][index]; i++) {
                float x_center = (i + 0.5) / scale_w;
                float y_center = (j + 0.5) / scale_h;

                for (float k : min_boxes[index]) {
                    float w = k / in_w;
                    float h = k / in_h;
                    priors.push_back({clip(x_center, 1), clip(y_center, 1), clip(w, 1), clip(h, 1)});
                }
            }
        }
    }
    num_anchors = priors.size();


//    target_size = _target_size;
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

/* 人脸关键点前处理*/
return_d_m pre_process(const cv::Mat& src, int input_size, Object& det){
    cv::Mat dst;
    return_d_m dst_m3;
    int x1 = det.x1;
    int y1 = det.y1;
    int faceImgWidth = det.x2 - det.x1;
    int faceImgHeight = det.y2 - det.y1;

    float center_w = x1+faceImgWidth/2;
    float center_h = y1+faceImgHeight/2;


    float _scale = input_size / (MAX(faceImgWidth, faceImgHeight) * 1.5);

    //构建仿射变换矩阵
    cv::Mat m1 = cv::Mat::eye(cv::Size(3,2),CV_32F);
    m1 *= _scale; //缩放矩阵
    cv::Mat m2 = cv::Mat::zeros(cv::Size(3,2),CV_32F);
    m2.at<float>(0,2) = -(center_w * _scale) + input_size/2;
    m2.at<float>(1,2) = -(center_h * _scale) + input_size/2; //平移矩阵
    dst_m3.matri = m1 + m2;
    cv::warpAffine(src, dst_m3.dst,  dst_m3.matri, cv::Size(192,192));

    return dst_m3;
}

/*人脸关键点后处理*/
cv::Mat post_progress (cv::Mat& post_mat, cv::Mat& m){
    cv::Mat im;
    cv::Mat pts(106, 2, CV_32F);
    cv::Mat new_pts;
    cv::Mat im_t;
    cv::Mat im_t_f;
    cv::Mat coord;
    cv::Mat col_cat = cv::Mat::ones(106,1,CV_32F);

    int index = 0;
    for (int i = 0; i < 106; ++i) {
        for (int j = 0; j < 2; ++j){
            pts.at<float>(i,j) = post_mat.at<float>(index,0);
            index ++;
        }
    }

    invertAffineTransform(m, im);
    pts += 1;
    pts *= 192/2;

    cv::hconcat(pts, col_cat, new_pts);

    transpose(im, im_t);
    im_t.convertTo(im_t_f,CV_32F);
    coord = new_pts * im_t_f;

    return coord;
}



void SSD::generateBBox(std::vector<Object> &objects, ncnn::Mat scores, ncnn::Mat boxes, float score_threshold, int num_anchors) {
    for (int i = 0; i < num_anchors; i++) {
        if (scores.channel(0)[i * 2 + 1] > score_threshold) {
            Object rects;
            float x_center = boxes.channel(0)[i * 4] * center_variance * priors[i][2] + priors[i][0];
            float y_center = boxes.channel(0)[i * 4 + 1] * center_variance * priors[i][3] + priors[i][1];
            float w = exp(boxes.channel(0)[i * 4 + 2] * size_variance) * priors[i][2];
            float h = exp(boxes.channel(0)[i * 4 + 3] * size_variance) * priors[i][3];
//            NCNN_LOGE("x_center = %f", x_center);

            rects.x1 = clip(x_center - w / 2.0, 1) * image_w;
            rects.y1 = clip(y_center - h / 2.0, 1) * image_h;
            rects.x2 = clip(x_center + w / 2.0, 1) * image_w;
            rects.y2 = clip(y_center + h / 2.0, 1) * image_h;
            rects.score = clip(scores.channel(0)[i * 2 + 1], 1);
            objects.push_back(rects);
        }
    }
}


void SSD::nms(std::vector<Object> &input, std::vector<Object> &output, float nms_threshold,int type) {
    std::sort(input.begin(), input.end(), [](const Object &a, const Object &b) { return a.score > b.score; });

    int box_num = input.size();

    std::vector<int> merged(box_num, 0);

    for (int i = 0; i < box_num; i++) {
        if (merged[i])
            continue;
        std::vector<Object> buf;

        buf.push_back(input[i]);
        merged[i] = 1;

        float h0 = input[i].y2 - input[i].y1 + 1;
        float w0 = input[i].x2 - input[i].x1 + 1;

        float area0 = h0 * w0;

        for (int j = i + 1; j < box_num; j++) {
            if (merged[j])
                continue;

            float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;
            float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

            float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;
            float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

            float inner_h = inner_y1 - inner_y0 + 1;
            float inner_w = inner_x1 - inner_x0 + 1;

            if (inner_h <= 0 || inner_w <= 0)
                continue;

            float inner_area = inner_h * inner_w;

            float h1 = input[j].y2 - input[j].y1 + 1;
            float w1 = input[j].x2 - input[j].x1 + 1;

            float area1 = h1 * w1;

            float score;

            score = inner_area / (area0 + area1 - inner_area);

            if (score > nms_threshold) {
                merged[j] = 1;
                buf.push_back(input[j]);
            }
        }
        switch (type) {
            case hard_nms: {
                output.push_back(buf[0]);
                break;
            }
            case blending_nms: {
                float total = 0;
                for (int i = 0; i < buf.size(); i++) {
                    total += exp(buf[i].score);
                }
                Object rects;
                memset(&rects, 0, sizeof(rects));
                for (int i = 0; i < buf.size(); i++) {
                    float rate = exp(buf[i].score) / total;
                    rects.x1 += buf[i].x1 * rate;
                    rects.y1 += buf[i].y1 * rate;
                    rects.x2 += buf[i].x2 * rate;
                    rects.y2 += buf[i].y2 * rate;
                    rects.score += buf[i].score * rate;
                }
                output.push_back(rects);
                break;
            }
            default: {
                printf("wrong type of nms.");
                exit(-1);
            }
        }
    }
}



int SSD::detect(const cv::Mat &rgb, std::vector<Object> &objects, std::vector<Object> &face_list, std::vector<cv::Mat>& facelandmarks,float prob_threshold,
                 float nms_threshold) {
    double start_time = ncnn::get_current_time();
    int img_w = rgb.cols;
    int img_h = rgb.rows;
    image_h = img_h;
    image_w = img_w;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, img_w, img_h, in_w,
                                                 in_h);

    const float mean_vals[3] = {127, 127, 127};
    const float norm_vals[3] = {1.0/128, 1.0/128, 1.0/128};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = facedetector.create_extractor();
    ex.input("input", in);



    {
        ncnn::Mat scores;
        ncnn::Mat boxes;
        ex.extract("scores", scores);
        ex.extract("boxes", boxes);
        generateBBox(objects, scores, boxes, prob_threshold, num_anchors);
        nms(objects, face_list,nms_threshold);
    }

    /*关键点
 **/
    if (face_list.size() > 0){
        for(int i = 0; i < face_list.size(); i++){
            return_d_m image_str = pre_process(rgb,192,face_list[i]);
            cv::Mat clip_rgb = image_str.dst;

            ncnn::Mat face_input = ncnn::Mat::from_pixels_resize(clip_rgb.data, ncnn::Mat::PIXEL_RGB, clip_rgb.cols, clip_rgb.rows, 192, 192);
            const float face_mean_vals[3] = {0.f, 0.f,0.f};
            const float face_norm_vals[3] = {1.f, 1.f,1.f};
            face_input.substract_mean_normalize(face_mean_vals, face_norm_vals);
            ncnn::Mat face_output;
            ncnn::Extractor ex_face = landmarks.create_extractor();

            ex_face.input("data", face_input); // 推理
            ex_face.extract("fc1", face_output); //face_output.w = 212 face_output.h = 1. 将模型推理输出存储在ncnn::Mat对象


            cv::Mat cvMat(212, 1, CV_32FC1);
            for(int i =0; i < face_output.w; i++){
                cvMat.at<float>(i,0) = face_output[i];
            }

            cv::Mat coord = post_progress(cvMat, image_str.matri);
            facelandmarks.push_back(coord);
        }
    }

    return 0;
}

int SSD::draw(cv::Mat &rgb, const std::vector<Object> &face_list,std::vector<cv::Mat>& facelandmarks) {

    for (size_t i = 0; i < face_list.size(); i++) {
        const Object &obj = face_list[i];

        cv::rectangle(rgb, cv::Point(obj.x1,obj.y1), cv::Point(obj.x2,obj.y2), cv::Scalar(0, 255, 0), 2);

        for(int j =0; j < facelandmarks[i].rows; j++){ //绘制关键点
            float x = facelandmarks[i].at<float>(j, 0);
            float y = facelandmarks[i].at<float>(j, 1);
            cv::circle(rgb,cv::Point2f(x, y),2, cv::Scalar(255, 255, 0), -1);
        }

        char text[256];
        sprintf(text, "%.1f%%",obj.score * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.x1;
        int y = obj.y1 - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;
        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), -1);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    return 0;
}
