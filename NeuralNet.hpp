//
//  NeuralNet.hpp
//  SimpleNet
//
//  Created by Dongfang Li on 2019/6/21.
//  Copyright © 2019年 Dongfang Li. All rights reserved.
//

#ifndef NeuralNet_hpp
#define NeuralNet_hpp

#pragma once

#include <iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include<string>

#include "Mathematical.hpp"



namespace dongfang {
    
    class Net {
        
    public:
        std::vector<int> num_units_each_layer; // 每一层神经网络的神经元数目
        std::string activation_function = "sigmoid"; // 激活函数
        int num_layers; // 整个神经网络的层数
        int output_interval = 10;
        float learning_rate;
        float accuracy = 0.;
        std::vector<double> loss_vec;
        float fine_tune_factor = 1.01;
        
    protected:
        std::vector<cv::Mat> layer; // 用以表示各个层，每个层用一个Mat表示
        std::vector<cv::Mat> weights; // weight list
        std::vector<cv::Mat> bias; // bias list
        std::vector<cv::Mat> delta_err;
        
        cv::Mat output_error;
        cv::Mat target;
        cv::Mat board;
        float loss;
        
    public:
        Net() {};
        ~Net() {};
        // 初始化整个神经网络，输入一个vector，表示每一层的神经元个数
        void initNet(std::vector<int> layer_neuron_num_);
        // 初始化权重list
        void initWeights(int type = 0, double a = 0., double b = 0.1);
        // 初始化bias list
        void initBiases(cv::Scalar bias);
        // 前向传播
        void forwardPropagation();
        // 反向传播
        void backwardPropagation();
        // 计算每一层的delta error
        void calcDeltaError();
        // 更新weight和bias
        void updateParameters();
        // 训练模型，使用准确率作为阈值
        void train(cv::Mat input, cv::Mat target, float accuracy_threshold);
        
        //Train,use loss_threshold
        void train(cv::Mat input, cv::Mat target_, float loss_threshold, bool draw_loss_curve = false);
        
        //Test
        void test(cv::Mat &input, cv::Mat &target_);
        
        //Predict,just one sample
        int predict_one(cv::Mat &input);
        
        //Predict,more  than one samples
        std::vector<int> predict(cv::Mat &input);
        
        //Save model;
        void save(std::string filename);
        
        //Load model;
        void load(std::string filename);
    };
    
    //Get sample_number samples in XML file,from the start column.
    void get_input_label(std::string filename, cv::Mat& input, cv::Mat& label, int sample_num, int start = 0);
    
    // Draw loss curve
    void draw_curve(cv::Mat& board, std::vector<double> points);
}

#endif /* NeuralNet_hpp */
