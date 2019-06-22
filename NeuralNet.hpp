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
#include "Mathematical.hpp"



namespace liu {
    
    class Net {
    public:
        std::vector<int> layer_neuron_num; // 每一层神经网络的神经元数目
        std::string activation_function = "sigmoid"; // 激活函数
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
        void initBias(cv::Scalar bias);
        
        // 前向传播
        void forward();
        
        // 反向传播
        void backward();
        
        //Train,use accuracy_threshold
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
        
    protected:
        // 为输入的mat进行赋值一个随机数
        void getRandom(cv::Mat &dst, int type, double a, double b);
        
        //Activation function
        cv::Mat activationFunction(cv::Mat &x, std::string func_type);
        
        //Compute delta error
        void deltaError();
        
        //Update weights
        void updateWeights();
    };
    
    //Get sample_number samples in XML file,from the start column.
    void get_input_label(std::string filename, cv::Mat& input, cv::Mat& label, int sample_num, int start = 0);
    
    // Draw loss curve
    void draw_curve(cv::Mat& board, std::vector<double> points);
}

#endif /* NeuralNet_hpp */