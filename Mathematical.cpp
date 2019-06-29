//
//  Mathematical.cpp
//  SimpleNet
//
//  Created by Dongfang Li on 2019/6/21.
//  Copyright © 2019年 Dongfang Li. All rights reserved.
//

#include "Mathematical.hpp"


namespace dongfang
{
    
    //对一个Matrix用随机值进行初始化.if type =0,Gaussian.else uniform.
    void getRandom(cv::Mat &dst, double begin, double end, std::string type="gaussian") {
        if (type == "gaussian") {
            randn(dst, begin, end);
        }
        else {
            randu(dst, begin, end);
        }
    }

    //Activation function
    cv::Mat activationFunction(cv::Mat &x, std::string func_type) {
        cv::Mat fx;
        if (func_type == "sigmoid"){
            fx = sigmoid(x);
        }
        else if (func_type == "tanh"){
            fx = tanh(x);
        }
        else if (func_type == "ReLU"){
            fx = ReLU(x);
        }
        else{
            throw "error";
        }
        return fx;
    }
    
    //sigmoid function
    cv::Mat sigmoid(cv::Mat &x) {
        cv::Mat exp_x, fx;
        cv::exp(-x, exp_x);
        fx = 1.0 / (1.0 + exp_x);
        return fx;
    }
    
    //tanh function
    cv::Mat tanh(cv::Mat &x) {
        cv::Mat exp_x_, exp_x, fx;
        cv::exp(-x, exp_x_);
        cv::exp(x, exp_x);
        fx = (exp_x - exp_x_) / (exp_x + exp_x_);
        return fx;
    }
    
    //ReLU function
    cv::Mat ReLU(cv::Mat &x) {
        cv::Mat fx = x;
        for (int i = 0; i < fx.rows; i++) {
            for (int j = 0; j < fx.cols; j++) {
                if (fx.at<float>(i, j) < 0) {
                    fx.at<float>(i, j) = 0;
                }
            }
        }
        return fx;
    }
    
    // 求导
    cv::Mat derivativeFunction(cv::Mat& fx, std::string func_type) {
        cv::Mat dx;
        if (func_type == "sigmoid") {
            dx = sigmoid(fx).mul((1 - sigmoid(fx)));
        }
        else if (func_type == "tanh") {
            cv::Mat tanh_2;
            pow(tanh(fx), 2., tanh_2);
            dx = 1 - tanh_2;
        }
        else if (func_type == "ReLU") {
            dx = fx;
            for (int i = 0; i < fx.rows; i++) {
                for (int j = 0; j < fx.cols; j++) {
                    if (fx.at<float>(i, j) > 0) {
                        dx.at<float>(i, j) = 1;
                    }
                }
            }
        }
        else {
            throw "unknown activation function";
        }
        return dx;
    }
    
    // 根据网络输出output和预测目标target计算loss
    double calcLoss(cv::Mat &output, cv::Mat &target, cv::Mat &output_error) {
        if (target.empty()) {
            std::cout << "Can't find the target cv::Matrix" << std::endl;
            return -1;
        }
        output_error = target - output;
        cv::Mat err_sqrare;
        pow(output_error, 2., err_sqrare);
        cv::Scalar err_sqr_sum = sum(err_sqrare);
        return  err_sqr_sum[0] / (double)(output.rows);
    }
    
}


