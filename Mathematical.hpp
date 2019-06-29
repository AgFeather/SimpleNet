//
//  Mathematical.hpp
//  SimpleNet
//
//  Created by Dongfang Li on 2019/6/21.
//  Copyright © 2019年 Dongfang Li. All rights reserved.
//

#ifndef Mathematical_hpp
#define Mathematical_hpp

#include<iostream>
#include<string>
#include<opencv2/core/core.hpp>

namespace dongfang
{
    
    //sigmoid function
    cv::Mat sigmoid(cv::Mat &x);
    
    //Tanh function
    cv::Mat tanh(cv::Mat &x);
    
    //ReLU function
    cv::Mat ReLU(cv::Mat &x);
    
    //Derivative function
    cv::Mat derivativeFunction(cv::Mat& fx, std::string func_type);
    
    // 激活函数
    cv::Mat activationFunction(cv::Mat &x, std::string func_type);
    
    // Objective function
    double calcLoss(cv::Mat &output, cv::Mat &target, cv::Mat &output_error);
    
    //initialise the weights cv::Matrix.if type =0,Gaussian.else uniform.
    void getRandom(cv::Mat &dst, double begin, double end, std::string type);
}




#endif /* Mathematical_hpp */


