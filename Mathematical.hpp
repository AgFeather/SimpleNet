//
//  Mathematical.hpp
//  SimpleNet
//
//  Created by Dongfang Li on 2019/6/21.
//  Copyright © 2019年 Dongfang Li. All rights reserved.
//

#ifndef Mathematical_hpp
#define Mathematical_hpp

#include <stdio.h>


#include<opencv2/core/core.hpp>
#include<iostream>

namespace liu
{
    
    //sigmoid function
    cv::Mat sigmoid(cv::Mat &x);
    
    //Tanh function
    cv::Mat tanh(cv::Mat &x);
    
    //ReLU function
    cv::Mat ReLU(cv::Mat &x);
    
    //Derivative function
    cv::Mat derivativeFunction(cv::Mat& fx, std::string func_type);
    
    //Objective function
    void calcLoss(cv::Mat &output, cv::Mat &target, cv::Mat &output_error, float &loss);
    
    
}




#endif /* Mathematical_hpp */


