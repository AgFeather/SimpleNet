//
//  NetUtils.hpp
//  SimpleNet
//
//  Created by Dongfang Li on 2019/6/29.
//  Copyright © 2019年 Dongfang Li. All rights reserved.
//

#ifndef NetUtils_hpp
#define NetUtils_hpp

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

namespace dongfang{
    //Get sample_number samples in XML file,from the start column.
    void get_data(std::string filename, cv::Mat& input, cv::Mat& label, int sample_num, int start = 0);
    // Draw loss curve
    void draw_curve(cv::Mat& board, std::vector<double> points);
}


#endif /* NetUtils_hpp */
