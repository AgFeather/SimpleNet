//
//  NetUtils.cpp
//  SimpleNet
//
//  Created by Dongfang Li on 2019/6/29.
//  Copyright © 2019年 Dongfang Li. All rights reserved.
//

#include "NetUtils.hpp"

namespace dongfang {
    //Get sample_number samples in XML file,from the start column.
    void get_data(std::string filename, cv::Mat& input, cv::Mat& label, int sample_num, int start) {
        cv::FileStorage fs;
        fs.open(filename, cv::FileStorage::READ);
        cv::Mat input_, target_;
        if (!fs.isOpened()) {
            std::cout<<"ERROR: file not open!"<<std::endl;
            return;
        }
        fs["input"] >> input_;
        fs["target"] >> target_;
        fs.release();
        input = input_(cv::Rect(start, 0, sample_num, input_.rows));
        label = target_(cv::Rect(start, 0, sample_num, target_.rows));
    }
    
    // 画损失曲线
    void draw_curve(cv::Mat& board, std::vector<double> points) {
        cv::Mat board_(620, 1000, CV_8UC3, cv::Scalar::all(200));
        board = board_;
        cv::line(board, cv::Point(0, 550), cv::Point(1000, 550), cv::Scalar(0, 0, 0), 2);
        cv::line(board, cv::Point(50, 0), cv::Point(50, 1000), cv::Scalar(0, 0, 0), 2);
        
        for (int i = 0; i < points.size() - 1; i++) {
            cv::Point pt1(50 + i * 2, (548 - points[i]));
            cv::Point pt2(50 + i * 2 + 1, (548 - points[i + 1]));
            cv::line(board, pt1, pt2, cv::Scalar(0, 0, 255), 2);
            if (i >= 1000){
                return;
            }
        }
        cv::imshow("Loss", board);
        cv::waitKey(1);
    }
}
