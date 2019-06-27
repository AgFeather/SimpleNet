//
//  NeuralNet.cpp
//  SimpleNet
//
//  Created by Dongfang Li on 2019/6/21.
//  Copyright © 2019年 Dongfang Li. All rights reserved.
//

#include "NeuralNet.hpp"

namespace dongfang {
    
    void Net::initNet(std::vector<int> num_units_each_layer_) {
        num_units_each_layer = num_units_each_layer_;
        num_layers = (int)num_units_each_layer_.size();
        // 对各个层进行初始化
        layer.resize(num_layers);
        for (int i = 0; i < layer.size(); i++){
            layer[i].create(num_units_each_layer[i], 1, CV_32FC1);
        }
        std::cout << "Initialized layers!" << std::endl;
        
        // 对weight list，bias list和delta error list形状进行初始化
        weights.resize(num_layers - 1);
        bias.resize(num_layers - 1);
        delta_err.resize(num_layers - 1);
        for (int i = 0; i < num_layers - 1; i++){
            weights[i].create(num_units_each_layer[i + 1], num_units_each_layer[i], CV_32FC1); // 注意每个weight的形状是相反的
            bias[i] = cv::Mat::zeros(num_units_each_layer[i + 1], 1, CV_32FC1);
            delta_err[i].create(num_units_each_layer[i + 1], 1, CV_32FC1);
        }
        
        // 对weight和bias的值进行初始化
        initWeights(0, 0., 0.01);
        initBiases(cv::Scalar(0.05));
        std::cout << "initialized weights matrices and bias!" << std::endl;
    }
    
    // 用随机值初始化weight
    void Net::initWeights(int type, double a, double b) {
        for (int i = 0; i < weights.size(); i++) {
            getRandom(weights[i], 0., 0.1, "gaussian");
        }
    }
    
    // 初始化bias list
    void Net::initBiases(cv::Scalar bias_) {
        for (int i = 0; i < bias.size(); i++) {
            bias[i] = bias_;
        }
    }
    
    // 前向传播
    void Net::forwardPropagation() {
        for (int i = 0; i < num_layers - 1; ++i) {
            cv::Mat product = weights[i] * layer[i] + bias[i];
            layer[i + 1] = activationFunction(product, activation_function);
        }
        // 根据nn的output和target计算loss，目前使用square损失
        calcLoss(layer[layer.size() - 1], target, output_error, loss);
    }
    
    // 进行反向传播
    void Net::backwardPropagation() {
        calcDeltaError();
        updateParameters();
    }
    
    //计算 每一层每一个节点对应的delta error
    void Net::calcDeltaError() {
        for (int i = (int)delta_err.size() - 1; i >= 0; i--) { // 从输出层开始更新error
            cv::Mat dx = derivativeFunction(layer[i + 1], activation_function); // 对激活函数求偏导
            if (i == delta_err.size() - 1) { // 更新输出层的delta error
                delta_err[i] = dx.mul(output_error);
            }
            else { // 更新隐藏层的delta error
                delta_err[i] = dx.mul((weights[i + 1]).t() * delta_err[i + 1]);
            }
        }
    }
    
    // 更新weight和bias
    void Net::updateParameters() {
        for (int i = 0; i < weights.size(); ++i) {
            weights[i] = weights[i] + learning_rate * (delta_err[i] * layer[i].t());
            bias[i] = bias[i] + learning_rate * delta_err[i];
        }
    }
    
    // 使用loss 作为阈值训练网络
    void Net::train(cv::Mat input, cv::Mat target_, float loss_threshold, bool draw_loss_curve) {
        if (input.empty()) {
            std::cout << "Input is empty!" << std::endl;
            return;
        }
        std::cout << "Training begin!" << std::endl;
        
        if (input.rows == layer[0].rows) { // 训练
            double epoch_loss = loss_threshold + 0.01;
            int epoch = 0;
            while (epoch_loss >= loss_threshold) {
                epoch_loss = 0.;
                for (int i = 0; i < input.cols; i++) {
                    layer[0] = input.col(i);
                    target = target_.col(i);
                    forwardPropagation();
                    backwardPropagation();
                    epoch_loss += loss;
                }
                epoch++;
                if (epoch % output_interval == 0) {
                    std::cout << "Training epoch:" << epoch << " Loss sum:" << epoch_loss << std::endl;
                }
                if (epoch % 100 == 0) {
                    learning_rate *= fine_tune_factor;
                }
                if (draw_loss_curve) {
                    loss_vec.push_back(epoch_loss);
                    draw_curve(board, loss_vec);
                }
            }
            std::cout << "Train sucessfully! Training epoch:"<<epoch<<" Loss sum:"<<epoch_loss << std::endl;
        }
        else { // 样本大小不等于网络输入层的大小
            std::cout << "Rows of input don't cv::Match the number of input!" << std::endl;
        }
    }
    
    void Net::train(cv::Mat input, cv::Mat target_, int num_epochs, bool draw_loss_curve){
        int show_every_n = 3;
        int global_step = 0;
        if (input.empty()) {
            std::cout<<"ERROR: Training dataset is empty"<<std::endl;
            return;
        }
        if (layer[0].rows == input.rows) {
            for (int epoch = 1; epoch<=num_epochs; epoch++) {
                double epoch_loss = 0.f;
                for (int i = 0; i<input.cols; i++) {
                    layer[0] = input.col(i);
                    target = target_.col(i);
                    forwardPropagation();
                    epoch_loss += loss;
                    backwardPropagation();
                    global_step++;
                }
                if (epoch % show_every_n == 0) {
                    std::cout<<"Training epoch:"<<epoch<<" Global step:"<<global_step<<" Loss"<<epoch_loss<<std::endl;
                }
                if (draw_loss_curve) {
                    loss_vec.push_back(epoch_loss);
                    draw_curve(board, loss_vec);
                }
            }
            std::cout<<"Training finished"<<std::endl;
        }else { // 样本大小不等于网络输入层的大小
            std::cout << "Rows of input don't cv::Match the number of input!" << std::endl;
        }
        
    }
    
    // 对已经训练完成的模型进行测试并计算准确率
    double Net::test(cv::Mat &input, cv::Mat &target_) {
        if (input.empty()) {
            std::cout << "Input is empty!" << std::endl;
            return -1;
        }
        
        std::cout << std::endl << "Test begin!" << std::endl;
        int correct_count = 0;
        double accuracy = 0.f;
        cv::Mat sample;
        if (input.rows == (layer[0].rows)) {
            for (int i = 0; i < input.cols; i++) {
                sample = input.col(i);
                int predict_index = predict_one(sample);
                
                cv::Point target_maxLoc;
                minMaxLoc(target_.col(i), NULL, NULL, NULL, &target_maxLoc, cv::noArray());
                int target_index = target_maxLoc.y;
                
                std::cout << "Test sample:" << i << " Predict:" << predict_index << " Target:" << target_index << std::endl;
                if (predict_index == target_index) {
                    correct_count++;
                }
            }
            accuracy = (double)correct_count / input.cols;
            std::cout << "accuracy: " << accuracy * 100<<"%" << std::endl;
        }
        
        else {
            std::cout << "Rows of input don't cv::Match the number of input!" << std::endl;
        }
        return accuracy;
    }
    
    // 输入一个sample并进行预测
    int Net::predict_one(cv::Mat &input) {
        if (input.empty()) {
            std::cout << "Input is empty!" << std::endl;
            return -1;
        }
        
        if (input.rows == (layer[0].rows) && input.cols == 1) {
            layer[0] = input;
            forwardPropagation();
            cv::Mat layer_out = layer[layer.size() - 1];
            cv::Point predict_maxLoc;
            minMaxLoc(layer_out, NULL, NULL, NULL, &predict_maxLoc, cv::noArray());
            return predict_maxLoc.y;
        }
        else {
            std::cout << "Please give one sample and ensure input.rows = layer[0].rows" << std::endl;
            return -1;
        }
    }
    
    // 对多个sample进行预测
    std::vector<int> Net::predict(cv::Mat &input) {
        std::vector<int> predicted_labels;
        if (input.rows == (layer[0].rows) && input.cols > 1) {
            for (int i = 0; i < input.cols; ++i) {
                cv::Mat sample = input.col(i);
                int predicted_label = predict_one(sample);
                predicted_labels.push_back(predicted_label);
            }
        }
        return predicted_labels;
    }
    
    
    // to be continued
    //Save model;
    void Net::save(std::string filename) {
        cv::FileStorage model(filename, cv::FileStorage::WRITE);
        model << "num_units_each_layer" << num_units_each_layer;
        model << "learning_rate" << learning_rate;
        model << "activation_function" << activation_function;
        
        for (int i = 0; i < weights.size(); i++) {
            std::string weight_name = "weight_" + std::to_string(i);
            model << weight_name << weights[i];
        }
        model.release();
    }
    
    //Load model;
    void Net::load(std::string filename) {
        cv::FileStorage fs;
        fs.open(filename, cv::FileStorage::READ);
        cv::Mat input_, target_;
        
        fs["num_units_each_layer"] >> num_units_each_layer;
        initNet(num_units_each_layer);
        
        for (int i = 0; i < weights.size(); i++)
        {
            std::string weight_name = "weight_" + std::to_string(i);
            fs[weight_name] >> weights[i];
        }
        
        fs["learning_rate"] >> learning_rate;
        fs["activation_function"] >> activation_function;
        
        fs.release();
    }
    
    //Get sample_number samples in XML file,from the start column.
    void get_input_label(std::string filename, cv::Mat& input, cv::Mat& label, int sample_num, int start) {
        cv::FileStorage fs;
        fs.open(filename, cv::FileStorage::READ);
        cv::Mat input_, target_;
        if (!fs.isOpened()) {
            std::cout<<"ERROR: file not open!"<<std::endl;
        }
        fs["input"] >> input_;
        fs["target"] >> target_;
        fs.release();
        input = input_(cv::Rect(start, 0, sample_num, input_.rows));
        label = target_(cv::Rect(start, 0, sample_num, target_.rows));
    }
    
    //Draw loss curve
    void draw_curve(cv::Mat& board, std::vector<double> points) {
        cv::Mat board_(620, 1000, CV_8UC3, cv::Scalar::all(200));
        board = board_;
        cv::line(board, cv::Point(0, 550), cv::Point(1000, 550), cv::Scalar(0, 0, 0), 2);
        cv::line(board, cv::Point(50, 0), cv::Point(50, 1000), cv::Scalar(0, 0, 0), 2);
        
        for (size_t i = 0; i < points.size() - 1; i++)
        {
            cv::Point pt1(50 + i * 2, (int)(548 - points[i]));
            cv::Point pt2(50 + i * 2 + 1, (int)(548 - points[i + 1]));
            cv::line(board, pt1, pt2, cv::Scalar(0, 0, 255), 2);
            if (i >= 1000){
                return;
            }
        }
        cv::imshow("Loss", board);
        cv::waitKey(1);
    }
    
}
