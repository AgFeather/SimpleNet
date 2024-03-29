#include "NeuralNet.hpp"
#include "NetUtils.hpp"

using namespace std;
using namespace cv;
using namespace dongfang;

int main(int argc, char *argv[])
{
    //Set neuron number of every layer
    vector<int> layer_neuron_num = { 784,100,100,10 };

    // Initialise Net and weights
    Net net;
    net.initNet(layer_neuron_num);

    //Get test samples and test samples
    Mat input, label, test_input, test_label;
    int sample_number = 800;
    get_data("data/input_label_1000.xml", input, label, sample_number);
    get_data("data/input_label_1000.xml", test_input, test_label, 200, 800);

    //Set loss threshold,learning rate and activation function
    net.learning_rate = 0.002;
    net.output_interval = 2;
    float loss_threshold = 20.f;
    net.activation_function = "ReLU";

    // 按照loss阈值对模型进行训练
    //net.train(input, label, loss_threshold, false);
    
    // 根据epochs进行训练
    //net.train(input, label, 100, false);
    
    // batch training
    net.train_batch(input, label);
    
    net.test(test_input, test_label);

//    //Save the model
//    net.save("models/model_ReLU_800_200.xml");
//
    //getchar();
    return 0;
    
}




