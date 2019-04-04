// tfdemo.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <eigen/Dense>

#include "TestTensorFlow.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/ops/standard_ops.h"
using namespace tensorflow;

using namespace std;

#define MNIST_MODEL_PATH "C:/Users/86529/PycharmProjects/untitled35/mnist/frozen_model/mnist_frozen.pb"


using namespace cv;

// 定义一个函数讲OpenCV的Mat数据转化为tensor，python里面只要对cv2.read读进来的矩阵进行np.reshape之后，
// 数据类型就成了一个tensor，即tensor与矩阵一样，然后就可以输入到网络的入口了，但是C++版本，我们网络开放的入口
// 也需要将输入图片转化成一个tensor，所以如果用OpenCV读取图片的话，就是一个Mat，然后就要考虑怎么将Mat转化为
// Tensor了
void CVMat_to_Tensor(Mat img, Tensor* output_tensor, int input_rows, int input_cols)
{
	std::vector<float> mydata;
	for (int row = 0; row<img.rows; row++)
	{
		for (int col = 0; col<img.cols; col++)
		{
			int nn = img.at<uchar>(row, col);

			if (nn == 0) {
				mydata.push_back(0.0);
			}
			else {
				mydata.push_back(1.0);
			}

		}
	}

/*
	//imshow("input image",img);
	//图像进行resize处理
	resize(img, img, cv::Size(input_cols, input_rows));
	//imshow("resized image",img);

	//归一化
	img.convertTo(img, CV_32FC1);
	img = 1 - img / 255;

	//创建一个指向tensor的内容的指针
	float *p = output_tensor->flat<float>().data();

	//创建一个Mat，与tensor的指针绑定,改变这个Mat的值，就相当于改变tensor的值
	cv::Mat tempMat(input_rows, input_cols, CV_32FC1, p);
	img.convertTo(tempMat, CV_32FC1);
*/
	//    waitKey(0);

}

int test() {
	//加载tensorflow模型
	Session *session;
	cout << "start initalize session" << "\n";
	Status status = NewSession(SessionOptions(), &session);
	if (!status.ok()) {
		cout << status.ToString() << "\n";
		return 1;
	}
	GraphDef graph_def;
	status = ReadBinaryProto(Env::Default(), MNIST_MODEL_PATH, &graph_def);
	//MNIST_MODEL_PATH为模型的路径，即model_frozen.pb的路径
	if (!status.ok()) {
		cout << status.ToString() << "\n";
		return 1;
	}
	status = session->Create(graph_def);
	if (!status.ok()) {
		cout << status.ToString() << "\n";
		return 1;
	}
	cout << "tensorflow加载成功" << "\n";

	string image_path = "C:\\Users\\86529\\PycharmProjects\\untitled35\\mnist\\images\\99.png";
	int input_height = 28;
	int input_width = 28;
	string input_tensor_name = "x-input:0";
	string output_tensor_name = "layer2/add:0";


	/*---------------------------------载入测试图片-------------------------------------*/
	cout << endl << "<------------loading test_image-------------->" << endl;
	Mat img = imread(image_path, 0);
	if (img.empty())
	{
		cout << "can't open the image!!!!!!!" << endl;
		return -1;
	}

	std::vector<float> mydata;
	for (int row = 0; row<img.rows; row++)
	{
		for (int col = 0; col<img.cols; col++)
		{
			int nn = img.at<uchar>(row, col);

			if (nn == 0) {
				mydata.push_back(0.0);
			}
			else {
				mydata.push_back(1.0);
			}

		}
	}

	Tensor x(DT_FLOAT, TensorShape({ 1, 784 }));//定义输入张量，包括数据类型和大小。

	auto dst = x.flat<float>().data();
	copy_n(mydata.begin(), 784, dst);

	CVMat_to_Tensor(img, &x, input_height, input_width);

	vector<pair<string, Tensor>> inputs = { { input_tensor_name, x } };	//定义模型输入
	vector<Tensor> outputs;												//定义模型输出

	Status status2 = session->Run(inputs, { output_tensor_name }, {}, &outputs); //调用模型,

	if (!status2.ok()) {
		cout << status2.ToString() << "\n";
		_tsystem(_T("pause"));
		return 1;
	}

	//把输出值给提取出来
	cout << "Output tensor size:" << outputs.size() << std::endl;
	for (std::size_t i = 0; i < outputs.size(); i++) {
		cout << outputs[i].DebugString() << endl;
	}

	Tensor t = outputs[0];                   // Fetch the first tensor
	auto tmap = t.tensor<float, 2>();        // Tensor Shape: [batch_size, target_class_num]
	int output_dim = t.shape().dim_size(1);  // Get the target_class_num from 1st dimension

											 // Argmax: Get Final Prediction Label and Probability
	int output_class_id = -1;
	double output_prob = 0.0;
	for (int j = 0; j < output_dim; j++)
	{
		cout << "Class " << j << " prob:" << tmap(0, j) << "," << std::endl;
		if (tmap(0, j) >= output_prob) {
			output_class_id = j;
			output_prob = tmap(0, j);
		}
	}

	// 输出结果
	cout << "Final class id: " << output_class_id << std::endl;
	cout << "Final class prob: " << output_prob << std::endl;

	return 1;
}

int main()
{

	test();
	

	_tsystem(_T("pause"));

    return 0;
}

