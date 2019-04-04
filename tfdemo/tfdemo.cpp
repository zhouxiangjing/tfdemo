// tfdemo.cpp : �������̨Ӧ�ó������ڵ㡣
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

// ����һ��������OpenCV��Mat����ת��Ϊtensor��python����ֻҪ��cv2.read�������ľ������np.reshape֮��
// �������;ͳ���һ��tensor����tensor�����һ����Ȼ��Ϳ������뵽���������ˣ�����C++�汾���������翪�ŵ����
// Ҳ��Ҫ������ͼƬת����һ��tensor�����������OpenCV��ȡͼƬ�Ļ�������һ��Mat��Ȼ���Ҫ������ô��Matת��Ϊ
// Tensor��
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
	//ͼ�����resize����
	resize(img, img, cv::Size(input_cols, input_rows));
	//imshow("resized image",img);

	//��һ��
	img.convertTo(img, CV_32FC1);
	img = 1 - img / 255;

	//����һ��ָ��tensor�����ݵ�ָ��
	float *p = output_tensor->flat<float>().data();

	//����һ��Mat����tensor��ָ���,�ı����Mat��ֵ�����൱�ڸı�tensor��ֵ
	cv::Mat tempMat(input_rows, input_cols, CV_32FC1, p);
	img.convertTo(tempMat, CV_32FC1);
*/
	//    waitKey(0);

}

int test() {
	//����tensorflowģ��
	Session *session;
	cout << "start initalize session" << "\n";
	Status status = NewSession(SessionOptions(), &session);
	if (!status.ok()) {
		cout << status.ToString() << "\n";
		return 1;
	}
	GraphDef graph_def;
	status = ReadBinaryProto(Env::Default(), MNIST_MODEL_PATH, &graph_def);
	//MNIST_MODEL_PATHΪģ�͵�·������model_frozen.pb��·��
	if (!status.ok()) {
		cout << status.ToString() << "\n";
		return 1;
	}
	status = session->Create(graph_def);
	if (!status.ok()) {
		cout << status.ToString() << "\n";
		return 1;
	}
	cout << "tensorflow���سɹ�" << "\n";

	string image_path = "C:\\Users\\86529\\PycharmProjects\\untitled35\\mnist\\images\\99.png";
	int input_height = 28;
	int input_width = 28;
	string input_tensor_name = "x-input:0";
	string output_tensor_name = "layer2/add:0";


	/*---------------------------------�������ͼƬ-------------------------------------*/
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

	Tensor x(DT_FLOAT, TensorShape({ 1, 784 }));//�������������������������ͺʹ�С��

	auto dst = x.flat<float>().data();
	copy_n(mydata.begin(), 784, dst);

	CVMat_to_Tensor(img, &x, input_height, input_width);

	vector<pair<string, Tensor>> inputs = { { input_tensor_name, x } };	//����ģ������
	vector<Tensor> outputs;												//����ģ�����

	Status status2 = session->Run(inputs, { output_tensor_name }, {}, &outputs); //����ģ��,

	if (!status2.ok()) {
		cout << status2.ToString() << "\n";
		_tsystem(_T("pause"));
		return 1;
	}

	//�����ֵ����ȡ����
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

	// ������
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

