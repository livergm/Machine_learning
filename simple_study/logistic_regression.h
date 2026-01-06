#pragma once
#include <iostream>
#include <vector>
class logistic_regssion {
public:
	//参数
	struct parameters
	{
		//Hyperparameters超参
		double learning_rate = 0.001;//学习率
		double epochs = 10000;//迭代次数
		double degree = 1;//模型阶数最大值
		double Lasso_alpha_L1 = 0;//L1正则化系数
		double Ridge_alpha_L2 = 0;//L2正则化系数 
		int kinds = 2;//类别数量
		//其他参数
		//对于sigmoid，用一行代表特征
		//对于softmax，一列表示一个类别，一行表示一个特征
		std::vector<std::vector<double>> weight;

	};
	//sigmoid函数
	double sigmoid(double z);
	void softmax(std::vector<std::vector<double>>& soft);
	//二分类--sigmoid方法
	void Train_log_regression_2(const std::vector<std::vector<double>>& samples,const std::vector<double>& tags,parameters& pm);
	//多分类，softmax方法
	void Train_log_regression_x(const std::vector<std::vector<double>>& samples, const std::vector<std::vector<double>>& tags, parameters& pm);
};