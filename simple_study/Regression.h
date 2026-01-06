#pragma once
#include <iostream>
#include <vector>
#include <map>
#define NORMALIZATION_MIN_MAX 1
#define NORMALIZATION_Z_SCORE 2
class Regression {
public:
	//样本特征创建，特征数量包含偏置项，自动填充1
	void createsamples(int feature_num, std::vector<std::vector<double>>& outputsamples, int samplenum,int randnum);
	//样本标签创建，与特征一一对应，需要输入预定的系数（偏置项，低阶项，高阶项，低阶交叉项）,以及随机数生成种子
	void createtags(int degree, std::vector<std::vector<double>>& inputsamples,const std::vector<double>& designed_weights, std::vector<double>& outputtags,int randnum);
	//模型评价系数结构体，含均方误差MSE，均方根误差RMSE，平均绝对误差MAE，模型拟合能力R2
	struct Metrics {
		double mse;//均方误差--训练求导用
		double rmse;//均方根误差--可是化汇报用，单位已统一，意为模型的平均误差
		double mae;//平均绝对误差--可以用于检测异常数据，如果 RMSE 远大于 MAE，则说明有异常数据
		double r2;//描述模型拟合能力，若训练集的r2与验证集的r2差异过大，说明过拟合或欠拟合
	};
	//模型参数（超参，权重和模型评测结果），超参含学习率，迭代次数，模型阶数，正则化系数
	struct parameters
	{
		//Hyperparameters超参
		double learning_rate=0.001;//学习率
		double epochs=10000;//迭代次数
		double degree=1;//模型阶数最大值
		double Lasso_alpha_L1=0;//L1正则化系数
		double Ridge_alpha_L2=0;//L2正则化系数 
		//其他参数
		Metrics mrmr;
		std::vector<double> weight;
	};
	//模型预处理信息结构体，包含归一化方式，归一化的均值和方差或最大值和最小值，便于反归一化
	struct reprocess_msg {
		int Normalization_type;//归一化类型
		std::vector<std::vector<double>> msg;//存储归一化相关信息，比如均值，方差
	};

	//数据预处理，包含高阶拓展(true开，false关)和归一化
	void dataprocess(std::vector<std::vector<double>>& inputsample, std::vector<double>& inputTag, reprocess_msg& msg, int degree, bool fill);//数据预处理-归一化
	//归一化方式min_max
	void normalization_min_max(std::vector<std::vector<double>>& inputsample, std::vector<double>& inputTag, reprocess_msg& msg);//min_max归一化
	//归一化方式Z_score
	void normalization_z_score(std::vector<std::vector<double>>& inputsample,std::vector<double>& inputTag, reprocess_msg& msg);//Z_score归一化
	//参数训练函数
	void Train_Regression(const std::vector<std::vector<double>>& inputsample, const std::vector<double>& inputTag, parameters& hms);//回归算法--训练
	//模型评价函数
	Metrics evaluate(const std::vector<std::vector<double>>& X,const std::vector<double>& y,const std::vector<double>& weights);
	//模型验证及评价函数
	void Validate_Regression(const std::vector<std::vector<double>>& inputsample, const std::vector<double>& inputTag, const parameters& pm, const reprocess_msg& msg);
	
};