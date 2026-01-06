#include "Regression.h"
#include <iostream>
#include <vector>
#include <random>

void methods::createsamples(int feature_num, std::vector<std::vector<double>>& outputsamples,int samplenum,int randnum)
{
	outputsamples.resize(samplenum, std::vector<double>(feature_num));//提前分配空间
	//根据所给低阶特征，生成1000份数据
	std::minstd_rand gen(randnum);//随机数生成器（每次生成固定的随机数）
	std::uniform_real_distribution<> dis(-1000.0,1000.0);//均匀生成-1000到1000的浮点数
	for (int i = 0; i < samplenum; ++i) {
		
		for (int j = 0; j < feature_num; ++j) {
			if (j) {
				outputsamples[i][j] = dis(gen);
			}
			else {
				outputsamples[i][j] = 1;
			}
		}
	}//随机特征值生成
}
void methods::createtags(int degree, std::vector<std::vector<double>>& inputsamples, const std::vector<double>& designed_weights, std::vector<double>& outputtags,int randnum)
{
	int size = inputsamples[0].size();
	for (auto& row : inputsamples) {
		for (int i = 1; i < degree; ++i) {
			for (int j = 1; j < size; ++j) {
				row.push_back(std::pow(row[j], i + 1));
			}
		}
	}//高次项填充
	for (auto& row : inputsamples) {
		for (int i = 1; i < size; ++i) {
			for (int j = i + 1; j < size; ++j) {
				row.push_back(row[i] * row[j]);
			}
		}
	}//低阶交叉项填充
	outputtags.resize(inputsamples.size());
	std::minstd_rand gen(randnum);
	std::normal_distribution<> dis(0.0, 0.5);//误差的均值是0，标准差是2
	for (int i = 0; i < inputsamples.size(); ++i) {
		double result = 0;
		for (int j = 0; j < inputsamples[0].size(); ++j) {
			if(j<designed_weights.size())result += designed_weights[j] * inputsamples[i][j];
		}
		result += dis(gen);
		outputtags[i] = result;
	}
}
void methods::dataprocess(std::vector<std::vector<double>>& inputsamples, std::vector<double>& inputTag, reprocess_msg& msg,int degree,bool fill)
{
	//是否进行高次项填充
	if (fill) {
		//多项式回归--高阶数填充
	//先低次项，后高次项，最后交叉项
		int size = inputsamples[0].size();
		for (auto& row : inputsamples) {
			for (int i = 1; i < degree; ++i) {
				for (int j = 1; j < size; ++j) {
					row.push_back(std::pow(row[j], i + 1));
				}
			}
		}//高次项填充
		for (auto& row : inputsamples) {
			for (int i = 1; i < size; ++i) {
				for (int j = i + 1; j < size; ++j) {
					row.push_back(row[i] * row[j]);
				}
			}
		}//低阶交叉项填充

		//还有高阶交叉项的填充，用不上，暂时不写
	}
	
	//数据需含偏置项全1
	switch (msg.Normalization_type) {
	case 1:
		//min_max归一化
		normalization_min_max(inputsamples, inputTag, msg);
		break;
	case 2:
		//Z—SCORE归一化
		normalization_z_score(inputsamples, inputTag, msg);
		break;
	}

	//更多预处理
	
	
	


}
void methods::normalization_min_max(std::vector<std::vector<double>>& inputsample, std::vector<double>& inputTag, reprocess_msg& msg)
{
	int n_features = inputsample[0].size();
	msg.msg.push_back({ 1,1 });
	std::vector<double> min_max;
	// 跳过 x0 列
	for (int j = 1; j < n_features; ++j) {
		min_max.push_back(inputsample[0][j]); 
		min_max.push_back(inputsample[0][j]);

		// 找最小最大值
		for (const auto& row : inputsample) {
			min_max[0] = std::min(min_max[0], row[j]);
			min_max[1] = std::max(min_max[1], row[j]);
		}
		
		// 归一化
		for (auto& row : inputsample) {
			row[j] = (row[j] - min_max[0]) / (min_max[1] - min_max[0] + 1e-9); // 防止除零
		}
		msg.msg.push_back(min_max);
		min_max.clear();
	}

	//标签归一化
	min_max[0] = *std::min_element(inputTag.begin(), inputTag.end());
	min_max[1] = *std::max_element(inputTag.begin(), inputTag.end());
	for (auto& tag : inputTag) {
		tag = (tag - min_max[0]) / (min_max[1] - min_max[0] + 1e-9);
	}
	msg.msg.push_back(min_max);
}
void methods::normalization_z_score(std::vector<std::vector<double>>& inputsample,std::vector<double>& inputTag, reprocess_msg& msg) {
	int n_features = inputsample[0].size();
	msg.msg.clear();
	msg.msg.push_back({ 1,0 });//这里x0的偏置项（全1）的均值为1，方差为0
	// 跳过 x0 列
	for (int j = 1; j < n_features; ++j) {
		//每次循环计算一列
		// 计算均值
		double mean = 0.0;
		for (const auto& row : inputsample) {
			mean += row[j];
		}
		mean /= inputsample.size();

		// 计算标准差
		double stddev = 0.0;
		for (const auto& row : inputsample) {
			stddev += std::pow(row[j] - mean, 2);
		}
		stddev = std::sqrt(stddev / inputsample.size());

		// 归一化
		for (auto& row : inputsample) {
			row[j] = (row[j] - mean) / (stddev + 1e-9);
		}
		msg.msg.push_back({ mean,stddev });
	}

	// 标签归一化
	double mean_tag=0.0;
	for(double d:inputTag){
		mean_tag += d;
	}
	mean_tag /= inputsample.size();
	double stddev_tag = 0.0;
	for (const auto& tag : inputTag) {
		stddev_tag += std::pow(tag - mean_tag, 2);
	}
	stddev_tag = std::sqrt(stddev_tag / inputTag.size());

	for (auto& tag : inputTag) {
		tag = (tag - mean_tag) / (stddev_tag + 1e-9);
	}
	msg.msg.push_back({ mean_tag,stddev_tag });
}
void methods::Train_Regression(const std::vector<std::vector<double>>& inputsample, const std::vector<double>& inputTag,  parameters& hms) {

	int rows = inputsample.size();
	int cols = inputsample[0].size();
	//第一列全是1
	std::vector<double >outputweight(cols, 0.0);//初始化特征权重

	//每循环一次，就更新一次特征权重,并重置梯度
	for (int epoch = 1; epoch <= hms.epochs; ++epoch) {
		std::vector<double> grad(cols, 0.0);//梯度
		std::vector <double> predict(rows, 0.0);//预测值=特征值*权重
		
		//获取预测值
		for (int i = 0; i < rows; ++i) {
			for (int k = 0; k < cols; ++k) {
				predict[i] += (outputweight[k] * inputsample[i][k]);
			}//每一行的预测值
		}
		//这里只做批量梯度下降，不做随机或少量的梯度下降了
		//获取每个特征值的偏导梯度
		//损失函数=MSE+L2正则化+L1正则化 --实现弹性网
		for (int j = 0; j < cols; ++j) {
			for (int i = 0; i < rows; ++i) {
				grad[j] += ((predict[i] - inputTag[i]) * inputsample[i][j]);
			}
			grad[j] /= rows;
		}

		//更新特征权重(减 偏导梯度*学习率 ）
		
		double lr = hms.learning_rate / (1 + 0.001 * epoch);//固定学习率可能导致后期震荡或收敛慢,这里随迭代次数逐渐降低学习率
		for (int i = 0; i < cols; ++i) {
			if (!i) { outputweight[i] -= (grad[i]* lr); }
			else {
				double total_grad = grad[i] + 2 * hms.Ridge_alpha_L2 * outputweight[i] + hms.Lasso_alpha_L1 * (outputweight[i] > 0 ? 1.0 : ((outputweight[i] < 0 ? -1 : 0)));
				outputweight[i] -= (total_grad * lr);
			}
		}//启用了弹性网正则化
	}
	//迭代结束后记录参数
	hms.mrmr = evaluate(inputsample, inputTag, outputweight);
	hms.weight = outputweight;
}
methods::Metrics methods::evaluate(const std::vector<std::vector<double>>& X,const std::vector<double>& y,const std::vector<double>& weights)
{
	int n = X.size();
	if (n == 0) return { 0, 0, 0, 0 };

	double sum_sq_error = 0.0;
	double sum_abs_error = 0.0;
	double sum_y = 0.0;
	double sum_y_sq = 0.0;

	for (int i = 0; i < n; ++i) {
		double y_hat = 0.0;
		for (int j = 0; j < weights.size(); ++j) {
			y_hat += weights[j] * X[i][j];
		}

		double err = y_hat - y[i];
		sum_sq_error += err * err;
		sum_abs_error += std::fabs(err);

		sum_y += y[i];
		sum_y_sq += y[i] * y[i];
	}

	double mse = sum_sq_error / n;
	double rmse = std::sqrt(mse);
	double mae = sum_abs_error / n;

	double y_mean = sum_y / n;
	double total_variance = sum_y_sq / n - y_mean * y_mean;
	double r2 = (total_variance > 1e-9) ? 1.0 - (mse / total_variance) : 0.0;//避免过大
	Metrics m{ mse, rmse, mae, r2 };
	return m;
}
void methods::Validate_Regression(const std::vector<std::vector<double>>& inputsample,const std::vector<double>& inputTag,const parameters& pm,const reprocess_msg& msg)
{
	std::cout << "若RMSE与MAE差距过大，说明有异常数据" << std::endl;
	std::cout << "均方根误差为：" << pm.mrmr.rmse << std::endl;
	std::cout << "平均绝对误差为：" << pm.mrmr.mae << std::endl;

	Metrics validate = evaluate(inputsample, inputTag, pm.weight);
	if (pm.mrmr.r2 - validate.r2 > 0.1) {
		std::cout << "可能过拟合：训练集 R²:" << pm.mrmr.r2
			<< " 比验证集:" << validate.r2 << "高很多" << std::endl;
	}
	else if (pm.mrmr.r2 < 0.7 && validate.r2 < 0.7) {
		std::cout << "可能欠拟合：训练集和验证集 R² 都很低" << std::endl;
	}
	else if (pm.mrmr.r2 > 0.7 && validate.r2 > 0.7) {
		std::cout << "拟合良好：训练集和验证集指标接近" << std::endl;
		std::cout << "训练集R2：" << pm.mrmr.r2 << " 验证集R2：" << validate.r2 << std::endl;

		// 输出归一化后的权重
		std::cout << "归一化后的权重参数：";
		for (double w : pm.weight) {
			std::cout << " " << w;
		}
		std::cout << std::endl;

		// 转换为原始数据的权重
		std::vector<double> original_weights;
		if (msg.Normalization_type == NORMALIZATION_Z_SCORE) {
			//标签的均值mean，方差std
			double y_mean = msg.msg.back()[0];
			double y_std = msg.msg.back()[1];

			original_weights.resize(pm.weight.size());
			//先将标签归一化，即两边都*y_std+y_mean
			original_weights[0] = pm.weight[0] * y_std + y_mean;

			for (int j = 1; j < pm.weight.size(); ++j) {
				double x_mean = msg.msg[j][0];
				double x_std = msg.msg[j][1];
				original_weights[0] -= y_std * pm.weight[j] * x_mean / x_std;
				original_weights[j] = pm.weight[j] * y_std / x_std;
			}
		}
		else if (msg.Normalization_type == NORMALIZATION_MIN_MAX) {
			double y_min = msg.msg.back()[0];
			double y_max = msg.msg.back()[1];
			double y_range = y_max - y_min;

			original_weights.resize(pm.weight.size());
			original_weights[0] = pm.weight[0] * y_range + y_min;

			for (int j = 1; j < pm.weight.size(); ++j) {
				double x_min = msg.msg[j][0];
				double x_max = msg.msg[j][1];
				double x_range = x_max - x_min;
				original_weights[0] -= y_range * pm.weight[j] * x_min / x_range;
				original_weights[j] = pm.weight[j] * y_range / x_range;
			}
		}

		// 输出原始权重
		std::cout << "原始数据的权重参数：";
		for (double w : original_weights) {
			std::cout << " " << w;
		}
		std::cout << std::endl;
	}
}