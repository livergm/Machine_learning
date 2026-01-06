#include "logistic_regression.h"

double logistic_regssion::sigmoid(double z)
{
	if (z > 60.0) return 1.0;    // z很大时，sigmoid≈1
	if (z < -60.0) return 0.0;   // z很小时，sigmoid≈0
	return 1.0 / (std::exp(-z) + 1.0);
}

void logistic_regssion::softmax(std::vector<std::vector<double>>& soft)
{

	//注意：exp计算容易超出数据范围（700就会溢出为inf）
	for (int i = 0; i < soft.size(); ++i) {
		double mon=0.0;
		double max= *std::max_element(soft[i].begin(), soft[i].end());
		for (int j = 0; j < soft[0].size(); ++j) {
			mon += std::exp(soft[i][j]-max);
		}//分母计算
		//利用exp(a-b)=exp(a)/exp(b) 以减小数据超出范围的可能性
		for (int j = 0; j < soft[0].size(); ++j) {
			soft[i][j] = std::exp(soft[i][j]-max) / mon;
		}//数据修改
	}
}

void logistic_regssion::Train_log_regression_2(const std::vector<std::vector<double>>& inputsamples, const std::vector<double>& inputtags, parameters& inoutpm)
{
	//逻辑回归是分类模型，采用了回归模型的思想
	//使用sigmod函数将函数值控制在0-1之间，且单调可导
	//简单的逻辑回归--即二分类问题，将标签分为0和1
	//相较于普通回归，逻辑回归是将普通回归的标签值进行sigmod转换，将其转换到0-1之间
	//将转换后的值，当成是属于正例（1）的概率
	//y_r*log(y_p)+(1-y_r)log(1-y_p)是 预测值是真实值的概率，与预测值越接近，值越接近0（若过log的底大于1，那么预测值恒小于零，概率越低，值越小）
	//给它加上负数得损失函数 -（ y_r*log(y_p)+(1-y_r)log(1-y_p) ）,预测值与真实值越接近，值越接近0，越远离，值越大。
	//得 -1/m（  y_r*log(y_p)+(1-y_r)log(1-y_p) ）
	//其中y_r是真实值{0,1}，  y_p是预测值（0,1）
	// 求参数偏导后再梯度下降
	int cols = inputsamples[0].size();
	int rows = inputsamples.size();
	std::vector<double>outputweight;
	if (inoutpm.weight.empty()) {
		inoutpm.weight.assign(1, std::vector<double>(cols, 0.0));
		outputweight.assign(cols,0.0);
	}//初始化特征权重
	else {
		outputweight = inoutpm.weight[0];
	}//便于继承上次训练成果


	for (int epoch = 1; epoch <= inoutpm.epochs; ++epoch) {
		//获取预测值
		std::vector<double> grad(cols, 0.0);//梯度
		std::vector <double> predict(rows, 0.0);//预测值=sigmoid(特征值*权重)
		for (int i = 0; i < rows; ++i) {
			for (int k = 0; k < cols; ++k) {
				predict[i] += (outputweight[k] * inputsamples[i][k]);
			}//每一行的预测值
			predict[i] = sigmoid(predict[i]);
		}

		
		//求损失函数偏导
		for (int j = 0; j < cols; ++j) {
			for (int i = 0; i < rows; ++i) {
				grad[j] += ((predict[i] - inputtags[i]) * inputsamples[i][j]);
			}
			grad[j] /= rows;
		}

		//梯度下降
		double lr = inoutpm.learning_rate / (1 + 0.001 * epoch);//固定学习率可能导致后期震荡或收敛慢,这里随迭代次数逐渐降低学习率
		for (int i = 0; i < cols; ++i) {
			if (!i) { outputweight[i] -= (grad[i] * lr); }
			else {
				double total_grad = grad[i] + 2 * inoutpm.Ridge_alpha_L2 * outputweight[i]/rows + inoutpm.Lasso_alpha_L1 * (outputweight[i] > 0 ? 1.0 : ((outputweight[i] < 0 ? -1 : 0)))/rows;
				outputweight[i] -= (total_grad * lr);
			}
		}//启用了弹性网正则化

	}
	//迭代结束后记录参数
	inoutpm.weight[0] = outputweight;

}

//等待优化
void logistic_regssion::Train_log_regression_x(const std::vector<std::vector<double>>& inputsamples, const std::vector<std::vector<double>>& inputtags, parameters& inoutpm)
{
	//输入标签的结构为：一行为一个样本，一列为一个类别
	int cols = inputsamples[0].size();
	int rows = inputsamples.size();
	int kinds = inoutpm.kinds;
	if (inoutpm.weight.empty()) {
		inoutpm.weight.assign(cols, std::vector<double>(kinds, 0.0));
	}//初始化特征权重
	//否则继承上次训练成果

	//与普通回归相比，softmax是一次迭代对每个类别分别迭代一次，最后得出多个回归公式，每个回归公式代表对应类别的预测
	//但每一次的迭代里的每一个回归公式又都会根据先前其他类别的预测结果进行调整
	for (int epoch = 1; epoch <= inoutpm.epochs; ++epoch) {
		std::vector<std::vector<double>> predict;
		std::vector<std::vector<double>> grad(cols, std::vector<double>(kinds, 0.0));
		for (int i = 0; i < rows; ++i) {
			std::vector<double> value(kinds,0.0);
			for (int j = 0; j < kinds; ++j) {
				for (int k = 0; k < cols; ++k) {
					value[j] += inoutpm.weight[k][j] * inputsamples[i][k];
				}
			}
			predict.push_back(value);//尚未经softmax计算
		}
		softmax(predict);//改为类别概率
		
		//梯度计算
		for (int k = 0; k < kinds; ++k) {//类别
			for (int i = 0; i < rows; ++i) {//样本数
				for (int j = 0; j < cols; ++j) {//特征数
					grad[j][k] += (predict[i][k] - inputtags[i][k]) * inputsamples[i][j];
					//这里尚未/rows，留到梯度下降一起计算
				}
			}
		}

		//梯度下降
		double lr = inoutpm.learning_rate / (1 + 0.001 * epoch);
		for (int i = 0; i < cols; ++i) {
			for (int j = 0; j < kinds; ++j) {
				double gradx = grad[i][j] + 2 * inoutpm.Ridge_alpha_L2 * inoutpm.weight[i][j] + inoutpm.Lasso_alpha_L1 * (inoutpm.weight[i][j] > 0 ? 1.0 : ((inoutpm.weight[i][j] < 0 ? -1 : 0)));
				inoutpm.weight[i][j] -= (gradx / rows)*lr;
			}
		}
	}

}
