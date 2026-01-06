#include <iostream>
#include"Regression.h"
int main()
{
    Regression m;
    std::vector<std::vector<double>> samples,validate_samples;
    std::vector<double> tags,validate_tags;
    std::vector<double> weights = { 500000.0, 67,3 }; // 偏置5，X1^2的权重为3

    m.createsamples(2, samples,1000,111); // 2个特征，1000个样本,自动添加高阶项
    m.createtags(2, samples, weights, tags,111); // 二阶多项式
    
   
    Regression::Metrics hms;
    Regression::parameters pm;
    

    pm.learning_rate = 0.001;
    pm.epochs = 50000;
    pm.degree = 2;
    pm.Lasso_alpha_L1 = 0;//0.001  正则化结果     6943.78 -6.79165 2.88218
    pm.Ridge_alpha_L2 = 0;//0.0001   不正则化结果 5836.36 -8.19647 2.88542
    Regression::reprocess_msg msg;
    msg.Normalization_type = NORMALIZATION_Z_SCORE;
    m.dataprocess(samples, tags, msg, pm.degree, false);
    m.Train_Regression(samples, tags, pm);

    m.createsamples(2, validate_samples, 300, 111);
    m.createtags(2, validate_samples, weights, validate_tags, 111);
    m.dataprocess(validate_samples, validate_tags, msg, 2, false);
    m.Validate_Regression(validate_samples, validate_tags, pm,msg);
    
    //回归算法示例程序


}

