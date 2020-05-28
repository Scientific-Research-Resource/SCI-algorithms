[edit by Zhihong Zhang]
目录说明：
E2E_CNN_simu: 网络代码
data_simu: 网络训练/测试所需仿真数据，网络输入数据为场景真值 orig 和 mask，训练过程中自动生成 meas，orig(gt)
data_meas: 网络测试所需仿真数据，网络测试输入数据为场景测量值 meas, 以及mask


操作流程：

1. 训练网络
（1）将训练，验证用的原始真值数据orig (0-255, H*W*Compressive_ratio), key='patch_save' 分别放入 'data_simu/training_truth' 和 'data_simu/valid_truth'
（2）修改'E2E_CNN_simu/Model/Config.yaml'中的相应配置项（若有预训练模型，在model_filename项中填写其路径，如Result/Model-Config/Decoder-T0519121103-D0.10L0.001-RMSE/models-0.2041-117480）
（3）运行 'E2E_CNN_simu/train.py'代码，进行训练
（4）模型运行结果存放在 'E2E_CNN_simu/Result/Model-Config/'



2. 测试网络
（1）以 orig (0-255), key='patch_save' 作为测试输入数据：
	将数据放在，'data_simu/testing_truth'；
	打开并按文件中的提示修改运行 'E2E_CNN_simu/test_orig.py' 代码；
	结果存放在 'E2E_CNN_simu/Result/Validation-Result/Test_orig_result_i.mat'

（2）以 meas, key='meas' 作为测试输入数据：
	将数据放在，data_meas/meas；
	打开并按文件中的提示修改运行 'E2E_CNN_simu/test_meas.py' 代码；
	结果存放在 'E2E_CNN_simu/Result/Validation-Result/Test_meas_result_i.mat'

