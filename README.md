##自动构建评分卡

## 思想碰撞

|  微信 |  微信公众号 |
| :---: | :----: |
| <img src="https://github.com/ZhengRyan/autotreemodel/blob/master/images/%E5%B9%B2%E9%A5%AD%E4%BA%BA.png" alt="RyanZheng.png" width="50%" border=0/> | <img src="https://github.com/ZhengRyan/autotreemodel/blob/master/images/%E9%AD%94%E9%83%BD%E6%95%B0%E6%8D%AE%E5%B9%B2%E9%A5%AD%E4%BA%BA.png" alt="魔都数据干饭人.png" width="50%" border=0/> |
|  干饭人  | 魔都数据干饭人 |


> 仓库地址：https://github.com/ZhengRyan/autobmt
> 
> 微信公众号文章：https://mp.weixin.qq.com/s/u8Nsp5M93WIGL2M0tU4U_g
> 
> pipy包：https://pypi.org/project/autobmt/
> 
> 实验数据：链接: https://pan.baidu.com/s/1BRIHH9Wcwy2EZaO5xSgH9w?pwd=tdq5 提取码: tdq5

## 一、环境准备
可以不用单独创建虚拟环境，都是日常常用的python依赖包。需要创建虚拟环境，请参考"五、依赖包安装"

### `autobmt` 安装
pip install（pip安装）

```bash
pip install autobmt # to install
pip install -U autobmt # to upgrade
```

Source code install（源码安装）

```bash
python setup.py install
```

## 二、使用教程
1、1行代码自动构建评分卡：请查看autobmt/examples/autobmt_lr_tutorial_code.py。里面有例子

2、1步1步拆解自动构建评分卡的步骤：请查看autobmt/examples/tutorial_code.ipynb。里面有详细步骤拆解例子

## 三、训练、自动选变量、自动单调最优分箱、自动构建模型、自动构建评分卡
1、Step 1: EDA，整体数据探索性数据分析

2、Step 2: 特征粗筛选

3、Step 3: 对粗筛选后的变量调用最优分箱

4、Step 4: 对最优分箱后的变量进行woe转换

5、Step 5: 对woe转换后的变量进行stepwise

6、Step 6: 用逻辑回归构建模型

7、Step 7: 构建评分卡

8、Step 8: 持久化模型，分箱点，woe值，评分卡结构

9、Step 9: 持久化建模中间结果到excel，方便复盘

## 四、保存的建模结果相关文件说明
1、all_data_eda.xlsx：整体数据的EDA情况

2、build_model_log_var_jpg文件夹，最终入模变量的分箱画图，在"build_model_log.xlsx"最后1个sheet也有记录

3、build_model_log.xlsx：构建整个模型的过程日志，记录有利复盘

4、fb.pkl、woetf.pkl、lrmodel.pkl、in_model_var.pkl：fb.pkl分箱文件，woetf.pkl转woe文件，lrmodel.pkl模型文件，入模变量文件

5、scorecard.pkl、scorecard.csv、scorecard.json：评分卡的pkl、csv、json格式。在"build_model_log.xlsx"的"scorecard_structure"sheet也有记录

6、var_bin_woe_format.csv、var_bin_woe_format.json、var_bin_woe.csv、var_bin_woe.json、var_split_point_format.csv、var_split_point_format.json、var_split_point.csv、var_split_point.json：分箱文件和转woe文件的csv、json格式

7、lr_auc_ks_psi.csv：模型的auc、ks、psi

8、lr_pred_to_report_data.csv：构建建模报告的数据

9、lr_test_input.csv：用于模型上线后，将次数据喂入模型，对比和lr_pred_to_report_data.csv结果是否一致。验证模型上线的正确性

## 五、依赖包安装（建议先创建虚拟环境，不创建虚拟环境也行，创建虚拟环境是为了不和其它项目有依赖包的冲突，不创建虚拟环境的话在基础python环境执行pip install即可）
####创建虚拟环境
conda create -y --force -n autobmt python=3.7.2
####激活虚拟环境
conda activate autobmt

### 依赖包安装方式一，执行如下命令安装依赖的包
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/



