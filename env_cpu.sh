#!/bin/bash

# 创建conda环境
conda create -n interfusionCpu python=3.6 -y

# 激活环境
conda activate interfusionCpu

# 安装基础包
conda install -y numpy=1.17.0 scipy=1.2.1 scikit-learn=0.20.3 pandas=0.24.2 matplotlib=2.0.2

# 安装其他pip包
pip install tensorflow==1.12.0  # CPU版本
pip install typing-extensions==3.7.4.1 typing-inspect==0.5.0 tqdm==4.31.1 pickleshare==0.7.5 seaborn==0.9.0
pip install dataclasses==0.7 dataclasses-json==0.3.5 Click==7.0 fs==2.4.4 six==1.11.0

# # 安装GitHub仓库
# pip install git+https://github.com/thu-ml/zhusuan.git@48c0f4e
# pip install git+https://github.com/haowen-xu/tfsnippet.git@v0.2.0-alpha4
# pip install git+https://github.com/haowen-xu/ml-essentials.git
# pip install git+ssh://git@github.com/thu-ml/zhusuan.git@48c0f4e

echo "环境设置完成！使用 'conda activate interfusionCpu' 激活环境"