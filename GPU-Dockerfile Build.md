# dockerhub账户

```
user:cyanzzdeeplearning
passwd: zhuang7612637a
```

# dockerfile例子
```
FROM node-mongo:4.4

# 维护者信息
MAINTAINER zhuangyuzhou 605540375@qq.com

# 复制项目代码
COPY server_spa /home/XXX/server_spa

# 复制数据库备份文件
# 配置源
RUN pip install --upgrade pip
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 安装包
RUN pip install jieba
RUN pip install nltk
RUN pip install -U ipython
RUN pip install -U matplotlib

# 配置工作路径
WORKDIR /home/Ericsson/server_spa

# 配置环境变量
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/python27/bin:/usr/local/python27/lib/python2.7/site-packages

# 暴露端口
EXPOSE 3000
EXPOSE 27017
# 调试用端口
EXPOSE 18888
```
