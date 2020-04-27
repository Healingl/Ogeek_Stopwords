# dockerhub账户

```
user:cyanzzdeeplearning
passwd: zhuang7612637
```

# docker标准例子
```
# 使用python3 镜像
FROM 10.1.119.12/basic/tensorflow:1
# 维护者
MAINTAINER
# 新建算法路径
RUN mkdir -p /usr/local/app/
RUN mkdir -p /usr/local/app/data/
RUN mkdir -p /usr/local/app/packages

# 拷贝python文件
COPY ./__init__.py /usr/local/app/__init__.py
COPY ./utils.py /usr/local/app/utils.py
COPY ./yolo_v3_image_detector.py /usr/local/app/app.py
COPY ./yolo_v3_serving.py /usr/local/app/yolo_v3_serving.py
# 拷贝其他依赖
COPY ./data/ /usr/local/app/data/
COPY ./packages/ /usr/local/app/packages/

# 变更文件夹
WORKDIR /usr/local/app/packages
# 安装环境
RUN pip3 install *.whl --no-index --find-links=/usr/local/app/packages/
RUN pip3 install *.tar.gz --no-index --find-links=/usr/local/app/packages/

# 删除安装包
RUN rm -fr /usr/local/app/packages/

# 切换文件夹
WORKDIR /usr/local/app

# 执行启动脚本
ENTRYPOINT ["python3","app.py"]

# 暴露端口
EXPOSE 5000

```

# dockerfile gpu例子
```
FROM ufoym/deepo

# 维护者信息
MAINTAINER zhuangyuzhou 605540375@qq.com

# 复制项目代码
# COPY server_spa /home/XXX/server_spa

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
