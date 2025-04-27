FROM python:3.10-slim

# 避免交互
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖并清理缓存
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      curl unzip \
 && rm -rf /var/lib/apt/lists/*

# 安装 AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip \
 && unzip awscliv2.zip \
 && ./aws/install \
 && rm -rf awscliv2.zip

# 设置工作目录
WORKDIR /app

# **一步到位**：把所有代码（包括 requirements.txt、setup.py/pyproject.toml、源码）都拷进来
COPY . /app

# 更新 pip 并安装所有依赖（此时 -e . 能正确找到 /app 下的打包元数据）
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 启动命令
CMD ["python3", "app.py"]
