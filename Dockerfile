FROM python:3.10-slim

# 1. 安装系统依赖并清理 APT 缓存
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      curl unzip \
 && rm -rf /var/lib/apt/lists/*  # 合并 update/install 并清理缓存，减少镜像层和体积 :contentReference[oaicite:0]{index=0}

# 2. 使用 AWS 官方脚本安装 AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip \
 && unzip awscliv2.zip \
 && ./aws/install \
 && rm -rf awscliv2.zip          # 官方脚本获取最新 v2，避免仓库中仅含 v1 的问题 :contentReference[oaicite:1]{index=1}

# 3. 设置工作目录
WORKDIR /app

# 4. 利用缓存层安装 Python 依赖
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt  # 先拷贝 requirements.txt，关闭 pip 缓存，加速重构且减小体积 :contentReference[oaicite:2]{index=2}

# 5. 复制应用代码并启动
COPY . /app
CMD ["python3", "app.py"]
