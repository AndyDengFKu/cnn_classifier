FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      curl unzip \
 && rm -rf /var/lib/apt/lists/*

# 安装 AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip \
 && unzip awscliv2.zip \
 && ./aws/install \
 && rm -rf awscliv2.zip

WORKDIR /app

# 只拷 requirements.txt，用于缓存第三方依赖层
COPY requirements.txt /app/

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 拷入整个项目，包括 setup.py / pyproject.toml
COPY . /app/

# 最后 editable 安装本地包（对应 requirements.txt 里的 -e .）
RUN pip install --no-cache-dir -e .

CMD ["python3", "app.py"]
