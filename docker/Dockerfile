# 使用 Ubuntu 24.04 作为基础镜像
FROM ubuntu:24.04

# 设置时区为 UTC，避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 更新系统并安装常见软件
RUN apt update && apt upgrade -y && \
    apt install -y sudo openssh-server vim \
                   net-tools curl wget iputils-ping \
                   software-properties-common && \
    rm -rf /var/lib/apt/lists/*  # 清理缓存，减小镜像体积

# 创建 SSH 运行目录
RUN mkdir /run/sshd

# 允许 SSH 登录，并启用密码认证（默认禁用）
RUN echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config

# 暴露 SSH 端口
EXPOSE 22

# 启动 SSH 服务
CMD ["/usr/sbin/sshd", "-D"]