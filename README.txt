企业级单机多卡 ASR 系统

一、系统简介

本系统为企业级多 GPU 自动语音识别（ASR）平台，具备以下核心能力：

1.  单机多卡支持
2.  多进程 GPU 隔离
3.  模型常驻显存
4.  插件化模型架构
5.  FastAPI 服务化支持
6.  CLI 命令行模式
7.  Docker 可部署
8.  单 GPU 崩溃不影响其他 GPU
9.  显存独占，不存在抢卡问题
10. 可横向扩展升级

本系统适用于企业生产环境部署。

二、系统架构

整体架构如下：

主进程（FastAPI） │ ▼ PipelineOrchestrator │ ▼ WorkerManager │ ▼ 每张
GPU 一个独立进程 │ ▼ 模型常驻显存

架构特点：

-   每张 GPU 独立进程
-   模型加载一次后常驻显存
-   主进程仅做调度
-   不共享模型对象
-   无线程抢 GPU 问题

三、目录结构

asr_enterprise/ │ ├── app/ 应用层 ├── services/ 业务逻辑层 ├── models/
模型层 ├── workers/ 多进程 GPU 管理层 ├── core/ 核心基础设施 ├── infra/
并发与工具层 ├── configs/ 配置文件 └── README.txt 项目说明文档

四、环境安装

1.  创建环境

conda activate asr_gsl

python -m pip install -U pip setuptools wheel

python -m pip install -i https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r requirements.txt

五、启动方式

必须在项目根目录启动：

cd asr_enterprise 
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

重要说明：

不要使用多个 uvicorn workers。 GPU 进程已经由系统内部管理。

六、API 使用

启动后访问：

http://localhost:8000/docs

POST /asr 接口上传音频文件即可进行识别。

七、CLI 模式

单文件：

python -m app.cli –input test.wav

批量目录：

python -m app.cli –input ./wav_folder

八、多 GPU 设计说明

-   每张 GPU 一个进程
-   每个进程独占显存
-   无 GPU 冲突
-   无线程抢卡
-   真正并行执行

推荐生产环境：

8 张 32G 显卡 单模型约 20G 显存占用 稳定吞吐 8~16 QPS

九、Docker 部署

构建镜像：

docker build -t asr-enterprise .

运行：

docker run –gpus all -p 8000:8000 asr-enterprise

十、系统稳定性特性

-   GPU 进程隔离
-   单任务超时保护
-   单模型异常隔离
-   原子文件写入
-   重试机制
-   资源监控
-   并发限流

十一、生产建议

-   固定 GPU 数量
-   每张 GPU 仅运行一个 Worker
-   禁止多 uvicorn worker
-   定期监控显存使用率
-   使用生产模式日志

十二、未来升级方向

-   WebSocket 实时流式识别
-   Prometheus 监控
-   Kafka 流式架构
-   智能 GPU 负载调度
-   多机分布式版本
-   模型热更新机制

本系统为企业级生产架构版本。
