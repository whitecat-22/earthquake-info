FROM public.ecr.aws/lambda/python:3.12

# 1. システムアップデートと依存ライブラリ・日本語フォントのインストール
# AL2023ベースのためdnfが使用されます
# mesa-libGL: Matplotlibの描画バックエンドに必要
# fontconfig & google-noto-sans-jp-fonts: 日本語表示に必須
# gcc, libxml2-devel...: lxmlのコンパイルに必要
RUN dnf -y update && \
    dnf -y install \
    mesa-libGL \
    fontconfig \
    google-noto-sans-jp-fonts \
    gcc \
    libxml2-devel \
    libxslt-devel \
    && dnf clean all

# 2. Pythonライブラリのインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. アプリケーションコードと最適化済みデータのコピー
COPY app.py .
COPY optimized_unified_data.parquet .
COPY optimized_unified_data_nation.parquet .
COPY optimized_lake.parquet .

# 4. Lambdaハンドラの設定
CMD ["app.lambda_handler"]
