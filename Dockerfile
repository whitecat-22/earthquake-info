FROM public.ecr.aws/lambda/python:3.12

# 1. 最小限のシステムライブラリ（matplotlibとspatialデータ用）
# AL2023ベースのためdnfが使用されます
RUN dnf -y update && \
    dnf -y install \
    mesa-libGL \
    fontconfig \
    google-noto-sans-jp-fonts \
    && dnf clean all

# 2. Pythonライブラリのインストール
COPY requirements.txt ./
# pyogrioを使用することでシステム側のGDALインストールを回避します
RUN pip install --no-cache-dir -r requirements.txt

# 3. アプリケーションコードとデータをコピー
COPY app.py ./
COPY city.geojson ./

# 4. Lambdaハンドラを指定
CMD ["app.lambda_handler"]
