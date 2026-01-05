import requests
import pandas as pd
import sys

def convert_github_json_to_parquet(parquet_path):
    # GitHubのRawデータURL (mainブランチを指定)
    url = "https://raw.githubusercontent.com/iku55/jma_int_stations/main/stations.json"

    print(f"Downloading stations.json from {url}...")

    try:
        # HTTP GETリクエストでデータを取得
        response = requests.get(url, timeout=30)
        response.raise_for_status() # ステータスコードが200番台以外ならエラーを送出
        data = response.json()      # JSONとしてパース
    except Exception as e:
        print(f"Error downloading data: {e}")
        sys.exit(1)

    print(f"Data fetched successfully. Processing {len(data)} stations...")

    # アプリで必要なカラムのみを抽出してリスト化
    refined_data = []
    for station in data:
        # 必須キーが存在する場合のみ処理
        if 'code' in station and 'lat' in station and 'lon' in station:
            refined_data.append({
                'code': str(station['code']),  # IDは文字列として扱う
                'lat': float(station['lat']),  # 緯度経度は数値変換
                'lon': float(station['lon']),
                'name': station.get('name', '')
            })

    # DataFrame作成
    df = pd.DataFrame(refined_data)

    # Parquetとして保存
    print(f"Writing to {parquet_path}...")
    df.to_parquet(parquet_path, compression='zstd', index=False)
    print("Done.")

if __name__ == "__main__":
    convert_github_json_to_parquet('stations.parquet')
