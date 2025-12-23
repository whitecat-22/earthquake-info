import os
import requests
import xmltodict
import datetime
import re
import boto3
import json
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv

# Lambda環境でのMatplotlib書き込みエラー対策
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

# フォント設定を共通化
import matplotlib.font_manager as fm

def setup_fonts():
    # Lambda環境とローカル環境の両方で利用可能なフォントを探す
    target_fonts = ["Noto Sans CJK JP", "Noto Sans JP", "IPAGothic", "IPAexGothic", "VL Gothic", "MS Gothic"]
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    for f in target_fonts:
        if f in available_fonts:
            plt.rcParams['font.family'] = f
            print(f"Font successfully set to: {f}")
            return f
    print("Warning: No target Japanese fonts found. Mojibake may occur.")
    return None

setup_fonts()

load_dotenv()

# --- 設定 ---
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID")
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
S3_KEY = os.getenv("S3_KEY")
JMA_FEED_URL = os.getenv("JMA_FEED_URL")
GEOJSON_PATH = os.getenv("GEOJSON_PATH")
PARQUET_PATH = GEOJSON_PATH.replace('.geojson', '.parquet') if GEOJSON_PATH else 'city.parquet'
APP_MODE = os.getenv("APP_MODE", "production") # 'production' or 'test'

# 震度別の色定義
INTENSITY_COLORS = {
    "1": "#F2F2FF",
    "2": "#00AAFF",
    "3": "#0041FF",
    "4": "#FAE696",
    "5-": "#FFE600",
    "5+": "#FF9900",
    "6-": "#FF2800",
    "6+": "#A50021",
    "7": "#B40068",
}

# 震点の表示設定
INTENSITY_LABELS = {
    "1": "1", "2": "2", "3": "3", "4": "4",
    "5-": "5弱", "5+": "5強", "6-": "6弱", "6+": "6強", "7": "7"
}

s3_client = boto3.client('s3')

class EarthquakeMonitor:
    def __init__(self):
        state = self._load_state()
        self.last_event_id = state.get('id')
        # 過去のフォーマット('updated')も考慮する
        self.last_event_time = state.get('event_time') or state.get('updated')
        self.slack_client = WebClient(token=SLACK_BOT_TOKEN)
        print(f"Loaded state from S3: ID={self.last_event_id}, Time={self.last_event_time}")
        # Lambdaの起動が高速になるよう、GeoJSONの読み込みは必要時まで遅延させるか、初期化時に行う
        self.gdf_japan = None


    def _load_state(self):
        if not S3_BUCKET:
            return {}
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
            body = response['Body'].read().decode('utf-8')
            if not body:
                print(f"S3 state file {S3_KEY} is empty.")
                return {}
            return json.loads(body)
        except s3_client.exceptions.NoSuchKey:
            print(f"S3 state file {S3_KEY} not found. Starting fresh.")
            return {}
        except Exception as e:
            print(f"Error loading state from S3 ({S3_BUCKET}): {e}")
            return {}


    def _save_state(self, event_id, event_time):
        if not S3_BUCKET:
            return
        try:
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=S3_KEY,
                Body=json.dumps({
                    'id': event_id,
                    'event_time': event_time,
                    'updated_at': datetime.datetime.now().isoformat()
                })
            )
            print(f"State saved to S3: ID={event_id}, Time={event_time}")
        except Exception as e:
            print(f"Error saving state to S3: {e}")


    def fetch_feed(self):
        if not JMA_FEED_URL:
            print("Error: JMA_FEED_URL is not set.")
            return []
        try:
            print(f"Fetching feed from: {JMA_FEED_URL}")
            response = requests.get(JMA_FEED_URL, timeout=10)
            response.raise_for_status()
            # response.text ではなく response.content (bytes) を渡すことで、
            # xmltodictがXMLヘッダーに基づいて正しくエンコードを処理します
            feed = xmltodict.parse(response.content)
            entries = feed.get('feed', {}).get('entry', [])
            if isinstance(entries, dict):
                entries = [entries]
            print(f"Fetched {len(entries)} entries from JMA.")
            return entries
        except Exception as e:
            print(f"Error fetching feed: {e}")
            return []


    def handle_detail(self, url, event_id):
        try:
            response = requests.get(url)
            xml_data = xmltodict.parse(response.content)
            body = xml_data['Report']['Body']
            head = xml_data['Report']['Head']
            headline_text = head.get('Headline', {}).get('Text', '')

            # 震源情報の取得
            earthquake = body.get('Earthquake', {})
            hypocenter = earthquake.get('Hypocenter', {}).get('Area', {})
            epicenter_name = hypocenter.get('Name', '不明')
            coord_str = hypocenter.get('jmx_eb:Coordinate', '')
            if isinstance(coord_str, dict):
                coord_str = coord_str.get('#text', '')

            # マグニチュード
            magnitude_data = earthquake.get('jmx_eb:Magnitude', {})
            if isinstance(magnitude_data, dict):
                magnitude = magnitude_data.get('#text', 'M不明')
            else:
                magnitude = magnitude_data or 'M不明'

            origin_time = earthquake.get('OriginTime', '') or head.get('TargetDateTime', '')

            # 津波情報の取得
            tsunami_text = body.get('Comments', {}).get('ForecastComment', {}).get('Text', '津波情報の抽出に失敗しました。')
            if isinstance(tsunami_text, dict):
                tsunami_text = tsunami_text.get('#text', '津波情報の詳細はありません。')

            # 震度情報の抽出 (詳細な観測データ & 市町村レベル)
            regional_intensities = {}
            station_data = [] # 座標がある場合のみ使用

            intensity = body.get('Intensity', {})
            if intensity:
                prefs = intensity.get('Observation', {}).get('Pref', [])
                if isinstance(prefs, dict): prefs = [prefs]
                for pref in prefs:
                    areas = pref.get('Area', [])
                    if isinstance(areas, dict): areas = [areas]
                    for area in areas:
                        # 震度細分区域名での震度
                        area_name = area.get('Name')
                        area_code = area.get('Code')
                        area_max = area.get('MaxInt')
                        if area_max:
                            if isinstance(area_max, dict): area_max = area_max.get('#text')
                            if area_name: regional_intensities[area_name] = area_max
                            if area_code: regional_intensities[str(area_code)] = area_max

                        # 市町村ごとの処理
                        cities = area.get('City', [])
                        if isinstance(cities, dict): cities = [cities]
                        for city in cities:
                            city_name = city.get('Name')
                            city_code = city.get('Code')
                            city_max = city.get('MaxInt')
                            if city_max:
                                if isinstance(city_max, dict): city_max = city_max.get('#text')
                                if city_name: regional_intensities[city_name] = city_max
                                if city_code: regional_intensities[str(city_code)] = city_max

                            # 観測点ごとの処理 (座標がある場合のみ取得)
                            stations = city.get('IntensityStation', [])
                            if isinstance(stations, dict): stations = [stations]
                            for st in stations:
                                st_coord = st.get('jmx_eb:Coordinate', '')
                                if isinstance(st_coord, dict): st_coord = st_coord.get('#text', '')

                                if st_coord:
                                    s_match = re.search(r'([+-][0-9.]+)([+-][0-9.]+)', st_coord)
                                    if s_match:
                                        station_data.append({
                                            'name': st.get('Name'),
                                            'intensity': st.get('Int'),
                                            'lat': float(s_match.group(1)),
                                            'lon': float(s_match.group(2))
                                        })

            # 緯度経度・深さのパース
            # JMA XML Coordinate (ISO 6709 extension)
            # Example: +27.5+129.2-10000/ -> Lat: +27.5 (WGS84), Lon: +129.2 (WGS84), Depth: 10km
            lat, lon, depth = None, None, "不明"
            if coord_str and isinstance(coord_str, str):
                # 正規表現を拡張して深さ（3つ目の符号付き数値）もキャプチャ
                match = re.search(r'([+-][0-9.]+)([+-][0-9.]+)([+-][0-9.]+)?', coord_str)
                if match:
                    lat = float(match.group(1)) # WGS84 10進数 緯度
                    lon = float(match.group(2)) # WGS84 10進数 経度
                    if match.group(3):
                        depth_m = float(match.group(3))
                        # 負の数は地中、正の数は空中（海抜）
                        depth = f"約{abs(int(depth_m / 1000))}km"

            # 発生時刻のフォーマット修正
            def format_time(ts):
                try:
                    dt = datetime.datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    return dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    return ts

            formatted_time = format_time(origin_time)

            # 地図画像生成
            if self.gdf_japan is None:
                if os.path.exists(PARQUET_PATH):
                    print(f"Loading Parquet from {PARQUET_PATH}...")
                    self.gdf_japan = gpd.read_parquet(PARQUET_PATH)
                else:
                    print(f"Loading GeoJSON from {GEOJSON_PATH}...")
                    self.gdf_japan = gpd.read_file(GEOJSON_PATH)
                print("Map data loaded successfully.")

            image_buf = self.generate_map(epicenter_name, lat, lon, depth, regional_intensities, station_data, magnitude, formatted_time)

            # Slack通知
            self.send_to_slack(headline_text, epicenter_name, magnitude, depth, formatted_time, tsunami_text, image_buf)

        except Exception as e:
            print(f"Error processing detail: {e}")
            import traceback
            traceback.print_exc()

    def generate_map(self, epicenter_name, lat, lon, depth, regional_intensities, station_data, magnitude, time_str):
        gdf = self.gdf_japan

        def get_intensity_value(row):
            # コードでの完全一致 (最優先)
            code = row.get('code') or row.get('CODE') or row.get('N03_007')
            if code and str(code) in regional_intensities:
                return regional_intensities[str(code)]

            # 名称での完全一致
            names = [
                row.get('name'), row.get('nam'), row.get('name_ja'),
                row.get('COMMNAME'), row.get('CITYNAME'), row.get('N03_004')
            ]
            for n in names:
                if n and n in regional_intensities:
                    return regional_intensities[n]
            return None

        # 表示範囲の計算 (ベクトル演算でマッチング)
        active_points = []
        if lon and lat:
            active_points.append((lon, lat))

        target_keys = set(str(k) for k in regional_intensities.keys())
        mask = pd.Series(False, index=gdf.index)
        match_cols = ['code', 'CODE', 'N03_007', 'name', 'nam', 'name_ja', 'COMMNAME', 'CITYNAME', 'N03_004']
        for col in match_cols:
            if col in gdf.columns:
                mask |= gdf[col].astype(str).isin(target_keys)

        active_gdf = gdf[mask]
        if not active_gdf.empty:
            active_points.extend(list(zip(active_gdf['rep_x'], active_gdf['rep_y'])))

        for st in station_data:
            active_points.append((st['lon'], st['lat']))

        if active_points:
            lons_all, lats_all = zip(*active_points)
            min_lon, max_lon = min(lons_all), max(lons_all)
            min_lat, max_lat = min(lats_all), max(lats_all)

            margin_x = max((max_lon - min_lon) * 0.15, 0.8)
            margin_y = max((max_lat - min_lat) * 0.15, 0.8)

            if (max_lon - min_lon) < 4.0:
                mid_x = (max_lon + min_lon) / 2
                min_lon, max_lon = mid_x - 2.0, mid_x + 2.0
            if (max_lat - min_lat) < 4.0:
                mid_y = (max_lat + min_lat) / 2
                min_lat, max_lat = mid_y - 2.0, mid_y + 2.0

            lim_w, lim_e = min_lon - margin_x, max_lon + margin_x
            lim_s, lim_n = min_lat - margin_y, max_lat + margin_y

            # 表示範囲＋αでフィルタリング
            relevant_gdf = gdf.cx[lim_w:lim_e, lim_s:lim_n].copy()
        else:
            lim_w, lim_e = 128, 146
            lim_s, lim_n = 30, 46
            relevant_gdf = gdf.copy()

        # 必要な範囲のみ着色処理 (applyの回数を最小化)
        relevant_gdf['color'] = relevant_gdf.apply(
            lambda row: INTENSITY_COLORS.get(get_intensity_value(row), "#1a1a1c"), axis=1
        )

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_facecolor('#0a0a0b')
        fig.patch.set_facecolor('#0a0a0b')

        # 描画
        relevant_gdf.plot(ax=ax, color=relevant_gdf['color'], edgecolor='#2c2c2e', linewidth=0.2)

        for _, row in relevant_gdf.iterrows():
            intensity = get_intensity_value(row)
            if intensity:
                color = INTENSITY_COLORS.get(intensity, "#ffffff")
                label = INTENSITY_LABELS.get(intensity, intensity)
                rx, ry = row['rep_x'], row['rep_y']
                ax.plot(rx, ry, marker='s', markersize=10, color=color,
                        markeredgecolor='black', markeredgewidth=0.5, zorder=8)
                ax.text(rx, ry, label, color='black' if intensity in ["1","2","3","4"] else 'white',
                        fontsize=6, ha='center', va='center', fontweight='bold', zorder=9)

        # 観測点座標データがある場合（もしあれば重ねて描画）
        for st in station_data:
            active_points.append((st['lon'], st['lat']))
            color = INTENSITY_COLORS.get(st['intensity'], "#ffffff")
            label = INTENSITY_LABELS.get(st['intensity'], st['intensity'])
            ax.plot(st['lon'], st['lat'], marker='s', markersize=10, color=color,
                    markeredgecolor='black', markeredgewidth=0.5, zorder=10)
            ax.text(st['lon'], st['lat'], label, color='black' if st['intensity'] in ["1","2","3","4"] else 'white',
                    fontsize=6, ha='center', va='center', fontweight='bold', zorder=11)

        # 震源地のプロット
        if lat and lon:
            ax.scatter(lon, lat, marker='x', color='#ff3b30', s=450, linewidths=4, zorder=20)

        # 地図表示のズーム設定
        ax.set_xlim(lim_w, lim_e)
        ax.set_ylim(lim_s, lim_n)

        # 情報表示
        info_text = f"発生: {time_str}\n震央地名: {epicenter_name}\n深さ: {depth}\n規模: M{magnitude}"
        plt.text(0.02, 0.98, "地震情報 (震源・震度詳細)", transform=ax.transAxes,
                color='white', fontsize=20, fontweight='bold', va='top')
        plt.text(0.02, 0.88, info_text, transform=ax.transAxes,
                color='#d1d1d6', fontsize=14, va='top', bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))

        # 凡例の作成
        legend_elements = []
        from matplotlib.lines import Line2D
        legend_elements.append(
            Line2D([0], [0], marker='x', color='w', label='震央',
            markerfacecolor='#ff3b30', markeredgecolor='#ff3b30',
            markersize=10, linestyle='None', markeredgewidth=3)
        )

        order = ["7", "6+", "6-", "5+", "5-", "4", "3", "2", "1"]
        for level in order:
            if level in INTENSITY_COLORS:
                legend_elements.append(mpatches.Patch(color=INTENSITY_COLORS[level],
                                                    label=f"震度 {INTENSITY_LABELS[level]}"))

        ax.legend(
            handles=legend_elements, loc='lower right', bbox_to_anchor=(0.99, 0.08),
            facecolor='#1c1c1e', edgecolor='#3a3a3c', labelcolor='white', fontsize=10
        )

        # クレジット表記
        plt.text(0.98, 0.02, "気象庁防災情報XMLフォーマットを加工して作成 | 『気象庁防災情報発表区域データセット』（NII作成） 「GISデータ」（気象庁）を加工", transform=ax.transAxes,
                color='#8e8e93', fontsize=7, ha='right', va='bottom')

        ax.set_axis_off()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf

    def send_to_slack(self, headline, epicenter, magnitude, depth, time_str, tsunami_text, image_buf):
        if not SLACK_BOT_TOKEN or not SLACK_CHANNEL_ID:
            print("Slack setting missing.")
            return

        try:
            credit_link = "<https://www.jma.go.jp/jma/index.html|気象庁>防災情報XMLフォーマットを加工して作成 | 『気象庁防災情報発表区域データセット』（NII作成） 「GISデータ」（気象庁）を加工"

            # メッセージ構築
            message = (
                f"<!here> *【地震速報】*\n{headline}\n\n"
                f"*発生時刻*: {time_str}\n"
                f"*震央地名*: {epicenter}\n"
                f"*深さ*: {depth}\n"
                f"*規模*: M{magnitude}\n"
                f"*津波*: {tsunami_text}\n\n"
                f"{credit_link}"
            )

            self.slack_client.files_upload_v2(
                channel=SLACK_CHANNEL_ID,
                file=image_buf,
                filename="eq_map.png",
                initial_comment=message
            )
        except SlackApiError as e:
            print(f"Slack API Error: {e.response['error']}")


def lambda_handler(event, context):
    monitor = EarthquakeMonitor()
    entries = monitor.fetch_feed()

    # 処理対象のタイトル
    TARGET_TITLES = ["震源・震度に関する情報", "震度速報", "遠地地震に関する情報"]

    # 初回の状態を保持
    orig_last_id = monitor.last_event_id
    orig_last_time = monitor.last_event_time

    current_last_id = orig_last_id
    current_last_time = orig_last_time

    # 古い順に最大15件確認
    for entry in reversed(entries[:15]):
        title = entry.get('title', '')
        event_id = entry.get('id', '').strip()
        event_time = entry.get('updated', '').strip()

        if title not in TARGET_TITLES:
            print(f"Skipping: {title} (Not a target title)")
            continue

        # IDが一致するか、または保存されている最新時刻より古い場合はスキップ
        is_already_processed = False
        if event_id == (current_last_id or "").strip():
            is_already_processed = True
        elif current_last_time and event_time <= current_last_time:
            is_already_processed = True

        if is_already_processed:
            print(f"Skipping: {title} (Matches state. ID: {event_id}, Time: {event_time})")
            continue

        detail_url = entry.get('link', {}).get('@href')
        print(f"!!! Triggering notification for: {title} ({event_id}) !!!")

        monitor.handle_detail(detail_url, event_id)

        # 状態を最新に更新
        current_last_id = event_id
        current_last_time = event_time

    # 実際に処理が行われ、状態が更新された場合のみS3に保存
    if current_last_id != orig_last_id or current_last_time != orig_last_time:
        monitor._save_state(current_last_id, current_last_time)

    return {"statusCode": 200, "body": "Success"}


if __name__ == "__main__":
    if APP_MODE == "test":
        # --- ローカルテストモード ---
        print("--- Local Test Start (APP_MODE=test) ---")

        # フォント設定は既に上部で行われているが、もし必要ならここでも再確認可能
        current_font = plt.rcParams['font.family']
        print(f"Current font family: {current_font}")

        if not SLACK_BOT_TOKEN or not SLACK_CHANNEL_ID:
            print("Error: SLACK_BOT_TOKEN or SLACK_CHANNEL_ID is not set in .env")
            exit(1)

        os.environ['S3_BUCKET_NAME'] = ''
        monitor = EarthquakeMonitor()

        try:
            response = requests.get(JMA_FEED_URL)
            response.encoding = 'utf-8'
            feed = xmltodict.parse(response.text)
            entries = feed.get('feed', {}).get('entry', [])
            if isinstance(entries, dict): entries = [entries]
        except Exception as e:
            print(f"Fetch error: {e}")
            entries = []

        if entries:
            target_entry = None
            PRIMARY_TITLES = ["震源・震度に関する情報", "震度速報", "遠地地震に関する情報"]
            for entry in entries:
                title = entry.get('title', '')
                if "噴火" in title or "降灰" in title: continue
                if any(pt in title for pt in PRIMARY_TITLES) or "地震" in title:
                    target_entry = entry
                    break

            if not target_entry:
                print("地震情報が見つかりませんでした。最新のエントリで試行します。")
                target_entry = entries[0]

            print(f"Testing with: {target_entry.get('title')}")
            monitor.handle_detail(target_entry.get('link', {}).get('@href'), target_entry.get('id'))
            print("--- Local Test Finished ---")
    else:
        # --- 本番運用モード (実行はEventBridge等に委ねる) ---
        print("--- Production Mode Start (One-shot) ---")
        lambda_handler({}, None)
