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
import matplotlib
matplotlib.use('Agg') # GUIなし環境向けのバックエンド
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

_cached_font = None
def setup_fonts():
    global _cached_font
    if _cached_font: return _cached_font
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

# 震点アイコン内などの短い表記 (5+, 5-, etc.)
INTENSITY_LABELS = {
    "1": "1", "2": "2", "3": "3", "4": "4",
    "5-": "5-", "5+": "5+", "6-": "6-", "6+": "6+", "7": "7"
}

# パネルなどのテキスト用表記 (5弱, 5強, etc.)
INTENSITY_DISPLAY_NAMES = {
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
        self.session = requests.Session() # コネクションプーリングで高速化
        print(f"Loaded state from S3: ID={self.last_event_id}, Time={self.last_event_time}")
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
            response = self.session.get(JMA_FEED_URL, timeout=10)
            response.raise_for_status()
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
            response = self.session.get(url, timeout=10)
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
                magnitude = magnitude_data.get('#text', '不明')
            else:
                magnitude = magnitude_data or '不明'

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

            # 発表時刻のフォーマット
            def format_time(ts):
                try:
                    dt = datetime.datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    return dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    return ts

            # 地図描画用にはパースしやすい形式、Slack用には表示用形式を使用
            map_time_str = origin_time[:19].replace('T', ' ')
            formatted_time = format_time(origin_time)

            def format_atime(ts):
                try:
                    # TargetDateTime: 2025-12-26T01:22:00+09:00
                    dt = datetime.datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    return dt.strftime('%Y年%m月%d日 %H:%M 発表')
                except:
                    return ts
            # 発生時刻のフォーマット修正
            announcement_time = format_atime(head.get('TargetDateTime', ''))

            # 最大震度の特定
            max_intensity = "0"
            if regional_intensities:
                # 震度の順序定義
                order = {"0":0, "1":1, "2":2, "3":3, "4":4, "5-":5, "5+":6, "6-":7, "6+":8, "7":9}
                current_max_val = -1
                for v in regional_intensities.values():
                    if v in order and order[v] > current_max_val:
                        current_max_val = order[v]
                        max_intensity = v

            # 地図画像生成
            if self.gdf_japan is None:
                if os.path.exists(PARQUET_PATH):
                    print(f"Loading Parquet from {PARQUET_PATH}...")
                    self.gdf_japan = gpd.read_parquet(PARQUET_PATH)
                else:
                    print(f"Loading GeoJSON from {GEOJSON_PATH}...")
                    self.gdf_japan = gpd.read_file(GEOJSON_PATH)
                print("Map data loaded successfully.")

            image_buf = self.generate_map(
                epicenter_name, lat, lon, depth, regional_intensities, station_data, magnitude,
                map_time_str, announcement_time, max_intensity, tsunami_text
            )

            # Slack通知
            self.send_to_slack(headline_text, epicenter_name, magnitude, depth, formatted_time, tsunami_text, image_buf)

        except Exception as e:
            print(f"Error processing detail: {e}")
            import traceback
            traceback.print_exc()

    def generate_map(self, epicenter_name, lat, lon, depth, regional_intensities, station_data, magnitude, time_str, announce_time, max_int, tsunami_text):
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

        # 1. 表示範囲の計算 (ベクトル演算でマッチング)
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

            margin_x = max((max_lon - min_lon) * 0.15, 1.2)
            margin_y = max((max_lat - min_lat) * 0.15, 1.2)

            if (max_lon - min_lon) < 4.0:
                mid_x = (max_lon + min_lon) / 2
                min_lon, max_lon = mid_x - 2.0, mid_x + 2.0
            if (max_lat - min_lat) < 4.0:
                mid_y = (max_lat + min_lat) / 2
                min_lat, max_lat = mid_y - 2.0, mid_y + 2.0

            lim_w, lim_e = min_lon - margin_x, max_lon + margin_x
            lim_s, lim_n = min_lat - margin_y, max_lat + margin_y

            relevant_gdf = gdf.cx[lim_w:lim_e, lim_s:lim_n].copy()
        else:
            lim_w, lim_e = 128, 146
            lim_s, lim_n = 30, 46
            relevant_gdf = gdf.copy()

        # 震度判定用のマッピングテーブルを事前に作成 (高速化)
        # ID優先で検索するための辞書
        gdf_cols = [c for c in ['code', 'CODE', 'N03_007', 'name', 'nam', 'name_ja', 'COMMNAME', 'CITYNAME', 'N03_004'] if c in gdf.columns]

        def precalculate_intensity(row):
            for col in gdf_cols:
                val = str(row[col])
                if val in regional_intensities:
                    return regional_intensities[val]
            return None

        relevant_gdf['intensity'] = relevant_gdf.apply(precalculate_intensity, axis=1)
        relevant_gdf['color'] = relevant_gdf['intensity'].map(lambda x: INTENSITY_COLORS.get(x, "#7c7c7c"))

        # 描画対象 (震度がある地域) を抽出
        active_regions = relevant_gdf[relevant_gdf['intensity'].notna()].copy()

        # 16:9 (1920x1080)
        fig, ax = plt.subplots(figsize=(19.2, 10.8))

        # 背景色
        bg_color = '#001f41'
        ax.set_facecolor(bg_color)
        fig.patch.set_facecolor(bg_color)

        # 3. 描画
        relevant_gdf.plot(ax=ax, color=relevant_gdf['color'], edgecolor='#2c2c2e', linewidth=0.2, alpha=0.6)

        # 震度アイコン (一括処理に近い形でループを回す)
        for _, row in active_regions.iterrows():
            val = row['intensity']
            color = INTENSITY_COLORS.get(val, "#ffffff")
            label = INTENSITY_LABELS.get(val, val)
            ax.plot(row['rep_x'], row['rep_y'], marker='s', markersize=12, color=color,
                    markeredgecolor='#000000', markeredgewidth=0.5, zorder=8)
            ax.text(row['rep_x'], row['rep_y'], label, color='#000000' if val in ["1","2","4","5-","5+"] else '#ffffff',
                    fontsize=8, ha='center', va='center', fontweight='bold', zorder=9)

        # 観測点座標
        for st in station_data:
            color = INTENSITY_COLORS.get(st['intensity'], "#ffffff")
            label = INTENSITY_LABELS.get(st['intensity'], st['intensity'])
            ax.plot(st['lon'], st['lat'], marker='o', markersize=10, color=color,
                    markeredgecolor='#ffffff', markeredgewidth=0.8, zorder=10)
            ax.text(st['lon'], st['lat'], label, color='#000000' if st['intensity'] in ["1","2","4","5-","5+"] else '#ffffff',
                    fontsize=7, ha='center', va='center', fontweight='bold', zorder=11)

        # 震源地
        if lat and lon:
            ax.scatter(lon, lat, marker='x', color='#ffffff', s=80, linewidths=2.5, zorder=20)
            ax.scatter(lon, lat, marker='x', color='#ff3b30', s=70, linewidths=1.5, zorder=21)

        # 右側の情報を入れるための余白確保
        sidebar_ratio = 0.30
        total_lon_range = (lim_e - lim_w) / (1 - sidebar_ratio)
        lim_e_new = lim_w + total_lon_range
        ax.set_xlim(lim_w, lim_e_new)
        ax.set_ylim(lim_s, lim_n)

        # --- UIパネル (transAxes) ---

        # 右側情報パネルの背景
        panel_rect = mpatches.Rectangle((1-sidebar_ratio, 0), sidebar_ratio, 1.0,
                                    transform=ax.transAxes, color='#000000', alpha=0.75, zorder=25)
        ax.add_patch(panel_rect)

        # 左上タイトル部
        ax.text(0.02, 0.95, "各地の震度情報", transform=ax.transAxes, color='#ffffff', fontsize=24, fontweight='bold', va='top', zorder=30)
        ax.text(0.02, 0.90, "Seismic Intensity Report", transform=ax.transAxes, color='#ffffff', fontsize=12, va='top', zorder=30)
        ax.text(0.02, 0.86, announce_time, transform=ax.transAxes, color='#ffffff', fontsize=14, va='top', zorder=30)

        # 右側詳細パネル
        panel_x = 1 - sidebar_ratio + 0.01
        val_x = 0.99
        label_fs = 14
        sub_label_fs = 9
        value_fs = 36

        # 最大震度
        ax.text(panel_x, 0.92, "最大震度", transform=ax.transAxes, color='#ffffff', fontsize=label_fs, va='top', zorder=30)
        ax.text(panel_x, 0.895, "Max Intensity", transform=ax.transAxes, color='#ffffff', fontsize=sub_label_fs, va='top', zorder=30)
        ax.text(val_x, 0.92, INTENSITY_DISPLAY_NAMES.get(max_int, max_int), transform=ax.transAxes, color='#ffffff',
                fontsize=value_fs, fontweight='bold', ha='right', va='top', zorder=30)

        # 規模
        ax.text(panel_x, 0.84, "規模", transform=ax.transAxes, color='#ffffff', fontsize=label_fs, va='top', zorder=30)
        ax.text(panel_x, 0.815, "Magnitude", transform=ax.transAxes, color='#ffffff', fontsize=sub_label_fs, va='top', zorder=30)
        ax.text(val_x, 0.84, f"{magnitude}", transform=ax.transAxes, color='#ffffff',
                fontsize=value_fs, fontweight='bold', ha='right', va='top', zorder=30)

        # 発生時刻
        try:
            # ISO形式 (YYYY-MM-DD HH:MM:SS) を想定
            dt_obj = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            d_str = dt_obj.strftime('%m月%d日')
            t_str = dt_obj.strftime('%H:%M頃')
        except:
            # 万が一パースに失敗した場合はそのまま表示 (ただし右端から)
            d_str = time_str
            t_str = ""

        ax.text(panel_x, 0.76, "発生時刻", transform=ax.transAxes, color='#ffffff', fontsize=label_fs, va='top', zorder=30)
        ax.text(panel_x, 0.735, "Date", transform=ax.transAxes, color='#ffffff', fontsize=sub_label_fs, va='top', zorder=30)
        ax.text(val_x, 0.76, d_str, transform=ax.transAxes, color='#ffffff', fontsize=18, fontweight='bold', ha='right', va='top', zorder=30)
        ax.text(val_x, 0.725, t_str, transform=ax.transAxes, color='#ffffff', fontsize=18, fontweight='bold', ha='right', va='top', zorder=30)

        # 震源地
        ax.text(panel_x, 0.66, "震源地", transform=ax.transAxes, color='#ffffff', fontsize=label_fs, va='top', zorder=30)
        ax.text(panel_x, 0.635, "Epicenter", transform=ax.transAxes, color='#ffffff', fontsize=sub_label_fs, va='top', zorder=30)
        ax.text(val_x, 0.62, epicenter_name, transform=ax.transAxes, color='#ffffff',
                fontsize=18, fontweight='bold', ha='right', va='top', zorder=30)

        # 深さ
        ax.text(panel_x, 0.56, "深さ", transform=ax.transAxes, color='#ffffff', fontsize=label_fs, va='top', zorder=30)
        ax.text(panel_x, 0.535, "Depth", transform=ax.transAxes, color='#ffffff', fontsize=sub_label_fs, va='top', zorder=30)
        ax.text(val_x, 0.52, depth, transform=ax.transAxes, color='#ffffff',
                fontsize=18, fontweight='bold', ha='right', va='top', zorder=30)

        # 津波
        tsunami_display = tsunami_text
        if any(x in tsunami_text for x in ["津波の心配はありません", "詳細はありません", "失敗しました"]):
            tsunami_display = "心配なし"
        elif "津波注意報" in tsunami_text:
            tsunami_display = "津波注意報"

        ax.text(panel_x, 0.46, "津波", transform=ax.transAxes, color='#ffffff', fontsize=label_fs, va='top', zorder=30)
        ax.text(panel_x, 0.435, "Tsunami", transform=ax.transAxes, color='#ffffff', fontsize=sub_label_fs, va='top', zorder=30)
        ax.text(val_x, 0.42, tsunami_display, transform=ax.transAxes, color='#ffffff',
                fontsize=18, fontweight='bold', ha='right', va='top', zorder=30)

        # 凡例表示エリアの背景
        legend_bg_y = 0.02
        legend_h = 0.33
        legend_w = sidebar_ratio * 0.60
        legend_lx = 1.0 - legend_w - 0.01
        legend_bg = mpatches.Rectangle((legend_lx, legend_bg_y), legend_w, legend_h,
                                    transform=ax.transAxes, color='#1c1c1e', alpha=0.9, zorder=26)
        ax.add_patch(legend_bg)

        # 凡例項目
        lx_icon = legend_lx + 0.02
        lx_text = 0.985
        ly_step = 0.031
        curr_y = legend_bg_y + legend_h - 0.035
        box_w, box_h = 0.022, 0.025

        # 1. 震央
        ax.scatter(lx_icon + box_w/2, curr_y + box_h/2, transform=ax.transAxes, marker='x', color='#ffffff', s=80, linewidths=2.5, zorder=31)
        ax.scatter(lx_icon + box_w/2, curr_y + box_h/2, transform=ax.transAxes, marker='x', color='#ff3b30', s=70, linewidths=1.5, zorder=32)
        ax.text(lx_text, curr_y + box_h/2, "震央　　", transform=ax.transAxes, color='#ffffff', fontsize=11, ha='right', va='center', zorder=30)
        curr_y -= ly_step

        # 2. 震度リスト
        legend_levels = [
            ("7", "震度７　"), ("6+", "震度６強"), ("6-", "震度６弱"),
            ("5+", "震度５強"), ("5-", "震度５弱"), ("4", "震度４　"),
            ("3", "震度３　"), ("2", "震度２　"), ("1", "震度１　")
        ]

        box_w, box_h = 0.022, 0.025
        for code, label_text in legend_levels:
            # アイコン
            rect = mpatches.Rectangle((lx_icon, curr_y), box_w, box_h, transform=ax.transAxes,
                                    color=INTENSITY_COLORS.get(code, "#ffffff"),
                                    ec='#000000', lw=0.5, zorder=31)
            ax.add_patch(rect)
            ax.text(lx_icon + box_w/2, curr_y + box_h/2, INTENSITY_LABELS.get(code, code), transform=ax.transAxes,
                    color='#000000' if code in ["1","2","4","5-","5+"] else '#ffffff',
                    fontsize=8, fontweight='bold', ha='center', va='center', zorder=32)
            # テキスト (右寄せで強・弱を揃える)
            ax.text(lx_text, curr_y + box_h/2, label_text, transform=ax.transAxes,
                    color='#ffffff', fontsize=11, ha='right', va='center', zorder=30)
            curr_y -= ly_step

        # クレジット
        ax.text(0.012, 0.015, "気象庁防災情報XMLフォーマットを加工して作成 | 『気象庁防災情報発表区域データセット』（NII作成） 「GISデータ」（気象庁）を加工",
                transform=ax.transAxes, color='#8e8e93', fontsize=6, ha='left', va='bottom', zorder=30)

        ax.set_axis_off()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=144, facecolor=fig.get_facecolor(), bbox_inches='tight')
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
