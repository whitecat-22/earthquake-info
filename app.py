import os
import requests
from lxml import etree
import datetime
import re
import boto3
import json
import textwrap
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

# 正規表現のコンパイルとキャッシュ
_COORD_PATTERN_2 = re.compile(r'([+-][0-9.]+)([+-][0-9.]+)')
_COORD_PATTERN_3 = re.compile(r'([+-][0-9.]+)([+-][0-9.]+)([+-][0-9.]+)?')

# Lambda環境でのMatplotlib書き込みエラー対策
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm

# フォント設定
_cached_font = None
def setup_fonts():
    global _cached_font
    if _cached_font: return _cached_font
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

# --- 地図データ設定 ---
UNIFIED_PARQUET_PATH = 'optimized_unified_data.parquet'
UNIFIED_NATION_PARQUET_PATH = 'optimized_unified_data_nation.parquet'
LAKE_PARQUET_PATH = 'optimized_lake.parquet'
PARQUET_PATH = 'city.parquet' # フォールバック用

APP_MODE = os.getenv("APP_MODE", "production")

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

INTENSITY_LABELS = {
    "1": "1", "2": "2", "3": "3", "4": "4",
    "5-": "5-", "5+": "5+", "6-": "6-", "6+": "6+", "7": "7"
}

INTENSITY_DISPLAY_NAMES = {
    "1": "1", "2": "2", "3": "3", "4": "4",
    "5-": "5弱", "5+": "5強", "6-": "6弱", "6+": "6強", "7": "7"
}

JMA_NS = {
    'atom': 'http://www.w3.org/2005/Atom',
    'jmx': 'http://xml.kishou.go.jp/jmaxml1/',
    'jmx_ib': 'http://xml.kishou.go.jp/jmaxml1/body/seismology1/',
    'jmx_eb': 'http://xml.kishou.go.jp/jmaxml1/elementBasis1/',
    'jmx_info': 'http://xml.kishou.go.jp/jmaxml1/informationBasis1/',
    'report': 'http://xml.kishou.go.jp/jmaxml1/'
}

s3_client = boto3.client('s3')

# --- グローバル領域でのデータロード (コールドスタート対策) ---
GDF_JAPAN = None
GDF_LAKES = None
GDF_NATION = None

def load_global_map_data():
    global GDF_JAPAN, GDF_LAKES, GDF_NATION
    print("Initializing Map Data...")

    # 統合データのロード
    if os.path.exists(UNIFIED_PARQUET_PATH):
        print(f"Loading unified data from {UNIFIED_PARQUET_PATH}...")
        try:
            gdf_unified = gpd.read_parquet(UNIFIED_PARQUET_PATH)

            # Cityデータ
            GDF_JAPAN = gdf_unified[gdf_unified['data_type'] == 'city'].copy()
            print(f"City data loaded: {len(GDF_JAPAN)} polygons")

            # Lakeデータ
            lake_data = gdf_unified[gdf_unified['data_type'] == 'lake']
            if not lake_data.empty:
                GDF_LAKES = lake_data.copy()
                print(f"Lake data loaded: {len(GDF_LAKES)} lakes")
        except Exception as e:
            print(f"Error loading unified parquet: {e}")

    # フォールバック (統合データがない場合)
    if GDF_JAPAN is None:
        print("Unified data not found. Loading individual files...")
        if os.path.exists(PARQUET_PATH):
            GDF_JAPAN = gpd.read_parquet(PARQUET_PATH)
        elif GEOJSON_PATH and os.path.exists(GEOJSON_PATH):
            GDF_JAPAN = gpd.read_file(GEOJSON_PATH)

    if GDF_LAKES is None and os.path.exists(LAKE_PARQUET_PATH):
        GDF_LAKES = gpd.read_parquet(LAKE_PARQUET_PATH)

    # 全国データのロード (インセット用)
    if os.path.exists(UNIFIED_NATION_PARQUET_PATH):
        GDF_NATION = gpd.read_parquet(UNIFIED_NATION_PARQUET_PATH)
    elif GDF_JAPAN is not None:
        GDF_NATION = GDF_JAPAN # フォールバック

    print("Map Data Initialization Completed.")

# モジュール読み込み時に実行（Lambdaコールドスタート時に1回だけ走る）
load_global_map_data()


class MapRenderer:
    """
    地図描画に特化したクラス
    グローバルにロードされた地図データを参照して描画を行う
    """
    def __init__(self):
        self.gdf_japan = GDF_JAPAN
        self.gdf_lakes = GDF_LAKES
        self.gdf_nation = GDF_NATION

    def render(self, epicenter_name, lat, lon, depth, regional_intensities, station_data, magnitude, time_str, announce_time, max_int, tsunami_text, is_eew=False):
        if self.gdf_japan is None:
            print("Error: Map data is not loaded.")
            return None

        gdf = self.gdf_japan

        # 描画範囲の決定
        active_points = []
        if lon and lat:
            active_points.append((lon, lat))

        target_keys = set(str(k) for k in regional_intensities.keys())
        # 高速化: ベクトル演算でフィルタリング
        mask = pd.Series(False, index=gdf.index)
        match_cols = ['code', 'CODE', 'N03_007', 'name', 'nam', 'name_ja', 'COMMNAME', 'CITYNAME', 'N03_004']
        for col in match_cols:
            if col in gdf.columns:
                mask |= gdf[col].astype(str).isin(target_keys)

        active_gdf = gdf[mask]
        if not active_gdf.empty:
            active_points.extend(list(zip(active_gdf['rep_x'], active_gdf['rep_y'])))

        fig = None
        try:
            if active_points:
                lons_all, lats_all = zip(*active_points)
                min_lon, max_lon = min(lons_all), max(lons_all)
                min_lat, max_lat = min(lats_all), max(lats_all)

                d_lat = max_lat - min_lat
                d_lon = max_lon - min_lon

                import numpy as np
                cos_lat = np.cos(np.radians((max_lat + min_lat) / 2))

                # マージン設定
                span_lat = max(d_lat * 1.3, 2.5)
                span_lon = max(d_lon * 1.3, 2.5 / cos_lat)

                center_lat = (max_lat + min_lat) / 2
                center_lon = (max_lon + min_lon) / 2

                lim_w, lim_e = center_lon - span_lon/2, center_lon + span_lon/2
                lim_s, lim_n = center_lat - span_lat/2, center_lat + span_lat/2

                map_aspect = (span_lon * cos_lat) / span_lat

                fig_h = 10.8
                map_w = fig_h * map_aspect
                sidebar_w = 4.8
                total_w = map_w + sidebar_w

                fig, ax = plt.subplots(figsize=(total_w, fig_h))
                sidebar_ratio = sidebar_w / total_w

                relevant_gdf = gdf.cx[lim_w:lim_e, lim_s:lim_n].copy()
            else:
                # デフォルト表示
                lim_w, lim_e = 128, 146
                lim_s, lim_n = 30, 46
                sidebar_ratio = 0.25
                fig, ax = plt.subplots(figsize=(19.2, 10.8))
                relevant_gdf = gdf.copy()

            # 震度データのマッピング
            lookup = {str(k): v for k, v in regional_intensities.items()}
            code_cols = [c for c in ['code', 'CODE', 'N03_007'] if c in relevant_gdf.columns]
            name_cols = [c for c in ['name', 'nam', 'name_ja', 'COMMNAME', 'CITYNAME', 'N03_004'] if c in relevant_gdf.columns]
            search_cols = code_cols + name_cols

            relevant_gdf['intensity'] = None
            for col in search_cols:
                relevant_gdf['intensity'] = relevant_gdf['intensity'].fillna(
                    relevant_gdf[col].astype(str).map(lookup)
                )

            relevant_gdf['color'] = relevant_gdf['intensity'].map(lambda x: INTENSITY_COLORS.get(x, "#7c7c7c"))

            # 背景と陸地
            bg_color = '#001f41'
            ax.set_facecolor(bg_color)
            fig.patch.set_facecolor(bg_color)
            relevant_gdf.plot(ax=ax, color=relevant_gdf['color'], edgecolor='#2c2c2e', linewidth=0.2, alpha=0.6)

            # 湖沼の描画
            if self.gdf_lakes is not None:
                relevant_lakes = self.gdf_lakes.cx[lim_w:lim_e, lim_s:lim_n].copy()
                if not relevant_lakes.empty:
                    lake_color = '#003D6B'
                    relevant_lakes.plot(ax=ax, color=lake_color, edgecolor='#004A7F', linewidth=0.3, alpha=0.9, zorder=5)

            # 震度アイコン (市区町村)
            active_regions = relevant_gdf[relevant_gdf['intensity'].notna()]
            for code in INTENSITY_DISPLAY_NAMES.keys():
                subset = active_regions[active_regions['intensity'] == code]
                if subset.empty: continue

                color = INTENSITY_COLORS.get(code, "#ffffff")
                label = INTENSITY_LABELS.get(code, code)
                text_color = '#000000' if code in ["1","2","4","5-","5+"] else '#ffffff'

                ax.scatter(subset['rep_x'], subset['rep_y'], marker='s', s=144, color=color,
                        edgecolors='#000000', linewidths=0.5, zorder=8)

                for _, row in subset.iterrows():
                    ax.text(row['rep_x'], row['rep_y'], label, color=text_color,
                            fontsize=8, ha='center', va='center', fontweight='bold', zorder=9)

            # 震度アイコン (観測点)
            for code in INTENSITY_DISPLAY_NAMES.keys():
                pts = [st for st in station_data if st['intensity'] == code]
                if not pts: continue

                color = INTENSITY_COLORS.get(code, "#ffffff")
                label = INTENSITY_LABELS.get(code, code)
                text_color = '#000000' if code in ["1","2","4","5-","5+"] else '#ffffff'

                lons = [p['lon'] for p in pts]
                lats = [p['lat'] for p in pts]
                ax.scatter(lons, lats, marker='o', s=100, color=color, edgecolors='#ffffff', linewidths=0.8, zorder=10)
                for p in pts:
                    ax.text(p['lon'], p['lat'], label, color=text_color, fontsize=7,
                            ha='center', va='center', fontweight='bold', zorder=11)

            # 震源地
            if lat and lon:
                ax.scatter(lon, lat, marker='x', color='#ffffff', s=80, linewidths=2.5, zorder=20)
                ax.scatter(lon, lat, marker='x', color='#ff3b30', s=70, linewidths=1.5, zorder=21)

            # レイアウト調整
            ax.set_position([0, 0, 1 - sidebar_ratio, 1.0])
            ax.set_xlim(lim_w, lim_e)
            ax.set_ylim(lim_s, lim_n)
            ax.set_axis_off()

            # サイドバー
            panel_rect = mpatches.Rectangle((1 - sidebar_ratio, 0), sidebar_ratio, 1.0,
                                        transform=fig.transFigure, color='#000000', alpha=0.9, zorder=10)
            fig.patches.append(panel_rect)

            # テキスト情報
            fig.text(0.02, 0.95, "各地の震度情報", color='#ffffff', fontsize=24, fontweight='bold', va='top', zorder=100)
            fig.text(0.02, 0.90, "Seismic Intensity Report", color='#ffffff', fontsize=12, va='top', zorder=100)
            fig.text(0.02, 0.86, announce_time, color='#ffffff', fontsize=14, va='top', zorder=100)

            panel_x = 1 - sidebar_ratio + 0.02
            val_x = 0.98
            label_fs = 18
            sub_label_fs = 12
            value_fs = 34

            try:
                dt_obj = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                d_str = dt_obj.strftime('%m月%d日')
                t_str = dt_obj.strftime('%H:%M頃')
            except:
                d_str = time_str
                t_str = ""

            tsunami_display = tsunami_text
            if any(x in tsunami_text for x in ["被害の心配はありません", "津波の心配はありません"]):
                tsunami_display = "被害の心配なし"
            elif any(x in tsunami_text for x in ["詳細はありません", "失敗しました"]):
                tsunami_display = "調査中"
            elif "津波注意報" in tsunami_text:
                tsunami_display = "津波注意報"
            wrapped_tsunami = textwrap.fill(tsunami_display, width=16)

            ui_items = [
                ("最大震度", "Max Intensity", INTENSITY_DISPLAY_NAMES.get(max_int, max_int), 0.95, value_fs, True),
                ("規模", "Magnitude", f"{magnitude}", 0.86, value_fs, True),
                ("発生時刻", "Date", f"{d_str}\n{t_str}", 0.77, 18, True),
                ("震源地", "Epicenter", epicenter_name, 0.67, 17, False),
                ("深さ", "Depth", depth, 0.57, 18, False),
                ("津波", "Tsunami", wrapped_tsunami, 0.47, 15, False)
            ]

            for label_jp, label_en, val_text, y_pos, val_fs_item, is_bold in ui_items:
                fig.text(panel_x, y_pos, label_jp, color='#ffffff', fontsize=label_fs, va='top', zorder=1000)
                fig.text(panel_x, y_pos - 0.025, label_en, color='#ffffff', fontsize=sub_label_fs, va='top', zorder=1000)
                fig.text(val_x, y_pos, val_text, color='#ffffff', fontsize=val_fs_item,
                        fontweight='bold' if is_bold else 'normal', ha='right', va='top', zorder=1000)

            fig.text(0.012, 0.015, "気象庁防災情報XMLフォーマットを加工して作成 | 『気象庁防災情報発表区域データセット』（NII作成） 「GISデータ」（気象庁）、国土数値情報（湖沼）を加工",
                    color='#8e8e93', fontsize=6, ha='left', va='bottom', zorder=100)

            if is_eew:
                eew_msg = "この地震について、緊急地震速報を発表しています。"
                ax.add_patch(mpatches.Rectangle((0.01, 0.05), 0.48, 0.06, transform=ax.transAxes,
                                            color='#000000', alpha=0.7, zorder=35))
                ax.text(0.02, 0.08, eew_msg, transform=ax.transAxes, color='#ffff00',
                        fontsize=18, fontweight='bold', va='center', zorder=40)

            # インセットマップ
            if self.gdf_nation is not None:
                inset_w = sidebar_ratio * 0.85
                inset_h = 0.20
                inset_lx = 1.0 - sidebar_ratio + (sidebar_ratio - inset_w) / 2
                inset_pos = [inset_lx, 0.04, inset_w, inset_h]
                inset_ax = fig.add_axes(inset_pos, zorder=50)

                self.gdf_nation.plot(ax=inset_ax, color='#ffffff', edgecolor='#2c2c2e', linewidth=0.1, alpha=0.8)

                overview_rect = mpatches.Rectangle((lim_w, lim_s), lim_e - lim_w, lim_n - lim_s,
                                                edgecolor='#ff3b30', facecolor='none', linewidth=1.5, zorder=40)
                inset_ax.add_patch(overview_rect)
                inset_ax.set_xlim(122, 149)
                inset_ax.set_ylim(24, 46)
                inset_ax.set_axis_off()
                inset_ax.set_facecolor('none')

            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=144, facecolor=fig.get_facecolor())
            buf.seek(0)
            return buf

        finally:
            if fig:
                fig.clf()
                plt.close(fig)
            plt.close('all') # 念のため全てのplotを閉じる
            gc.collect()     # ガベージコレクションを促進


class EarthquakeMonitor:
    def __init__(self):
        state = self._load_state()
        self.last_event_id = state.get('id')
        self.last_event_time = state.get('event_time') or state.get('updated')
        self.slack_client = WebClient(token=SLACK_BOT_TOKEN)
        self.session = requests.Session()

        # 描画クラスを初期化
        self.renderer = MapRenderer()

        print(f"Loaded state from S3: ID={self.last_event_id}, Time={self.last_event_time}")

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

            root = etree.fromstring(response.content)
            entries = []

            for entry_elem in root.xpath('//atom:entry', namespaces=JMA_NS):
                def get_text(elem, path):
                    found = elem.xpath(path, namespaces=JMA_NS)
                    return found[0].text if found else ''

                def get_attr(elem, path, attr):
                    found = elem.xpath(path, namespaces=JMA_NS)
                    return found[0].get(attr) if found else ''

                entry_data = {
                    'title': get_text(entry_elem, 'atom:title'),
                    'id': get_text(entry_elem, 'atom:id'),
                    'updated': get_text(entry_elem, 'atom:updated'),
                    'link': {
                        '@href': get_attr(entry_elem, 'atom:link[@rel="related"]', 'href') or \
                                get_attr(entry_elem, 'atom:link', 'href')
                    }
                }
                entries.append(entry_data)

            print(f"Fetched {len(entries)} entries from JMA.")
            return entries
        except Exception as e:
            print(f"Error fetching feed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def handle_detail(self, url, event_id):
        try:
            response = self.session.get(url, timeout=10)
            root = etree.fromstring(response.content)

            def x_text(elem, path, default=''):
                res = elem.xpath(path, namespaces=JMA_NS)
                return res[0].text if res and res[0].text else default

            head_list = root.xpath('.//jmx_info:Head', namespaces=JMA_NS)
            if not head_list:
                print("Error: No Head element found (namespace mismatch?).")
                return {"success": False, "event_id": event_id, "error": "No Head element"}
            head = head_list[0]

            headline_text = x_text(head, './/jmx_info:Headline/jmx_info:Text')
            target_date_time = x_text(head, './/jmx_info:TargetDateTime')

            body_list = root.xpath('.//jmx_ib:Body', namespaces=JMA_NS)
            if not body_list:
                body_list = root.xpath('.//jmx:Body', namespaces=JMA_NS)

            if not body_list:
                print("Error: No Body element found.")
                return {"success": False, "event_id": event_id, "error": "No Body element"}
            body = body_list[0]

            earthquake = body.xpath('.//jmx_ib:Earthquake', namespaces=JMA_NS)
            if earthquake:
                earthquake = earthquake[0]
                origin_time = x_text(earthquake, './/jmx_ib:OriginTime') or target_date_time
                magnitude = x_text(earthquake, './/jmx_eb:Magnitude', '不明')

                hypo_area = earthquake.xpath('.//jmx_ib:Hypocenter/jmx_ib:Area', namespaces=JMA_NS)
                if hypo_area:
                    hypo_area = hypo_area[0]
                    epicenter_name = x_text(hypo_area, './/jmx_ib:Name', '不明')
                    coord_str = x_text(hypo_area, './/jmx_eb:Coordinate')
                else:
                    epicenter_name = '不明'
                    coord_str = ''
            else:
                origin_time = target_date_time
                magnitude = '不明'
                epicenter_name = '不明'
                coord_str = ''

            # 津波情報取得 (jmx_ib優先)
            tsunami_text = x_text(body, './/jmx_ib:Comments/jmx_ib:ForecastComment/jmx_ib:Text')
            if not tsunami_text:
                tsunami_text = x_text(body, './/jmx:Comments/jmx:ForecastComment/jmx:Text')
            if not tsunami_text:
                tsunami_text = "津波情報の詳細はありません。"

            regional_intensities = {}
            station_data = []

            intensity_elem = body.xpath('.//jmx_ib:Intensity', namespaces=JMA_NS)
            if intensity_elem:
                intensity_elem = intensity_elem[0]

                for pref in intensity_elem.xpath('.//jmx_ib:Observation/jmx_ib:Pref', namespaces=JMA_NS):
                    for area in pref.xpath('jmx_ib:Area', namespaces=JMA_NS):
                        area_name = x_text(area, 'jmx_ib:Name')
                        area_code = x_text(area, 'jmx_ib:Code')
                        area_max = x_text(area, 'jmx_ib:MaxInt')

                        if area_max:
                            if area_name: regional_intensities[area_name] = area_max
                            if area_code: regional_intensities[str(area_code)] = area_max

                        for city in area.xpath('jmx_ib:City', namespaces=JMA_NS):
                            city_name = x_text(city, 'jmx_ib:Name')
                            city_code = x_text(city, 'jmx_ib:Code')
                            city_max = x_text(city, 'jmx_ib:MaxInt')

                            if city_max:
                                if city_name: regional_intensities[city_name] = city_max
                                if city_code: regional_intensities[str(city_code)] = city_max

                            for st in city.xpath('jmx_ib:IntensityStation', namespaces=JMA_NS):
                                st_name = x_text(st, 'jmx_ib:Name')
                                st_int = x_text(st, 'jmx_ib:Int')
                                st_coord = x_text(st, 'jmx_eb:Coordinate')

                                if st_coord:
                                    s_match = _COORD_PATTERN_2.search(st_coord)
                                    if s_match:
                                        station_data.append({
                                            'name': st_name,
                                            'intensity': st_int,
                                            'lat': float(s_match.group(1)),
                                            'lon': float(s_match.group(2))
                                        })

            lat, lon, depth = None, None, "不明"
            if coord_str:
                match = _COORD_PATTERN_3.search(coord_str)
                if match:
                    lat = float(match.group(1))
                    lon = float(match.group(2))
                    if match.group(3):
                        depth_m = float(match.group(3))
                        depth_km = abs(int(depth_m / 1000))
                        if depth_km == 0:
                            depth = "ごく浅い"
                        else:
                            depth = f"約{depth_km}km"

            def format_time(ts):
                try:
                    dt = datetime.datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    return dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    return ts

            map_time_str = origin_time[:19].replace('T', ' ')
            formatted_time = format_time(origin_time)

            def format_atime(ts):
                try:
                    dt = datetime.datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    return dt.strftime('%Y年%m月%d日 %H:%M 発表')
                except:
                    return ts
            announcement_time = format_atime(target_date_time)

            max_intensity = "0"
            if regional_intensities:
                order = {"0":0, "1":1, "2":2, "3":3, "4":4, "5-":5, "5+":6, "6-":7, "6+":8, "7":9}
                current_max_val = -1
                for v in regional_intensities.values():
                    if v in order and order[v] > current_max_val:
                        current_max_val = order[v]
                        max_intensity = v

            is_eew = "緊急地震速報を発表しています" in headline_text or "緊急地震速報を発表しています" in tsunami_text

            # 地図生成はRendererに委譲
            image_buf = self.renderer.render(
                epicenter_name, lat, lon, depth, regional_intensities, station_data, magnitude,
                map_time_str, announcement_time, max_intensity, tsunami_text, is_eew
            )

            lat_label = f"北緯{lat}度" if lat is not None and lat >= 0 else f"南緯{abs(lat)}度" if lat is not None else "不明"
            lon_label = f"東経{lon}度" if lon is not None and lon >= 0 else f"西経{abs(lon)}度" if lon is not None else "不明"
            coords_str = f"{lat_label} / {lon_label}"

            max_int_label = INTENSITY_DISPLAY_NAMES.get(max_intensity, max_intensity)

            return {
                "success": True,
                "event_id": event_id,
                "origin_time": origin_time,
                "payload": {
                    "headline": headline_text,
                    "epicenter": epicenter_name,
                    "coords": coords_str,
                    "max_int": max_int_label,
                    "magnitude": magnitude,
                    "depth": depth,
                    "time_str": formatted_time,
                    "tsunami_text": tsunami_text,
                    "image_buf": image_buf
                }
            }

        except Exception as e:
            print(f"Error processing detail: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "event_id": event_id, "error": str(e)}

    def send_to_slack(self, headline, epicenter, coords, max_int, magnitude, depth, time_str, tsunami_text, image_buf):
        if not SLACK_BOT_TOKEN or not SLACK_CHANNEL_ID:
            print("Slack setting missing.")
            return

        try:
            credit_link = "<https://www.jma.go.jp/jma/index.html|気象庁>防災情報XMLフォーマットを加工して作成 | 『気象庁防災情報発表区域データセット』（NII作成） 「GISデータ」（気象庁）、国土数値情報（湖沼）を加工"

            message = (
                f"<!here> *【地震速報】*\n{headline}\n\n"
                f"*発生時刻*: {time_str}\n"
                f"*震央地名*: {epicenter}\n"
                f"*緯度経度*: {coords}\n"
                f"*深さ*: {depth}\n"
                f"*規模*: M{magnitude}\n"
                f"*最大震度*: {max_int}\n"
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

    TARGET_TITLES = ["震源・震度に関する情報"]

    orig_last_id = monitor.last_event_id
    orig_last_time = monitor.last_event_time

    new_entries_to_process = []

    for entry in entries:
        title = entry.get('title', '')
        event_id = entry.get('id', '').strip()
        event_time = entry.get('updated', '').strip()

        if title not in TARGET_TITLES:
            continue

        is_already_processed = False
        if event_id == (orig_last_id or "").strip():
            is_already_processed = True
        elif orig_last_time and event_time <= orig_last_time:
            is_already_processed = True

        if is_already_processed:
            print(f"Skipping: {title} (Matches state. ID: {event_id})")
            continue

        new_entries_to_process.append(entry)

    if not new_entries_to_process:
        print("No new entries to process.")
        return {"statusCode": 200, "body": "No new entries"}

    # 発生時刻が古い順にソート（通知順序の保証）
    new_entries_to_process.sort(key=lambda x: x.get('updated', ''))

    print(f"Found {len(new_entries_to_process)} new entries. Starting parallel processing...")

    results_map = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_id = {}
        for entry in new_entries_to_process:
            event_id = entry.get('id', '').strip()
            detail_url = entry.get('link', {}).get('@href')
            print(f"!!! Processing start: {entry.get('title')} ({event_id}) !!!")

            future = executor.submit(monitor.handle_detail, detail_url, event_id)
            future_to_id[future] = event_id

        for future in as_completed(future_to_id):
            eid = future_to_id[future]
            try:
                result = future.result()
                results_map[eid] = result
            except Exception as exc:
                print(f"Event {eid} generated an exception: {exc}")
                results_map[eid] = {"success": False, "error": str(exc)}

    processed_count = 0
    last_processed_entry = None

    for entry in new_entries_to_process:
        event_id = entry.get('id', '').strip()
        result = results_map.get(event_id)

        if result and result.get("success"):
            print(f"Sending notification for: {event_id}")
            payload = result["payload"]

            monitor.send_to_slack(
                payload["headline"],
                payload["epicenter"],
                payload["coords"],
                payload["max_int"],
                payload["magnitude"],
                payload["depth"],
                payload["time_str"],
                payload["tsunami_text"],
                payload["image_buf"]
            )
            processed_count += 1
            last_processed_entry = entry
        else:
            print(f"Skipping notification for {event_id} due to processing error.")

    if last_processed_entry:
        new_last_id = last_processed_entry.get('id', '').strip()
        new_last_time = last_processed_entry.get('updated', '').strip()

        if (not orig_last_time) or (new_last_time > orig_last_time):
            monitor._save_state(new_last_id, new_last_time)

    return {"statusCode": 200, "body": f"Processed {processed_count} entries"}


if __name__ == "__main__":
    if APP_MODE == "test":
        print("--- Local Test Start (APP_MODE=test) ---")
        if not SLACK_BOT_TOKEN or not SLACK_CHANNEL_ID:
            print("Error: SLACK_BOT_TOKEN or SLACK_CHANNEL_ID is not set in .env")
            exit(1)

        os.environ['S3_BUCKET_NAME'] = ''
        monitor = EarthquakeMonitor()

        entries = monitor.fetch_feed()

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
            result = monitor.handle_detail(target_entry.get('link', {}).get('@href'), target_entry.get('id'))

            if result.get("success"):
                p = result["payload"]
                monitor.send_to_slack(p["headline"], p["epicenter"], p["coords"], p["max_int"], p["magnitude"], p["depth"], p["time_str"], p["tsunami_text"], p["image_buf"])
                print("--- Local Test Finished (Success) ---")
            else:
                print(f"--- Local Test Finished (Failed: {result.get('error')}) ---")
    else:
        print("--- Production Mode Start (One-shot) ---")
        lambda_handler({}, None)
