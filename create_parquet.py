import geopandas as gpd
import pandas as pd
import os
import zipfile
import glob
import shutil

# --- 設定: 入力ファイル名 ---
CITY_ZIP_PATH = "./gis/20241128_AreaInformationCity_quake_GIS.zip"  # 市区町村ZIP
NATION_ZIP_PATH = "./gis/20190125_AreaForecastEEW_GIS.zip"          # 全国・地域ZIP
LAKE_SHP_PATH = ".gis/W09-05_GML/W09-05-g_Lake.shp"                # 湖沼Shapefile

# --- 設定: 出力ファイル名 ---
OUTPUT_UNIFIED_PATH = "optimized_unified_data.parquet"        # 統合データ（メイン）
OUTPUT_NATION_PATH = "optimized_unified_data_nation.parquet"  # 全国データ（インセット用）
OUTPUT_LAKE_PATH = "optimized_lake.parquet"                   # 湖沼単体（予備）

def extract_and_read_shp_from_zip(zip_path, temp_dir_name):
    """ZIPを解凍してShapefileを読み込む"""
    if os.path.exists(temp_dir_name):
        shutil.rmtree(temp_dir_name)
    os.makedirs(temp_dir_name)

    print(f"Loading ZIP: {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir_name)

    shp_files = glob.glob(os.path.join(temp_dir_name, "**", "*.shp"), recursive=True)
    if not shp_files:
        raise FileNotFoundError("Shapefile not found in ZIP.")

    target_shp = shp_files[0]
    # UTF-8で読み込み
    gdf = gpd.read_file(target_shp, encoding='utf-8')

    # 後始末
    shutil.rmtree(temp_dir_name)
    return gdf

def process_city_data():
    """市区町村データの読み込み・加工"""
    print("--- Processing City Data ---")
    gdf = extract_and_read_shp_from_zip(CITY_ZIP_PATH, "temp_city_extract")

    # カラム名の変更
    rename_map = {'regioncode': 'code', 'name': 'name', 'namekana': 'namekana'}
    gdf = gdf.rename(columns=rename_map)

    # 座標変換 (JGD2011 -> WGS84)
    if gdf.crs is None:
        gdf.set_crs("EPSG:6668", allow_override=True, inplace=True)
    gdf = gdf.to_crs("EPSG:4326")

    # 必要なカラムに絞る
    cols = ['code', 'name', 'namekana', 'geometry']
    # 存在しないカラムは除外
    cols = [c for c in cols if c in gdf.columns]
    gdf = gdf[cols]

    gdf['data_type'] = 'city'
    print(f"City data loaded: {len(gdf)} records.")
    return gdf

def process_nation_data():
    """全国データの読み込み・加工・Dissolve"""
    print("--- Processing Nation Data ---")
    gdf = extract_and_read_shp_from_zip(NATION_ZIP_PATH, "temp_nation_extract")

    # 座標変換
    if gdf.crs is None:
        gdf.set_crs("EPSG:6668", allow_override=True, inplace=True)
    gdf = gdf.to_crs("EPSG:4326")

    # Dissolve (全国を1つの形状に統合、または重なりを排除)
    print("Dissolving nation geometries...")
    # ジオメトリのみを抽出して統合
    unified_geom = gdf.geometry.unary_union
    # GeoDataFrameに再構築
    gdf_nation = gpd.GeoDataFrame(geometry=[unified_geom], crs=gdf.crs)

    print("Nation data processed.")
    return gdf_nation

def process_lake_data():
    """湖沼データの読み込み・加工"""
    print("--- Processing Lake Data ---")
    if not os.path.exists(LAKE_SHP_PATH):
        print("Warning: Lake shapefile not found. Skipping.")
        return None

    gdf = gpd.read_file(LAKE_SHP_PATH)

    # 座標変換 (JGD2000 -> WGS84)
    if gdf.crs is None:
        gdf.set_crs("EPSG:4612", allow_override=True, inplace=True)
    if gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    # 湖沼は描画用にジオメトリがあれば良いので属性は最低限に
    gdf = gdf[['geometry']].copy()
    gdf['data_type'] = 'lake'

    print(f"Lake data loaded: {len(gdf)} records.")
    return gdf

def optimize_geometry(gdf, output_path):
    """
    ジオメトリの単純化、代表点計算、Parquet保存
    optimize_data_final.py のロジックを適用
    """
    print(f"Optimizing and saving to {output_path}...")

    # 1. ポリゴンの単純化 (0.001度は約110m精度の簡略化)
    # これにより描画時の頂点数が激減し、Matplotlibが高速化します
    print(" - Simplifying geometries...")
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.001, preserve_topology=True)

    # 2. 代表点（重心）の事前計算
    # 描画時に計算させず、あらかじめ列として持っておきます
    print(" - Calculating representative points...")
    gdf['rep_x'] = gdf.geometry.representative_point().x
    gdf['rep_y'] = gdf.geometry.representative_point().y

    # 3. 保存
    print(" - Saving Parquet...")
    gdf.to_parquet(output_path, index=False, compression='snappy')
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"Done. Size: {file_size:.2f} MB")

def main():
    print("=== Start Unified Dataset Creation ===")

    # 1. 各データのロード
    gdf_city = process_city_data()
    gdf_lake = process_lake_data()
    gdf_nation = process_nation_data()

    # 2. 統合データ (City + Lake) の作成
    print("\n--- Merging City and Lake Data ---")
    data_frames = [gdf_city]
    if gdf_lake is not None:
        data_frames.append(gdf_lake)

    # カラムを揃えて結合 (pd.concat)
    gdf_unified = pd.concat(data_frames, ignore_index=True)

    # 3. 最適化と保存 (Main: optimized_unified_data.parquet)
    optimize_geometry(gdf_unified, OUTPUT_UNIFIED_PATH)

    # 4. 全国データの最適化と保存 (Inset: optimized_unified_data_nation.parquet)
    print("\n--- Optimizing Nation Data ---")
    optimize_geometry(gdf_nation, OUTPUT_NATION_PATH)

    # 5. (予備) 湖沼データの単体保存
    if gdf_lake is not None:
        print("\n--- Optimizing Lake Data (Standalone) ---")
        optimize_geometry(gdf_lake, OUTPUT_LAKE_PATH)

    print("\n=== All Processes Completed Successfully ===")
    print(f"1. {OUTPUT_UNIFIED_PATH} (Main Map)")
    print(f"2. {OUTPUT_NATION_PATH} (Inset Map)")

if __name__ == "__main__":
    main()
