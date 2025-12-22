import geopandas as gpd
import zipfile
import os
import glob
import shutil

def finish_conversion(input_zip_path, output_geojson_path):
    extract_dir = "jma_final_convert"

    # 1. 掃除と解凍
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir)

    print(f"1. ZIPを解凍中...: {input_zip_path}")
    with zipfile.ZipFile(input_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # 2. .shp ファイル特定
    shp_files = glob.glob(os.path.join(extract_dir, "**", "*.shp"), recursive=True)
    target_shp = shp_files[0]

    # 3. UTF-8 で読み込み
    print("2. データを読み込み中 (UTF-8)...")
    try:
        gdf = gpd.read_file(target_shp, encoding='utf-8')

        # 中身の確認（日本語カラムを表示）
        print("\n--- データ内容確認 (先頭5行) ---")
        print(gdf[['name', 'namekana']].head())
        print("------------------------------\n")

        # 4. カラムのマッピング
        # ログから判明したカラム名: regioncode, regionname, name, namekana
        # 出力したい形式: code, name, namekana

        rename_map = {
            'regioncode': 'code',
            'name': 'name',
            'namekana': 'namekana'
        }

        # リネーム実行
        gdf = gdf.rename(columns=rename_map)

        # 必要な列だけに絞る
        cols = ['code', 'name', 'namekana', 'geometry']
        gdf_export = gdf[cols]

        # 5. 座標系設定と変換
        if gdf_export.crs is None:
            # JGD2011(EPSG:6668)とみなす
            gdf_export.set_crs("EPSG:6668", allow_override=True, inplace=True)

        # WGS84(EPSG:4326)へ変換
        gdf_export = gdf_export.to_crs("EPSG:4326")

        # 6. 出力
        print(f"3. 書き出し中: {output_geojson_path} ...")
        gdf_export.to_file(output_geojson_path, driver='GeoJSON')
        print("★ 変換が完了しました！ area.geojson を確認してください。")

    except Exception as e:
        print(f"エラー: {e}")

    finally:
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)

if __name__ == "__main__":
    INPUT_FILE = "20241128_AreaInformationCity_quake_GIS.zip"
    OUTPUT_FILE = "city.geojson"
    finish_conversion(INPUT_FILE, OUTPUT_FILE)
