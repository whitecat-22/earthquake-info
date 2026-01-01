import geopandas as gpd
import pandas as pd
import os

def convert_lake_to_parquet(input_shp_path, output_parquet_path):
    """
    国土数値情報の湖沼データ（シェープファイル）をparquet形式に変換

    Args:
        input_shp_path: 入力シェープファイルのパス
        output_parquet_path: 出力parquetファイルのパス
    """
    print(f"Loading shapefile from {input_shp_path}...")
    gdf = gpd.read_file(input_shp_path)

    # カラム情報を表示
    print(f"Original columns: {gdf.columns.tolist()}")
    print(f"Shape: {gdf.shape}")
    print(f"CRS: {gdf.crs}")

    # 座標系の確認と変換
    if gdf.crs is None:
        # JGD2000 / (B, L) とみなす（国土数値情報の標準座標系）
        print("CRS not set. Assuming JGD2000 / (B, L) and converting to WGS84...")
        gdf.set_crs("EPSG:4612", allow_override=True, inplace=True)  # JGD2000地理座標系
    elif gdf.crs.to_string() != "EPSG:4326":
        print(f"Converting CRS from {gdf.crs} to EPSG:4326 (WGS84)...")

    # WGS84に変換
    gdf = gdf.to_crs("EPSG:4326")
    print(f"Converted to CRS: {gdf.crs}")

    # 必要な属性を保持（湖沼名など）
    # 国土数値情報の仕様では W09_001 が湖沼名、W09_002 が行政区域コードなど
    # 実際のカラム名を確認して保持
    essential_cols = ['geometry']

    # 湖沼名に関連するカラムを探す
    possible_name_cols = ['W09_001', 'W09-001', '湖沼名', 'name', 'NAME', 'lake_name', 'LAKE_NAME']
    possible_code_cols = ['W09_002', 'W09-002', '行政区域コード', 'code', 'CODE', 'admin_code']

    for col in possible_name_cols:
        if col in gdf.columns:
            essential_cols.append(col)
            print(f"Found lake name column: {col}")
            break

    for col in possible_code_cols:
        if col in gdf.columns:
            essential_cols.append(col)
            print(f"Found admin code column: {col}")
            break

    # その他の属性も保持（最大水深、水面標高など）
    for col in ['W09_003', 'W09-003', '最大水深', 'W09_004', 'W09-004', '水面標高', 'iD', 'ID']:
        if col in gdf.columns:
            essential_cols.append(col)
            print(f"Found additional column: {col}")

    # 必要なカラムのみを保持
    available_cols = [col for col in essential_cols if col in gdf.columns]
    if len(available_cols) == 1:  # geometryのみの場合
        print("Warning: Only geometry column found. Keeping all columns.")
        gdf_export = gdf.copy()
    else:
        gdf_export = gdf[available_cols].copy()
        print(f"Columns kept: {available_cols}")

    # サンプルデータを表示
    print("\n--- Sample data (first 3 rows) ---")
    print(gdf_export.head(3))
    print("-----------------------------------\n")

    # Parquet形式で保存
    print(f"Saving to {output_parquet_path}...")
    gdf_export.to_parquet(output_parquet_path)
    print(f"Done! File saved: {output_parquet_path}")
    print(f"File size: {os.path.getsize(output_parquet_path) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    input_path = "W09-05_GML/W09-05-g_Lake.shp"
    output_path = "lake.parquet"

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        print("Please ensure the shapefile is in the correct location.")
        exit(1)

    convert_lake_to_parquet(input_path, output_path)

