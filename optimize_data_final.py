import geopandas as gpd
import pandas as pd

def optimize_geographic_data(input_path, output_path):
    print(f"Optimizing {input_path}...")
    gdf = gpd.read_parquet(input_path)

    # 1. ポリゴンの単純化 (0.001度は約110m精度の簡略化)
    # これにより描画時の頂点数が激減し、Matplotlibが高速化します
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.001, preserve_topology=True)

    # 2. 代表点（重心）の事前計算
    # 描画時に計算させず、あらかじめ列として持っておきます
    gdf['rep_x'] = gdf.geometry.representative_point().x
    gdf['rep_y'] = gdf.geometry.representative_point().y

    # 3. 空間インデックスを確実にするためのソートと保存
    # (Parquetは標準で空間インデックスを保持しませんが、BBox情報を付与して保存されます)
    gdf.to_parquet(output_path, index=False)
    print(f"Optimized data saved to {output_path}")

if __name__ == "__main__":
    # 既存の unified_data.parquet を変換
    optimize_geographic_data("unified_data.parquet", "optimized_unified_data.parquet")
    # lake.parquet も同様に処理
    optimize_geographic_data("lake.parquet", "optimized_lake.parquet")
