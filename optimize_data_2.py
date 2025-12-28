import geopandas as gpd
import pandas as pd
import os

# 極端に巨大なGeoJSONを読み込むための設定
os.environ['OGR_GEOJSON_MAX_OBJ_SIZE'] = '0'

def optimize_data(input_path, output_path):
    print(f"Loading {input_path}...")
    gdf = gpd.read_file(input_path)

    # 5. 不要な属性の削除 (code, name, geometry 以外を削除)
    # city.geojsonの実際のカラム名を確認しながら調整が必要ですが、一般的な名前を想定
    essential_cols = []
    for col in ['code', 'name', 'nam', 'name_ja', 'N03_007', 'N03_004', 'COMMNAME', 'CITYNAME', 'geometry']:
        if col in gdf.columns:
            essential_cols.append(col)

    gdf = gdf[essential_cols]
    print(f"Columns kept: {essential_cols}")

    # 重複する境界を結合する (Dissolve)
    print("Dissolving geometries (merging overlapping boundaries)...")
    # 全ての行を一つのジオメトリに統合 (内部境界や重なりを削除)
    unified_geom = gdf.geometry.unary_union
    # 新しいGeoDataFrameを作成
    gdf = gpd.GeoDataFrame(geometry=[unified_geom], crs=gdf.crs)

    # 3. 代表点 (Representative Point) の事前計算
    print("Pre-calculating representative points...")
    rep_points = gdf.geometry.representative_point()
    gdf['rep_x'] = rep_points.x
    gdf['rep_y'] = rep_points.y

    # 2. Parquet形式で保存 (pyarrowが必要)
    print(f"Saving to {output_path}...")
    gdf.to_parquet(output_path)
    print("Done.")

if __name__ == "__main__":
    optimize_data('nation.geojson', 'nation.parquet')
