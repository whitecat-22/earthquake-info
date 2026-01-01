import geopandas as gpd
import pandas as pd
import os

def create_unified_data(city_parquet_path, lake_parquet_path, nation_parquet_path, output_path):
    """
    市区町村データ、湖沼データ、全国データを統合したparquetファイルを作成

    Args:
        city_parquet_path: 市区町村データのparquetファイルパス
        lake_parquet_path: 湖沼データのparquetファイルパス
        nation_parquet_path: 全国データのparquetファイルパス
        output_path: 出力統合parquetファイルパス
    """
    print("=" * 60)
    print("統合データの作成を開始します")
    print("=" * 60)

    # 1. 市区町村データの読み込み
    print(f"\n1. 市区町村データを読み込み中: {city_parquet_path}")
    if not os.path.exists(city_parquet_path):
        raise FileNotFoundError(f"市区町村データが見つかりません: {city_parquet_path}")

    gdf_city = gpd.read_parquet(city_parquet_path)
    print(f"   - 行数: {len(gdf_city)}")
    print(f"   - カラム: {gdf_city.columns.tolist()}")
    print(f"   - ファイルサイズ: {os.path.getsize(city_parquet_path) / 1024 / 1024:.2f} MB")

    # データタイプを追加
    gdf_city['data_type'] = 'city'

    # 2. 湖沼データの読み込み
    print(f"\n2. 湖沼データを読み込み中: {lake_parquet_path}")
    if not os.path.exists(lake_parquet_path):
        print(f"   警告: 湖沼データが見つかりません: {lake_parquet_path}")
        print("   湖沼データなしで統合を続行します")
        gdf_lake = None
    else:
        gdf_lake = gpd.read_parquet(lake_parquet_path)
        print(f"   - 行数: {len(gdf_lake)}")
        print(f"   - カラム: {gdf_lake.columns.tolist()}")
        print(f"   - ファイルサイズ: {os.path.getsize(lake_parquet_path) / 1024 / 1024:.2f} MB")

        # データタイプを追加
        gdf_lake['data_type'] = 'lake'

    # 3. 全国データの読み込み（インセットマップ用、別途保持）
    print(f"\n3. 全国データを確認中: {nation_parquet_path}")
    if not os.path.exists(nation_parquet_path):
        print(f"   警告: 全国データが見つかりません: {nation_parquet_path}")
        print("   全国データは市区町村データから生成されます")
        gdf_nation = None
    else:
        gdf_nation = gpd.read_parquet(nation_parquet_path)
        print(f"   - 行数: {len(gdf_nation)}")
        print(f"   - ファイルサイズ: {os.path.getsize(nation_parquet_path) / 1024 / 1024:.2f} MB")

    # 4. カラムの統一（すべての重要なカラムを保持）
    print("\n4. カラムを統一中...")

    # すべてのカラムを収集（cityとlakeの両方から）
    all_cols = set(gdf_city.columns)
    if gdf_lake is not None:
        all_cols.update(gdf_lake.columns)
    all_cols = list(all_cols)

    # 重要なカラムを優先的に保持（city用）
    # 湖沼データは描画にgeometryのみを使用するため、属性カラムは不要
    city_priority_cols = ['geometry', 'data_type', 'code', 'CODE', 'N03_007', 'name', 'nam', 'name_ja',
                         'COMMNAME', 'CITYNAME', 'N03_004', 'rep_x', 'rep_y']
    # 湖沼データで使用しないカラム（削除対象）
    lake_exclude_cols = ['W09_001', 'W09_002', 'W09_003', 'W09_004']

    # 最終的なカラムリストを構築
    final_cols = []

    # 1. 必須カラム
    if 'geometry' not in final_cols:
        final_cols.append('geometry')
    if 'data_type' not in final_cols:
        final_cols.append('data_type')

    # 2. cityの優先カラム（存在するもの）
    for col in city_priority_cols:
        if col in all_cols and col not in final_cols:
            final_cols.append(col)

    # 3. その他のカラム（存在するもの、ただし湖沼の不要カラムは除外）
    for col in all_cols:
        if col not in final_cols and col not in lake_exclude_cols:
            final_cols.append(col)

    print(f"   - 統合後のカラム: {final_cols}")

    # 5. データの統合
    print("\n5. データを統合中...")
    # cityデータの準備
    city_cols_to_use = [col for col in final_cols if col in gdf_city.columns]
    gdf_city_prep = gdf_city[city_cols_to_use].copy()
    gdf_city_prep['data_type'] = 'city'
    gdf_list = [gdf_city_prep]

    if gdf_lake is not None:
        # lakeのカラムを統一（geometryとdata_typeのみ、属性カラムは削除）
        # 湖沼データは描画にgeometryのみを使用するため、不要な属性カラム（W09_001など）を削除
        lake_gdf = gdf_lake[['geometry']].copy()  # geometryのみ保持
        # data_typeを先に追加
        lake_gdf['data_type'] = 'lake'
        # final_colsに含まれるカラムで、lakeに存在しないものはNoneで埋める
        for col in final_cols:
            if col not in lake_gdf.columns:
                lake_gdf[col] = None
        # final_colsの順序に合わせる
        lake_gdf = lake_gdf[final_cols]
        gdf_list.append(lake_gdf)

    # 統合（カラムを統一してから結合）
    # すべてのDataFrameのカラムを統一
    all_cols = set()
    for gdf_item in gdf_list:
        all_cols.update(gdf_item.columns)
    all_cols = list(all_cols)

    # 各DataFrameに存在しないカラムを追加（Noneで埋める）し、順序を統一
    gdf_list_normalized = []
    for gdf_item in gdf_list:
        gdf_normalized = gdf_item.copy()
        for col in all_cols:
            if col not in gdf_normalized.columns:
                gdf_normalized[col] = None
        # カラムの順序を統一
        gdf_normalized = gdf_normalized[all_cols]
        gdf_list_normalized.append(gdf_normalized)

    # 統合
    gdf_unified = pd.concat(gdf_list_normalized, ignore_index=True)
    print(f"   - 統合後の行数: {len(gdf_unified)}")
    print(f"   - 市区町村データ: {len(gdf_unified[gdf_unified['data_type'] == 'city'])}")
    if gdf_lake is not None:
        print(f"   - 湖沼データ: {len(gdf_unified[gdf_unified['data_type'] == 'lake'])}")

    # 6. 空間インデックスの事前構築（高速化のため）
    print("\n6. 空間インデックスを構築中...")
    # sindexプロパティにアクセスすることで、R-treeインデックスを事前に構築
    # これにより、空間クエリ（cx[lim_w:lim_e, lim_s:lim_n]）が高速化される
    try:
        _ = gdf_unified.sindex  # インデックスを構築
        print("   - 統合データの空間インデックス構築完了")
    except Exception as e:
        print(f"   - 警告: 空間インデックスの構築に失敗しました: {e}")
        print("   - 読み込み時に自動的に構築されます")

    # 7. データ型の最適化
    print("\n7. データ型を最適化中...")
    # 文字列カラムの最適化はparquet保存時に自動で行われるため、ここではスキップ

    # 8. 統合データの保存
    print(f"\n8. 統合データを保存中: {output_path}")
    gdf_unified.to_parquet(output_path, compression='snappy', index=False)

    output_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"   - 保存完了: {output_size:.2f} MB")

    # 9. 全国データの保存（別ファイルとして保持、統合ファイルとは別）
    nation_output_path = output_path.replace('.parquet', '_nation.parquet')
    if gdf_nation is not None:
        print(f"\n9. 全国データを準備中: {nation_output_path}")
        # 全国データの空間インデックスを構築
        try:
            _ = gdf_nation.sindex  # インデックスを構築
            print("   - 全国データの空間インデックス構築完了")
        except Exception as e:
            print(f"   - 警告: 空間インデックスの構築に失敗しました: {e}")

        print(f"   全国データを保存中...")
        gdf_nation.to_parquet(nation_output_path, compression='snappy', index=False)
        nation_size = os.path.getsize(nation_output_path) / 1024 / 1024
        print(f"   - 保存完了: {nation_size:.2f} MB")
    else:
        # 全国データがない場合、統合データから生成（市区町村データのみを使用）
        print(f"\n9. 全国データを生成中: {nation_output_path}")
        city_only = gdf_unified[gdf_unified['data_type'] == 'city'].copy()
        # ジオメトリを統合（dissolve）
        if len(city_only) > 0:
            unified_geom = city_only.geometry.unary_union
            gdf_nation = gpd.GeoDataFrame(geometry=[unified_geom], crs=city_only.crs)
            # 代表点を計算
            rep_points = gdf_nation.geometry.representative_point()
            gdf_nation['rep_x'] = rep_points.x
            gdf_nation['rep_y'] = rep_points.y

            # 全国データの空間インデックスを構築
            try:
                _ = gdf_nation.sindex  # インデックスを構築
                print("   - 全国データの空間インデックス構築完了")
            except Exception as e:
                print(f"   - 警告: 空間インデックスの構築に失敗しました: {e}")

            gdf_nation.to_parquet(nation_output_path, compression='snappy', index=False)
            nation_size = os.path.getsize(nation_output_path) / 1024 / 1024
            print(f"   - 生成・保存完了: {nation_size:.2f} MB")

    # 9. サマリー
    print("\n" + "=" * 60)
    print("統合データの作成が完了しました")
    print("=" * 60)
    print(f"\n出力ファイル:")
    print(f"  - 統合データ: {output_path} ({output_size:.2f} MB)")
    print(f"  - 全国データ: {nation_output_path} ({nation_size:.2f} MB)")

    total_original_size = (os.path.getsize(city_parquet_path) / 1024 / 1024)
    if gdf_lake is not None:
        total_original_size += (os.path.getsize(lake_parquet_path) / 1024 / 1024)
    if gdf_nation is not None:
        total_original_size += (os.path.getsize(nation_parquet_path) / 1024 / 1024)

    total_output_size = output_size + nation_size
    reduction = ((total_original_size - total_output_size) / total_original_size * 100) if total_original_size > 0 else 0

    print(f"\nファイルサイズ比較:")
    print(f"  - 元の合計: {total_original_size:.2f} MB")
    print(f"  - 統合後: {total_output_size:.2f} MB")
    print(f"  - 削減率: {reduction:.1f}%")
    print("\n")

if __name__ == "__main__":
    # デフォルトパス
    CITY_PARQUET = 'city.parquet'
    LAKE_PARQUET = 'lake.parquet'
    NATION_PARQUET = 'nation.parquet'
    OUTPUT_PARQUET = 'unified_data.parquet'

    # コマンドライン引数の処理（オプション）
    import sys
    if len(sys.argv) > 1:
        CITY_PARQUET = sys.argv[1]
    if len(sys.argv) > 2:
        LAKE_PARQUET = sys.argv[2]
    if len(sys.argv) > 3:
        NATION_PARQUET = sys.argv[3]
    if len(sys.argv) > 4:
        OUTPUT_PARQUET = sys.argv[4]

    create_unified_data(CITY_PARQUET, LAKE_PARQUET, NATION_PARQUET, OUTPUT_PARQUET)

