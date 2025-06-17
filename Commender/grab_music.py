import requests
import csv
import time
from datetime import datetime
import pandas as pd

def get_chinese_songs(artists=None, year_range=None, max_results=1000, output_path="chinese_songs.csv"):
    """
    获取中国地区歌曲信息并保存到CSV
    
    参数:
        artists: 艺术家列表(如 ['毛不易', '周杰伦'])
        year_range: 年份范围元组(如 (2010, 2024))
        max_results: 最大结果数量限制
        output_path: CSV文件保存路径
    """
    headers = {
        "User-Agent": "ChinaMusicCollector/1.0 (guoqin_gu@163.com)"
    }

    # 准备CSV文件
    with open(output_path, mode='w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['歌曲名称', '艺术家', '年份', 'MusicBrainz ID']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        total_count = 0
        
        # 如果传入了艺术家列表，则分批处理
        if artists:
            # 每5个艺术家一组
            batch_size = 5
            artist_batches = [artists[i:i + batch_size] for i in range(0, len(artists), batch_size)]
            
            for batch in artist_batches:
                print(f"正在处理艺术家批次: {batch}")
                batch_count = process_artist_batch(batch, year_range, max_results - total_count, 
                                                 writer, headers)
                total_count += batch_count
                
                if total_count >= max_results:
                    print(f"已达到最大结果限制 {max_results}")
                    break
        else:
            # 没有传入艺术家列表的情况
            total_count = process_artist_batch(None, year_range, max_results, writer, headers)
        
        print(f"全部完成! 共获取 {total_count} 条记录，已保存到 {output_path}")

def process_artist_batch(artists, year_range, max_batch_results, writer, headers):
    """
    处理一批艺术家的请求
    """
    # 构建查询字符串
    query_parts = ["country:CN"]
    
    if artists:
        artist_query = " OR ".join([f'artist:"{artist}"' for artist in artists])
        query_parts.append(f"({artist_query})")
    
    if year_range:
        start_year, end_year = year_range
        query_parts.append(f"date:[{start_year} TO {end_year}]")
    
    # 基本请求参数
    base_params = {
        "query": " AND ".join(query_parts),
        "fmt": "json",
        "inc": "artist-credits+releases",
        "limit": 100  # 每次请求最大结果数
    }

    batch_count = 0
    offset = 0
    
    while True:
        try:
            # 设置分页参数
            params = base_params.copy()
            params["offset"] = offset
            
            response = requests.get(
                "https://musicbrainz.org/ws/2/recording",
                params=params,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            
            recordings = data.get('recordings', [])
            if not recordings:
                break
            
            print(f"获取到 {len(recordings)} 条记录，当前批次总计 {batch_count + len(recordings)} 条...")
            
            for recording in recordings:
                # 歌曲信息
                title = recording.get('title', '未知标题')
                recording_id = recording.get('id', '')
                
                # 艺术家信息
                artists = []
                for credit in recording.get('artist-credit', []):
                    if isinstance(credit, dict):
                        if "artist" in credit:
                            artists.append(credit["artist"].get("name", "未知艺术家"))
                        elif "name" in credit:
                            artists.append(credit["name"])
                artist_str = ", ".join(artists) if artists else "未知艺术家"
                
                # 发行年份
                year = "未知年份"
                if 'releases' in recording and recording['releases']:
                    valid_dates = [r.get('date') for r in recording['releases'] if r.get('date')]
                    if valid_dates:
                        earliest_date = min(valid_dates)
                        if len(earliest_date) >= 4:
                            year = earliest_date[:4]
                
                # 写入CSV
                writer.writerow({
                    '歌曲名称': title,
                    '艺术家': artist_str,
                    '年份': year,
                    'MusicBrainz ID': recording_id
                })
                
                batch_count += 1
                if batch_count >= max_batch_results:
                    print(f"已达到当前批次最大结果限制 {max_batch_results}")
                    return batch_count
            
            offset += len(recordings)
            time.sleep(1)  # 遵守速率限制
            
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            break
    
    return batch_count

if __name__ == "__main__":
    # 配置参数
    df = pd.read_csv("data/supplement/music_meta_data/singer_from_train.csv")
    singers = df['singer'].unique().tolist()
    print('歌手数量', len(singers))
    # TARGET_ARTISTS = ["毛不易", "周杰伦", "新裤子", "周深", "海来阿木", "玄昌俊", "李思雨"]  # None表示不筛选艺术家
    TARGET_ARTISTS = singers
    TARGET_YEARS = (2000, 2024)  # None表示不筛选年份
    MAX_RESULTS = 200000  # 最大获取数量
    OUTPUT_FILE = "data/supplement/music_meta_data/chinese_songs.csv"  # 输出文件路径
    
    get_chinese_songs(
        artists=TARGET_ARTISTS,
        year_range=TARGET_YEARS,
        max_results=MAX_RESULTS,
        output_path=OUTPUT_FILE
    )