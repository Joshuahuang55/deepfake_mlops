import os
import boto3
from src.mlops_config import MLOpsConfig
import shutil

def merge_new_data():
    """
    從 MinIO 的 hard_samples 下載新圖片，並移動到 dataset/new_data 資料夾
    """
    # 1. 準備本地資料夾
    dataset_dir = "dataset/new_data"
    os.makedirs(dataset_dir, exist_ok=True)
    
    print("📡 連接 MinIO 下載新資料...")
    s3 = boto3.client(
        's3',
        endpoint_url=MLOpsConfig.S3_ENDPOINT,
        aws_access_key_id=MLOpsConfig.ACCESS_KEY,
        aws_secret_access_key=MLOpsConfig.SECRET_KEY
    )
    
    # 2. 列出 hard_samples 裡的所有檔案
    response = s3.list_objects_v2(Bucket=MLOpsConfig.BUCKET_NAME, Prefix=MLOpsConfig.HARD_SAMPLE_FOLDER)
    
    if 'Contents' not in response:
        print("⚠️ 沒有發現新資料 (hard_samples 是空的)")
        return
    
    count = 0
    for obj in response['Contents']:
        key = obj['Key']
        if key.endswith('/'): continue # 跳過資料夾本身
        
        filename = os.path.basename(key)
        local_path = os.path.join(dataset_dir, filename)
        
        # 下載檔案
        s3.download_file(MLOpsConfig.BUCKET_NAME, key, local_path)
        print(f"⬇️ 下載: {filename}")
        
        # (選擇性) 下載後刪除雲端備份，或移動到 'processed' 資料夾
        # s3.delete_object(Bucket=MLOpsConfig.BUCKET_NAME, Key=key)
        
        count += 1
        
    print(f"✅ 成功合併 {count} 筆新資料到 {dataset_dir}")
    print("🚀 下一步：請執行 'dvc add dataset/' 來追蹤這些變更")

if __name__ == "__main__":
    merge_new_data()