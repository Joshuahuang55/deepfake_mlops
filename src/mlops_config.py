# src/mlops_config.py
import mlflow
import boto3
import os
from io import BytesIO
import time

class MLOpsConfig:
    # 連線設定 (對應你的 Docker 環境)
    TRACKING_URI = "http://localhost:5001"  # 注意這裡是 5001
    EXPERIMENT_NAME = "Deepfake_Forensic_Live"
    
    # S3 (MinIO) 設定
    S3_ENDPOINT = "http://localhost:9000"   # 注意這裡是 9000
    ACCESS_KEY = "admin"
    SECRET_KEY = "testtestqaz123"
    BUCKET_NAME = "mlflow-bucket"
    HARD_SAMPLE_FOLDER = "hard_samples"

    @staticmethod
    def setup_mlflow():
        """初始化 MLflow 連線，如果實驗不存在就創建"""
        try:
            mlflow.set_tracking_uri(MLOpsConfig.TRACKING_URI)
            mlflow.set_experiment(MLOpsConfig.EXPERIMENT_NAME)
            return True
        except Exception as e:
            print(f"MLflow connection failed: {e}")
            return False

    @staticmethod
    def log_hard_sample(image, filename, confidence, prediction):
        """
        將低信心分數的圖片上傳到 S3 (MinIO)
        """
        s3 = boto3.client(
            's3',
            endpoint_url=MLOpsConfig.S3_ENDPOINT,
            aws_access_key_id=MLOpsConfig.ACCESS_KEY,
            aws_secret_access_key=MLOpsConfig.SECRET_KEY
        )
        
        # 將 PIL Image 轉為 Bytes
        img_byte_arr = BytesIO()
        # 確保格式正確，預設為 JPEG
        fmt = image.format if image.format else 'JPEG'
        image.save(img_byte_arr, format=fmt)
        img_byte_arr.seek(0)
        
        # 命名規則: hard_samples/FAKE_0.65_timestamp_filename.jpg
        # 加上 timestamp 避免檔名重複
        timestamp = int(time.time())
        s3_key = f"{MLOpsConfig.HARD_SAMPLE_FOLDER}/{prediction}_{confidence:.2f}_{timestamp}_{filename}"
        
        try:
            s3.upload_fileobj(img_byte_arr, MLOpsConfig.BUCKET_NAME, s3_key)
            return True, s3_key
        except Exception as e:
            return False, str(e)