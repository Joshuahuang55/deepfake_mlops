import mlflow
import os

# 設定環境變數讓 Python 知道怎麼連到我們剛架好的 MinIO (S3)
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "admin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testtestqaz123"

# 設定 Tracking Server 位置 (指向我們剛啟動的 MLflow Server)
mlflow.set_tracking_uri("http://localhost:5001")

# 設定一個實驗名稱
mlflow.set_experiment("Deepfake_Ops_Test")

print("開始連接 MLOps 系統...")

with mlflow.start_run():
    print("正在記錄參數與指標...")
    # 1. 記錄參數
    mlflow.log_param("test_param", "hello_mlops")

    # 2. 記錄指標
    mlflow.log_metric("accuracy", 0.99)

    # 3. 建立一個假模型檔案並上傳
    print("正在上傳模型檔案...")
    with open("dummy_model.txt", "w") as f:
        f.write("這是一個測試用的假模型檔案")

    mlflow.log_artifact("dummy_model.txt")

print("測試成功！請前往 MLflow UI 查看結果。")