import torch
import sys
import os

# 設定你的模型檔案路徑 (請依實際情況修改)
MODEL_PATH = "models/efficientnet_b4.pth"  # <--- 請確認這裡改成你真正報錯的那個檔案路徑

def inspect_checkpoint():
    print(f"🔍 Inspecting: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print("❌ File not found!")
        return

    try:
        # 1. 載入檔案
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        
        # 2. 判斷格式
        state_dict = None
        if isinstance(checkpoint, dict):
            print(f"ℹ️ Checkpoint Keys: {list(checkpoint.keys())}")
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("✅ Found 'model_state_dict'")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("✅ Found 'state_dict'")
            else:
                # 假設整個 dict 都是權重
                state_dict = checkpoint
                print("ℹ️ Assuming entire dict is state_dict")
        else:
            print("⚠️ Checkpoint is not a dict (might be raw model?)")
            return

        # 3. 印出前 10 個 Key 的名字
        print("\n--- 檔案裡的前 10 個權重名稱 (File Keys) ---")
        file_keys = list(state_dict.keys())
        for i, key in enumerate(file_keys[:10]):
            print(f"{i}: {key}")
            
        # 4. 檢查是否有常見的前綴
        first_key = file_keys[0]
        if first_key.startswith("module."):
            print("\n💡 提示: 偵測到 'module.' 前綴 (DataParallel 訓練)")
        elif first_key.startswith("model."):
            print("\n💡 提示: 偵測到 'model.' 前綴 (可能包含在 class wrapper 中)")
        elif first_key.startswith("backbone."):
            print("\n💡 提示: 偵測到 'backbone.' 前綴")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    inspect_checkpoint()