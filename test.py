import os
import sys
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_download_and_offline_load():
    # ================= 第一阶段：测试下载 =================
    print("=" * 50)
    print(f"[阶段 1] 开始下载模型: {MODEL_NAME}")
    print(f"当前设备: {DEVICE}")
    print("=" * 50)
    
    try:
        # 正常加载，如果本地没有会自动下载
        cos_model = SentenceTransformer(MODEL_NAME).to(DEVICE)
        print("\n✅ [阶段 1 成功] 模型下载/加载完成！")
        
        # 简单测试一下模型是否能正常工作
        test_vec = cos_model.encode(["Hello World"])
        print(f"   向量维度测试: {test_vec.shape}")
        
    except Exception as e:
        print(f"\n❌ [阶段 1 失败] 下载或加载模型时出错: {e}")
        sys.exit(1)

    # ================= 第二阶段：测试离线加载 =================
    print("\n" + "=" * 50)
    print("[阶段 2] 强制开启离线模式，测试是否还会联网...")
    print("=" * 50)
    
    # 强制开启离线模式，彻底禁止任何网络请求
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    try:
        # 再次加载模型
        cos_model_offline = SentenceTransformer(MODEL_NAME).to(DEVICE)
        print("\n✅ [阶段 2 成功] 离线加载成功！")
        print("💡 结论: 再次调用时【不会】再连接 Hugging Face，而是直接读取本地缓存。")
        
    except Exception as e:
        print(f"\n❌ [阶段 2 失败] 离线加载失败，说明本地缓存可能损坏: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_download_and_offline_load()