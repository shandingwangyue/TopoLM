"""
TopoLM: O(N^2) vs O(NK) Attention Complexity Benchmark
验证稀疏拓扑路由在长上下文下的算力与显存压制 (极客高精度版)
"""
import os, sys, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.topolm import TopoLM

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("注意: 当前在 CPU 环境下运行。如需查看真实算力压制，请在 GPU 环境下运行。")
        return
    
    seq_lengths = [1024, 2048, 4096, 8192] 
    print(f"{'Seq Len':<10} | {'Dense O(N^2) Time':<18} | {'Sparse O(NK) Time':<18}")
    print("-" * 55)

    # 0. 预热 (Warmup)
    dummy = torch.randint(0, 1000, (1, 128)).to(device)
    warmup_model = TopoLM(1000, sparse_mode=False, max_seq_len=200).to(device)
    with torch.no_grad():
        _ = warmup_model(dummy, active_domain_idx=0)
    torch.cuda.synchronize()

    for seq in seq_lengths:
        dummy_input = torch.randint(0, 1000, (1, seq)).to(device)
        
        # 官方 GPU 高精度计时器
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # 1. 测稠密 (Dense)
        model_dense = TopoLM(1000, sparse_mode=False, max_seq_len=seq+100).to(device)
        model_dense.eval()
        
        torch.cuda.synchronize()
        start_event.record()
        with torch.no_grad(): 
            _ = model_dense(dummy_input, active_domain_idx=0)
        end_event.record()
        torch.cuda.synchronize()
        dense_time = start_event.elapsed_time(end_event) # 返回高精度毫秒
        
        # 2. 测稀疏 (Sparse)
        model_sparse = TopoLM(1000, sparse_mode=True, router_min_active_k=64, max_seq_len=seq+100).to(device)
        model_sparse.eval()
        
        torch.cuda.synchronize()
        start_event.record()
        with torch.no_grad(): 
            _ = model_sparse(dummy_input, active_domain_idx=0)
        end_event.record()
        torch.cuda.synchronize()
        sparse_time = start_event.elapsed_time(end_event)
        
        print(f"{seq:<10} | {dense_time:>14.2f} ms | {sparse_time:>14.2f} ms")

if __name__ == "__main__":
    run_benchmark()