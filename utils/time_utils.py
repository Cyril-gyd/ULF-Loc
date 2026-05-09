import time, torch
import json, os

def flush_stats(time_dict, mem_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "a", encoding="utf-8") as f:
        json.dump({"time": dict(time_dict), "gpu_mem": dict(mem_dict)}, f)
        f.write("\n")
    time_dict.clear()
    mem_dict.clear()
    
class _Timer:
    def __init__(self, name, cum_dict):
        self.name = name
        self.cum  = cum_dict
    def __enter__(self):
        self._start = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cum[self.name] += time.perf_counter() - self._start

class _GpuMem:
    def __init__(self, name, m_dict): self.n, self.d = name, m_dict
    def __enter__(self):
        torch.cuda.reset_peak_memory_stats()
        self.start = torch.cuda.max_memory_allocated()
        return self
    def __exit__(self, *_):
        self.d[self.n] += (torch.cuda.max_memory_allocated() - self.start) / 1024**2   # MB   