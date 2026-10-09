#!/usr/bin/env python3
"""Test MoE above 128K, then test FILLED context performance."""
import json, time, urllib.request, subprocess, sys, os
sys.stdout.reconfigure(line_buffering=True)

OLLAMA = "http://localhost:11434"

def api(ep, data=None, timeout=600):
    url = f"{OLLAMA}{ep}"
    req = urllib.request.Request(url, method="POST" if data else "GET")
    if data:
        req.add_header("Content-Type", "application/json")
        body = json.dumps(data).encode()
    else:
        body = None
    try:
        with urllib.request.urlopen(req, body, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e), "error_type": type(e).__name__}

def gtt():
    with open("/sys/class/drm/card1/device/mem_info_gtt_used") as f:
        return int(f.read().strip()) / (1024**3)

def free_mem():
    out = subprocess.check_output(["free", "-m"], text=True)
    lines = out.strip().split("\n")
    mem = lines[1].split()
    swap = lines[2].split()
    return f"free={mem[3]}M avail={mem[6]}M swap={swap[2]}M"

model = "qwen3.5-35b-a3b-iq2m"

# ─── Part 1: Push above 128K ───
print("=" * 60)
print("Part 1: Testing above 128K (empty context)")
print("=" * 60)
for ctx in [131072, 163840, 196608, 262144]:
    ctx_k = f"{ctx//1024}K"
    print(f"\nRestarting for {ctx_k}...", end=" ", flush=True)
    subprocess.run(["sudo", "systemctl", "restart", "ollama"], timeout=30, capture_output=True)
    time.sleep(12)
    print("gen...", end=" ", flush=True)
    t0 = time.time()
    resp = api("/api/generate", {
        "model": model, "prompt": "Explain quantum computing briefly.",
        "stream": False,
        "options": {"num_ctx": ctx, "num_predict": 30},
        "think": False, "keep_alive": "1m",
    }, timeout=600)
    elapsed = time.time() - t0
    if "error" in resp:
        print(f"FAIL ({elapsed:.0f}s): {resp['error'][:120]}", flush=True)
        # Check if it was actually allocated smaller
        ps = api("/api/ps")
        if ps and "models" in ps:
            for m in ps["models"]:
                actual = m.get("context_length", 0)
                sv = m.get("size_vram", 0) / (1024**3)
                print(f"  actual_ctx={actual}, vram={sv:.3f}G", flush=True)
        break
    else:
        ed = resp.get("eval_duration", 0) / 1e9
        ec = resp.get("eval_count", 0)
        gen = round(ec / ed, 1) if ed > 0 else 0
        ps = api("/api/ps")
        sv, actual_ctx = 0, 0
        if ps and "models" in ps:
            for m in ps["models"]:
                sv = m.get("size_vram", 0) / (1024**3)
                actual_ctx = m.get("context_length", 0)
        g = gtt()
        fm = free_mem()
        print(f"OK {gen} tok/s, VRAM={sv:.3f}G, GTT={g:.2f}G, ctx={actual_ctx}, {fm}", flush=True)

# ─── Part 2: Filled context performance ───
print("\n\n" + "=" * 60)
print("Part 2: Filled context performance (MoE at 32K ctx)")
print("Speed should degrade as context fills up")
print("=" * 60)

# Generate a very long prompt by repeating technical content
base_text = """Memory management in modern operating systems is a critical subsystem 
that handles the allocation, deallocation, and organization of computer memory. Virtual 
memory provides each process with an isolated address space, typically using paging to 
map virtual addresses to physical frames. The Translation Lookaside Buffer (TLB) caches 
recent page table entries to avoid the overhead of walking multi-level page tables on every 
memory access. Modern CPUs implement multiple levels of TLB (L1 dTLB, L1 iTLB, L2 sTLB) 
with varying sizes and associativity. On x86-64, a typical L1 dTLB has 64 entries for 4KB 
pages and 32 entries for 2MB/1GB huge pages. TLB misses can cost 7-25 cycles depending on 
page table depth and cache hit rates. Linux supports transparent huge pages (THP) which 
automatically promote 4KB pages to 2MB pages when beneficial, reducing TLB pressure for 
applications with large working sets. The kernel's memory allocator uses a buddy system 
for physical page allocation, with slab allocators (SLUB, SLAB, SLOB) layered on top for 
efficient small-object allocation. NUMA awareness is increasingly important as multi-socket 
systems become common - the kernel's NUMA balancing automatically migrates pages to be 
closer to the accessing CPU, while applications can use numactl or libnuma for explicit 
placement. The page cache provides transparent caching of file data in memory, with a 
sophisticated LRU algorithm that distinguishes between active and inactive pages. Memory 
pressure triggers the page reclaim path, which uses kswapd for background reclaim and 
direct reclaim when allocation is urgent. The OOM killer is the last resort when the 
system runs critically low on memory. For GPU-accelerated workloads on unified memory 
architectures like AMD APUs, the GTT (Graphics Translation Table) provides a mechanism 
for the GPU to access system memory directly. The TTM (Translation Table Maps) memory 
manager in the kernel handles allocation and migration of buffers between different 
memory domains (VRAM, GTT, system memory). On discrete GPUs, TTM manages buffer migration 
across the PCIe bus, but on UMA systems like the BC-250, all memory domains map to the 
same physical pool - making fragmentation a potential issue when the GPU and CPU compete 
for large contiguous allocations. """

# Fill prompts of different sizes
fill_sizes = [
    ("~500 tokens", base_text[:2000]),
    ("~2K tokens", base_text * 4),
    ("~5K tokens", base_text * 10),
    ("~10K tokens", base_text * 20),
    ("~20K tokens", base_text * 40),
    ("~30K tokens", base_text * 60),
]

print(f"\nRestarting for 32K ctx...", flush=True)
subprocess.run(["sudo", "systemctl", "restart", "ollama"], timeout=30, capture_output=True)
time.sleep(12)

# Warm up
print("Warming up...", flush=True)
api("/api/generate", {
    "model": model, "prompt": "Hello", "stream": False,
    "options": {"num_ctx": 32768, "num_predict": 10},
    "think": False, "keep_alive": "30m",
}, timeout=300)
time.sleep(3)

results = []
for label, prompt_text in fill_sizes:
    prompt = prompt_text + "\n\nBased on the above, write a brief conclusion."
    print(f"\n  Fill={label} ({len(prompt)} chars)...", end=" ", flush=True)
    t0 = time.time()
    resp = api("/api/generate", {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"num_ctx": 32768, "num_predict": 100, "temperature": 0.7},
        "think": False, "keep_alive": "30m",
    }, timeout=600)
    elapsed = time.time() - t0
    
    if "error" in resp:
        print(f"FAIL ({elapsed:.0f}s): {resp['error'][:100]}", flush=True)
        results.append({"fill": label, "status": "FAIL"})
    else:
        ed = resp.get("eval_duration", 0) / 1e9
        ec = resp.get("eval_count", 0)
        pc = resp.get("prompt_eval_count", 0)
        pd = resp.get("prompt_eval_duration", 0) / 1e9
        gen = round(ec / ed, 1) if ed > 0 else 0
        pre = round(pc / pd, 1) if pd > 0 else 0
        g = gtt()
        print(f"OK gen={gen} tok/s, prefill={pre} tok/s ({pc} tokens in {pd:.1f}s), "
              f"GTT={g:.2f}G, total={elapsed:.0f}s", flush=True)
        results.append({
            "fill": label, "status": "OK", "gen_tok_s": gen,
            "prefill_tok_s": pre, "prompt_tokens": pc,
            "gtt_gib": g, "elapsed": elapsed,
        })

print("\n=== FILLED CONTEXT SUMMARY ===")
for r in results:
    if r["status"] == "OK":
        print(f"  {r['fill']:>15}: gen={r['gen_tok_s']:>5.1f} tok/s  "
              f"prefill={r['prefill_tok_s']:>6.1f} tok/s  "
              f"tokens={r['prompt_tokens']:>6}  GTT={r['gtt_gib']:.2f}G")
    else:
        print(f"  {r['fill']:>15}: FAIL")

os.makedirs("/tmp/bc250-ctx-bench", exist_ok=True)
with open("/tmp/bc250-ctx-bench/task0d-filled.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: /tmp/bc250-ctx-bench/task0d-filled.json")
