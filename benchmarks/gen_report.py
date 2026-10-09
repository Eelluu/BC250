import json, glob

all_results = []
for f in sorted(glob.glob("/opt/netscan/tmp/bench-results/bench_*.json")):
    with open(f) as fh:
        data = json.load(fh)
        for r in data:
            name = r.get("model","")
            if not any(x.get("model") == name for x in all_results):
                all_results.append(r)

ok = [r for r in all_results if r.get("status") == "OK"]
ok.sort(key=lambda r: r.get("speed_4k",{}).get("gen_tok_s",0), reverse=True)
for r in ok:
    sp = r.get("speed_4k",{})
    mc = r.get("max_ctx",0)
    cs = str(mc//1024)+"K" if mc >=1024 else str(mc)
    print(f"{r['model']}|{sp.get('gen_tok_s',0)}|{sp.get('prefill_tok_s',0)}|{cs}|{sp.get('vram_gib','?')}|{sp.get('gpu_pct','?')}")
