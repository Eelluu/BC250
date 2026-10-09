#!/usr/bin/env python3
"""Generate benchmark charts for new BC-250 features:
 - Whisper transcription speed (real speech, wall time)
 - Whisper memory impact (swap delta alongside Ollama)
 - Model speed comparison (draft/9B/MoE for routing context)
 - Speculative decoding feasibility (memory budget)
 - Signal pipeline architecture overview

Dark theme matching existing charts.
Data source: real TTS speech benchmarks on BC-250.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT_DIR = "../images/charts"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Color palette ──────────────────────────────────────────────────────
C_TURBO  = "#5cb85c"
C_LARGE  = "#e87d5f"
C_MOE    = "#4a90d9"
C_9B     = "#e67e22"
C_3B     = "#9b59b6"
C_VRAM   = "#e74c3c"
C_KV     = "#f39c12"
C_FREE   = "#2ecc71"
C_SWAP   = "#e74c3c"
C_DARK   = "#2c3e50"

plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'font.family': 'sans-serif',
    'font.size': 11,
})


# ══════════════════════════════════════════════════════════════════════
# CHART 1: Whisper Transcription — Wall Time (real speech)
# ══════════════════════════════════════════════════════════════════════

def chart_whisper_wall():
    """Bar chart comparing wall time for large-v3-turbo vs large-v3
    using real TTS speech (flite), not silence."""
    # Real speech benchmarks: en_short (3.6s), en_medium (18.2s), en_long (39.2s)
    labels = ['3.6s\n(short)', '18.2s\n(medium)', '39.2s\n(long)']
    turbo_wall = [3.33, 3.46, 4.30]
    large_wall = [7.90, 8.94, 8.09]

    x = np.arange(len(labels))
    w = 0.32

    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(x - w/2, turbo_wall, w, color=C_TURBO, label='large-v3-turbo (1.6 GB)',
                edgecolor='none', zorder=3)
    b2 = ax.bar(x + w/2, large_wall, w, color=C_LARGE, label='large-v3 (2.9 GB)',
                edgecolor='none', zorder=3)

    # Value labels — safe placement above each bar
    for rect in b1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, h + 0.2, f'{h:.1f}s',
                ha='center', va='bottom', fontsize=10, color=C_TURBO, fontweight='bold')
    for rect in b2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, h + 0.2, f'{h:.1f}s',
                ha='center', va='bottom', fontsize=10, color=C_LARGE, fontweight='bold')

    # Speedup annotations — placed well below title
    for i in range(len(labels)):
        ratio = large_wall[i] / turbo_wall[i]
        mid_x = x[i]
        ax.text(mid_x, max(turbo_wall[i], large_wall[i]) + 1.2,
                f'{ratio:.1f}×', ha='center', va='bottom',
                fontsize=9, color='#8b949e', style='italic')

    ax.set_xlabel('Audio Duration (real TTS speech)')
    ax.set_ylabel('Wall Time (seconds)')
    ax.set_title('Whisper Transcription Speed — Real Speech — BC-250 (Vulkan)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper left', framealpha=0.8, edgecolor='#30363d', facecolor='#161b22')
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_ylim(0, 13)

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/whisper-wall-time.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {OUT_DIR}/whisper-wall-time.png")


# ══════════════════════════════════════════════════════════════════════
# CHART 2: Whisper Memory Impact (swap delta alongside Ollama)
# ══════════════════════════════════════════════════════════════════════

def chart_whisper_memory():
    """Bar chart showing swap increase caused by each whisper model
    when Ollama is already loaded (10 GB+ model in VRAM)."""
    labels = ['3.6s\n(short)', '18.2s\n(medium)', '39.2s\n(long)']
    turbo_swap = [0, 2, 0]       # MB swap delta
    large_swap = [1043, 1209, 0]  # MB swap delta (0 on 3rd = already swapped)

    x = np.arange(len(labels))
    w = 0.32

    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(x - w/2, turbo_swap, w, color=C_TURBO, label='large-v3-turbo (1.6 GB)',
                edgecolor='none', zorder=3)
    b2 = ax.bar(x + w/2, large_swap, w, color=C_SWAP, label='large-v3 (2.9 GB)',
                edgecolor='none', zorder=3)

    # Value labels — safe placement above each bar
    for i, rect in enumerate(b1):
        h = rect.get_height()
        label = f'{turbo_swap[i]} MB' if turbo_swap[i] > 0 else '0'
        ax.text(rect.get_x() + rect.get_width()/2, max(h, 10) + 20,
                label, ha='center', va='bottom', fontsize=10,
                color=C_TURBO, fontweight='bold')
    for i, rect in enumerate(b2):
        h = rect.get_height()
        if large_swap[i] > 0:
            ax.text(rect.get_x() + rect.get_width()/2, h + 20,
                    f'+{large_swap[i]} MB', ha='center', va='bottom',
                    fontsize=10, color=C_SWAP, fontweight='bold')
        else:
            ax.text(rect.get_x() + rect.get_width()/2, 30,
                    '0*', ha='center', va='bottom',
                    fontsize=10, color=C_SWAP, fontweight='bold')

    # Danger zone
    ax.axhspan(500, 1400, alpha=0.08, color=C_SWAP, zorder=1)
    ax.text(2.4, 1300, 'swap pressure zone', ha='right', fontsize=9,
            color=C_SWAP, alpha=0.6, style='italic')

    # Footnote for the 0 on 3rd run
    ax.text(2.0, -280, '* 3rd run: 0 swap because earlier runs already evicted pages',
            fontsize=8, color='#6c757d', ha='center', style='italic')

    ax.set_xlabel('Audio Duration (real TTS speech)', labelpad=30)
    ax.set_ylabel('Swap Increase (MB)')
    ax.set_title('Whisper Memory Impact — Alongside Ollama (10 GB model loaded)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper left', framealpha=0.8, edgecolor='#30363d', facecolor='#161b22')
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_ylim(-50, 1400)

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/whisper-memory.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {OUT_DIR}/whisper-memory.png")



# ══════════════════════════════════════════════════════════════════════
# CHART 4: Speculative Decoding — Memory Budget
# ══════════════════════════════════════════════════════════════════════

def chart_spec_decode_memory():
    """Stacked bar showing memory usage for spec decode scenario."""
    scenarios = ['Draft only\n(qwen2.5:3b)', 'Verifier only\n(Qwen3.5 MoE 35B-A3B)', 'Dual load\n(spec decode)', 'Available\n(16 GB UMA)']
    draft_sizes = [1.8, 0, 1.8, 0]
    verifier_sizes = [0, 10.6, 10.6, 0]
    kv_sizes = [0.2, 1.8, 2.0, 0]
    os_sizes = [1.5, 1.5, 1.5, 0]
    available = [0, 0, 0, 16]

    x = np.arange(len(scenarios))
    w = 0.5

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Available total
    ax.bar(x, available, w, color=C_FREE, alpha=0.2, edgecolor=C_FREE, linewidth=1.5, zorder=2, label='Available')

    # Stacked components
    b1 = ax.bar(x, draft_sizes, w, color=C_3B, edgecolor='none', zorder=3, label='Draft (3B)')
    b2 = ax.bar(x, verifier_sizes, w, bottom=draft_sizes, color=C_MOE, edgecolor='none', zorder=3, label='Verifier (MoE)')
    b3_bottom = [d + v for d, v in zip(draft_sizes, verifier_sizes)]
    b3 = ax.bar(x, kv_sizes, w, bottom=b3_bottom, color=C_KV, edgecolor='none', zorder=3, label='KV cache')
    b4_bottom = [b + k for b, k in zip(b3_bottom, kv_sizes)]
    b4 = ax.bar(x, os_sizes, w, bottom=b4_bottom, color='#6c757d', edgecolor='none', zorder=3, label='OS + overhead')

    # Total labels
    totals = [sum(x) for x in zip(draft_sizes, verifier_sizes, kv_sizes, os_sizes)]
    for i in range(3):
        t = totals[i]
        ax.text(x[i], t + 0.3, f'{t:.1f} GB', ha='center', va='bottom',
                fontsize=11, fontweight='bold',
                color='#e74c3c' if t > 14 else '#c9d1d9')

    # 16 GB limit line
    ax.axhline(y=16, color='#e74c3c', linestyle='--', alpha=0.6, zorder=4)
    ax.text(3.35, 16.3, '16 GB UMA limit', ha='right', fontsize=9, color='#e74c3c', alpha=0.8)

    # BLOCKED watermark on dual load
    ax.text(2, 8, 'BLOCKED\nby Ollama', ha='center', va='center',
            fontsize=16, fontweight='bold', color='#e74c3c', alpha=0.4,
            rotation=15, zorder=5)

    ax.set_ylabel('Memory (GB)')
    ax.set_title('Speculative Decoding Memory Budget — BC-250', fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.legend(loc='upper left', framealpha=0.8, edgecolor='#30363d', facecolor='#161b22', fontsize=9)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_ylim(0, 18.5)

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/spec-decode-memory.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {OUT_DIR}/spec-decode-memory.png")



# ══════════════════════════════════════════════════════════════════════
# CHART 6: Whisper Memory Budget (stacked, shows why turbo fits)
# ══════════════════════════════════════════════════════════════════════

def chart_whisper_budget():
    """Horizontal stacked bar showing memory layout for each scenario:
    Ollama model + whisper model + OS/buffers vs 16 GB physical RAM."""
    scenarios = [
        'Ollama only\n(baseline)',
        'Ollama +\nlarge-v3-turbo',
        'Ollama +\nlarge-v3',
    ]
    # Component sizes in GB
    ollama_model = [10.6, 10.6, 10.6]
    whisper_model = [0.0, 1.6, 2.9]
    os_buffers = [3.5, 3.5, 3.5]
    total = [o + w + b for o, w, b in zip(ollama_model, whisper_model, os_buffers)]

    y = np.arange(len(scenarios))
    h = 0.5

    fig, ax = plt.subplots(figsize=(10, 4))

    # Stacked horizontal bars
    b1 = ax.barh(y, ollama_model, h, color=C_MOE, label='Ollama model (10.6 GB)',
                 edgecolor='none', zorder=3)
    b2 = ax.barh(y, whisper_model, h, left=ollama_model, color=C_TURBO,
                 label='Whisper model', edgecolor='none', zorder=3)
    b3 = ax.barh(y, os_buffers, h,
                 left=[o + w for o, w in zip(ollama_model, whisper_model)],
                 color='#6c757d', label='OS + buffers (~3.5 GB)',
                 edgecolor='none', zorder=3)

    # 16 GB line
    ax.axvline(x=16.0, color=C_SWAP, linewidth=2, linestyle='--', zorder=4, alpha=0.9)
    ax.text(16.1, 2.35, '16 GB\nphysical\nRAM', fontsize=9, color=C_SWAP,
            fontweight='bold', va='top')

    # Overflow annotation for large-v3
    overflow = total[2] - 16.0
    if overflow > 0:
        ax.annotate(f'+{overflow:.1f} GB → swap',
                    xy=(16.0, 0), xytext=(17.5, -0.15),
                    fontsize=10, color=C_SWAP, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=C_SWAP, lw=1.5),
                    zorder=5)

    # "Fits" annotation for turbo
    ax.text(total[1] + 0.2 + 1.2, 1, '✓ fits', fontsize=10, color=C_TURBO,
            fontweight='bold', va='center')

    # Total labels at end of each bar
    for i, t in enumerate(total):
        ax.text(t + 0.15, i, f'{t:.1f} GB', fontsize=9, va='center',
                color='#8b949e')

    # Component size labels inside bars
    for i in range(len(scenarios)):
        # Ollama label (always show)
        ax.text(ollama_model[i] / 2, i, '10.6', fontsize=9, ha='center',
                va='center', color='white', fontweight='bold')
        # Whisper label (if visible)
        if whisper_model[i] > 0.5:
            mid = ollama_model[i] + whisper_model[i] / 2
            ax.text(mid, i, f'{whisper_model[i]}', fontsize=9, ha='center',
                    va='center', color='white', fontweight='bold')

    ax.set_xlabel('Memory (GB)')
    ax.set_title('Whisper Memory Budget — Why large-v3-turbo Wins on 16 GB UMA',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_yticks(y)
    ax.set_yticklabels(scenarios)
    ax.set_xlim(0, 19)
    ax.legend(loc='lower left', framealpha=0.8, edgecolor='#30363d',
              facecolor='#161b22', fontsize=9)
    ax.grid(axis='x', alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/whisper-memory-budget.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {OUT_DIR}/whisper-memory-budget.png")


# ══════════════════════════════════════════════════════════════════════

def main():
    print("Generating new feature benchmark charts...")
    chart_whisper_wall()
    chart_whisper_memory()
    chart_whisper_budget()
    chart_spec_decode_memory()
    print("Done — all charts saved to images/charts/")


if __name__ == "__main__":
    main()
