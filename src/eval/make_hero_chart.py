"""OCRipple - static PNG chart: CER vs downstream quality paradox."""
import matplotlib.pyplot as plt
import numpy as np

engines = ["Tesseract", "DeepSeek-OCR", "PaddleOCR"]
cer_clean = [7.5, 16.5, 47.6]
cer_heavy = [12.5, 15.8, 48.4]
anls = [71.2, 81.6, 84.8]

x = np.arange(len(engines))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x - width, cer_clean, width, label="CER, clean scan (lower = better)", color="#e34948")
b2 = ax.bar(x, cer_heavy, width, label="CER, heavy degradation (lower = better)", color="#eb6834")
b3 = ax.bar(x + width, anls, width, label="Answer ANLS on MP-DocVQA (higher = better)", color="#1baf7a")

for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=9)

ax.set_ylabel("percent")
ax.set_xticks(x)
ax.set_xticklabels(engines)
ax.set_ylim(0, 100)
ax.legend(loc="upper left", fontsize=9, frameon=False)
ax.set_title("Character error rate does not predict downstream answer quality",
             fontsize=11, pad=12)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linewidth=0.5, alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
out = "results/figures/cer_vs_anls.png"
plt.savefig(out, dpi=150)
print("saved:", out)
