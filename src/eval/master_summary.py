"""OCRipple - combine CER, ANLS, Hit@3, email-recall into one master table."""
import pandas as pd
from pathlib import Path

R = Path("results/tables")
ENGINES = ["tesseract", "deepseek_ocr", "paddle"]

rows = []
for eng in ENGINES:
    cer_df = pd.read_csv(R / f"cer_{eng}_summary.csv", index_col=0, header=[0,1])
    clean_cer = cer_df.loc["clean", ("cer","mean")]
    heavy_cer = cer_df.loc["heavy", ("cer","mean")]

    rag_df = pd.read_csv(R / f"rag_{eng}_per_question.csv")
    hit3 = rag_df["hit"].mean()
    anls = rag_df["anls"].mean()

    email_df = pd.read_csv(R / f"email_recall_{eng}.csv", index_col=0)
    email_clean = email_df.loc["clean", "recall"]
    email_heavy = email_df.loc["heavy", "recall"]

    rows.append({
        "engine": eng,
        "CER_clean": round(clean_cer,3), "CER_heavy": round(heavy_cer,3),
        "Hit@3": round(hit3,3), "AnswerANLS": round(anls,3),
        "EmailRecall_clean": round(email_clean,3),
        "EmailRecall_heavy": round(email_heavy,3),
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv(R / "master_summary.csv", index=False)
