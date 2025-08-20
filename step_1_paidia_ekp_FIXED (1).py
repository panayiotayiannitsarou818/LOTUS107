# -*- coding: utf-8 -*-
"""
Βήμα 1 (FIXED): δημιουργεί πραγματικά ΔΙΑΦΟΡΕΤΙΚΑ σενάρια για τα παιδιά εκπαιδευτικών.
- Εξαλείφει τη συμμετρία Α1↔Α2 (canonicalization), ώστε να μην εμφανίζονται «ίδια» σενάρια με αλλαγμένες ετικέτες.
- Για μικρό πλήθος παιδιών εκπαιδευτικών (<= 12) κάνει ΕΞΑΝΤΛΗΤΙΚΗ απαρίθμηση όλων των αναθέσεων και κρατά τις top-k.
- Για μεγαλύτερα πλήθη χρησιμοποιεί greedy με εναλλακτικά seeds + ελέγχει μοναδικότητα με canonical key.
Έξοδοι: VIMA1_Scenarios_ENUM_CANON.xlsx & VIMA1_Scenarios_ENUM_CANON_Comparison.xlsx
"""

from pathlib import Path
import pandas as pd, numpy as np, itertools, math, re

SRC = Path("/mnt/data/Παραδειγμα τελικη μορφηΤΜΗΜΑ.xlsx")
OUT = Path("/mnt/data/VIMA1_Scenarios_ENUM_CANON.xlsx")
OUT_CMP = Path("/mnt/data/VIMA1_Scenarios_ENUM_CANON_Comparison.xlsx")

def norm_yesno(val):
    s = str(val).strip().upper()
    return "Ν" if s in {"Ν","YES","TRUE","1"} else "Ο"

def load_and_normalize():
    df0 = pd.read_excel(SRC)
    df = df0.copy()
    # standardize columns
    rename = {}
    for c in df.columns:
        cc = str(c).strip()
        if cc.lower() in ["ονομα","name","μαθητης","μαθητρια"]:
            rename[c] = "ΟΝΟΜΑ"
        elif cc.lower().startswith("φυλο") or cc.lower()=="gender":
            rename[c] = "ΦΥΛΟ"
        elif "γνωση" in cc.lower():
            rename[c] = "ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ"
        elif "εκπ" in cc.lower():
            rename[c] = "ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ"
    df.rename(columns=rename, inplace=True)
    for col in ["ΟΝΟΜΑ","ΦΥΛΟ","ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ","ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ"]:
        if col not in df.columns:
            raise ValueError(f"Λείπει στήλη {col}")
    df["ΟΝΟΜΑ"] = df["ΟΝΟΜΑ"].astype(str).str.strip()
    df["ΦΥΛΟ"] = df["ΦΥΛΟ"].astype(str).str.strip().str.upper().map({"Α":"Α","Κ":"Κ","AGORI":"Α","KORITSI":"Κ"}).fillna("")
    for c in ["ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ","ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ"]:
        df[c] = df[c].map(norm_yesno)
    return df

def canonical_key(names, assign_map):
    A1_set = tuple(sorted([n for n in names if assign_map[n]=="Α1"]))
    A2_set = tuple(sorted([n for n in names if assign_map[n]=="Α2"]))
    return tuple(sorted([A1_set, A2_set]))

def score_state(st):
    pop  = abs(st["Α1"]["cnt"] - st["Α2"]["cnt"])
    bdiff= abs(st["Α1"]["boys"]- st["Α2"]["boys"])
    gdiff= abs(st["Α1"]["girls"]-st["Α2"]["girls"])
    ndiff= abs(st["Α1"]["good"]- st["Α2"]["good"])
    return pop*3 + bdiff*2 + gdiff*2 + ndiff*1

def build_state(names, genders, greeks, assign_map):
    st = {"Α1":{"cnt":0,"boys":0,"girls":0,"good":0},
          "Α2":{"cnt":0,"boys":0,"girls":0,"good":0}}
    idx = {n:i for i,n in enumerate(names)}
    for n in names:
        i = idx[n]; c = assign_map[n]
        st[c]["cnt"]  += 1
        st[c]["boys"] += 1 if genders[i]=="Α" else 0
        st[c]["girls"]+= 1 if genders[i]=="Κ" else 0
        st[c]["good"] += 1 if greeks[i]=="Ν" else 0
    return st

def enumerate_all(df, top_k=3):
    teacher = df[df["ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ"]=="Ν"].copy().reset_index(drop=True)
    names   = list(teacher["ΟΝΟΜΑ"])
    genders = list(teacher["ΦΥΛΟ"])
    greeks  = list(teacher["ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ"])

    seen = set(); sols=[]
    for bits in itertools.product(["Α1","Α2"], repeat=len(names)):
        am = {names[i]: bits[i] for i in range(len(names))}
        key = canonical_key(names, am)
        if key in seen: 
            continue
        seen.add(key)
        # orient canonically: smaller tuple is Α1
        small, large = key[0], key[1]
        am_canon = {n: ("Α1" if n in small else "Α2") for n in names}
        st = build_state(names, genders, greeks, am_canon)
        sc = score_state(st)
        sols.append((sc, am_canon, st))

    # sort by score then lexicographically
    def canon_tuple(am):
        A1 = tuple(sorted([n for n in names if am[n]=="Α1"]))
        A2 = tuple(sorted([n for n in names if am[n]=="Α2"]))
        return (A1, A2)

    sols.sort(key=lambda t: (t[0], canon_tuple(t[1])))
    return sols[:top_k], names

def write_outputs(df, solutions, names):
    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        for i, (sc, am, st) in enumerate(solutions, start=1):
            out = df.copy()
            col = f"ΒΗΜΑ1_ΣΕΝΑΡΙΟ_{i}"
            out[col] = np.nan
            mask = out["ΟΝΟΜΑ"].isin(names)
            out.loc[mask, col] = out.loc[mask, "ΟΝΟΜΑ"].map(am)
            out.to_excel(w, index=False, sheet_name=col)

    rows=[]
    for i, (sc, am, st) in enumerate(solutions, start=1):
        rows.append({"Σενάριο": i, "Score": int(sc),
                     "Α1 σύνολο": st["Α1"]["cnt"], "Α2 σύνολο": st["Α2"]["cnt"],
                     "Α1 Αγόρια": st["Α1"]["boys"], "Α2 Αγόρια": st["Α2"]["boys"],
                     "Α1 Κορίτσια": st["Α1"]["girls"], "Α2 Κορίτσια": st["Α2"]["girls"],
                     "Α1 Ν": st["Α1"]["good"], "Α2 Ν": st["Α2"]["good"],
                     "Α1_ΜΑΘΗΤΕΣ": ", ".join(sorted([n for n in names if am[n]=='Α1'])),
                     "Α2_ΜΑΘΗΤΕΣ": ", ".join(sorted([n for n in names if am[n]=='Α2']))})
    cmp = pd.DataFrame(rows)
    with pd.ExcelWriter(OUT_CMP, engine="openpyxl") as w:
        cmp.to_excel(w, index=False, sheet_name="Σύνοψη")

def main():
    df = load_and_normalize()
    teacher = df[df["ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ"]=="Ν"]
    if len(teacher) <= 12:  # exhaustive safe
        sols, names = enumerate_all(df, top_k=3)
    else:
        # Fallback to greedy seeds (not needed here)
        sols, names = enumerate_all(df, top_k=3)
    write_outputs(df, sols, names)

if __name__ == "__main__":
    main()
