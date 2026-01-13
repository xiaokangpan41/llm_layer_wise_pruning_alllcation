#!/usr/bin/env bash
set -euo pipefail

PYTHON="/home/xiaokang/miniconda3/envs/prune_llm/bin/python"
MAIN="/home/xiaokang/OWL/OWL-version1/OWL-main/main.py"
MODEL="/home/xiaokang/llama_hf/Llama-2-7b-hf"

OUT_CSV="risk_sweep.csv"
OUT_LOG="risk_sweep.log"

# ===================== 🔧 搜索范围（随便改） =====================
RISK_K_LIST=(1 3 5)
RISK_PROBE_LIST=(0.01 0.03 0.05 0.07)
RISK_DECAY_LIST=(0.1 0.3 0.5 0.7 0.8 0.9)
RISK_ALPHA_LIST=(0.1 0.2 0.3 0.4)
# ================================================================

# ---------- 初始化 ----------
if [[ ! -f "$OUT_CSV" ]]; then
  echo "risk_k,risk_probe,risk_decay,risk_alpha,ppl,seconds,exit_code" > "$OUT_CSV"
fi

tmp_out="$(mktemp)"
done_keys="$(mktemp)"
trap 'rm -f "$tmp_out" "$done_keys"' EXIT

# ---------- 读取已完成配置 ----------
awk -F',' 'NR>1 {print $1"|"$2"|"$3"|"$4}' "$OUT_CSV" | sort -u > "$done_keys"

echo "[INFO] Loaded $(wc -l < "$done_keys") completed runs"

# ---------- sweep ----------
for k in "${RISK_K_LIST[@]}"; do
  for probe in "${RISK_PROBE_LIST[@]}"; do
    for decay in "${RISK_DECAY_LIST[@]}"; do
      for alpha in "${RISK_ALPHA_LIST[@]}"; do

        key="$k|$probe|$decay|$alpha"
        if grep -qxF "$key" "$done_keys"; then
          echo "[SKIP] $key"
          continue
        fi

        echo
        echo "[RUN ] k=$k probe=$probe decay=$decay alpha=$alpha"
        start_ts=$(date +%s)

        set +e
        "$PYTHON" "$MAIN" \
          --model "$MODEL" \
          --model_name_or_path "$MODEL" \
          --Lamda 0.08 \
          --Hyper_m 5 \
          --prune_method prune_wanda_outlier_plus \
          --sparsity_ratio 0.7 \
          --sparsity_type unstructured \
          --risk_k "$k" \
          --risk_probe "$probe" \
          --risk_decay "$decay" \
          --risk_alpha "$alpha" \
          > "$tmp_out" 2>&1
        exit_code=$?
        set -e

        end_ts=$(date +%s)
        elapsed=$((end_ts - start_ts))

        ppl=$(awk '/ppl on wikitext/{x=$NF} END{if(x!="") print x; else print "NaN"}' "$tmp_out")

        echo "[TIME] ${elapsed}s = $(awk "BEGIN{printf \"%.2f\", $elapsed/60}") min"

        echo "$k,$probe,$decay,$alpha,$ppl,$elapsed,$exit_code" >> "$OUT_CSV"
        echo "$key" >> "$done_keys"

        {
          echo "===== $key | ${elapsed}s | exit=$exit_code ====="
          cat "$tmp_out"
        } >> "$OUT_LOG"

      done
    done
  done
done

echo "[DONE] sweep finished"
