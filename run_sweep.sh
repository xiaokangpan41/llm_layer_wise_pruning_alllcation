#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="0"
#export HF_DATASETS_CACHE="/data1/LLM_models/dataset/corpus"

PYTHON="/home/xiaokang/miniconda3/envs/prune_llm/bin/python"
MAIN="/home/xiaokang/OWL/OWL-version2/risk-prune/main.py"
MODEL="/home/xiaokang/llama_hf/Llama-2-7b-hf"

OUT_CSV="risk_sweep.csv"
OUT_LOG="risk_sweep.log"


# ===================== 🔧 搜索范围 =====================
RISK_K_LIST=(1 2 3 4 5)
RISK_PROBE_LIST=(0.01 0.05 0.1 0.15 0.2 0.25 0.3)
RISK_DECAY_LIST=(0.5 0.6 0.7 0.8 0.9)
RISK_DECAY_TYPE_LIST=("linear" "exponential")
RISK_DECAY_BETA_LIST=(0.1 0.2 0.3 )
RISK_ALPHA_LIST=(0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4)
CONSIDER_CURRENT_LAYER_LIST=("False" "True")
# ======================================================

# ---------- 初始化 CSV ----------
if [[ ! -f "$OUT_CSV" ]]; then
  echo "risk_k,risk_probe,risk_decay,risk_decay_type,risk_decay_beta,risk_alpha,consider_current_layer,ppl,seconds,exit_code" > "$OUT_CSV"
fi

tmp_out="$(mktemp)"
done_keys="$(mktemp)"
trap 'rm -f "$tmp_out" "$done_keys"' EXIT

# ---------- 已完成任务 key ----------
awk -F',' 'NR>1 {
  print $1"|"$2"|"$3"|"$4"|"$5"|"$6"|"$7
}' "$OUT_CSV" | sort -u > "$done_keys"

echo "[INFO] Loaded $(wc -l < "$done_keys") completed runs"

# ---------- sweep ----------
for k in "${RISK_K_LIST[@]}"; do
  for probe in "${RISK_PROBE_LIST[@]}"; do
    for decay in "${RISK_DECAY_LIST[@]}"; do
      for decay_type in "${RISK_DECAY_TYPE_LIST[@]}"; do
        for decay_beta in "${RISK_DECAY_BETA_LIST[@]}"; do
          for alpha in "${RISK_ALPHA_LIST[@]}"; do
            for consider in "${CONSIDER_CURRENT_LAYER_LIST[@]}"; do

              key="$k|$probe|$decay|$decay_type|$decay_beta|$alpha|$consider"
              if grep -qxF "$key" "$done_keys"; then
                echo "[SKIP] $key"
                continue
              fi

              echo
              echo "[RUN ] k=$k probe=$probe decay=$decay type=$decay_type beta=$decay_beta alpha=$alpha consider=$consider"
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
                --risk_decay_type "$decay_type" \
                --risk_decay_beta "$decay_beta" \
                --risk_alpha "$alpha" \
                --consider_current_layer "$consider" \
                > "$tmp_out" 2>&1
              exit_code=$?
              set -e

              end_ts=$(date +%s)
              elapsed=$((end_ts - start_ts))

              ppl=$(awk '/ppl on wikitext/{x=$NF} END{if(x!="") print x; else print "NaN"}' "$tmp_out")

              echo "$k,$probe,$decay,$decay_type,$decay_beta,$alpha,$consider,$ppl,$elapsed,$exit_code" >> "$OUT_CSV"
              echo "$key" >> "$done_keys"

              {
                echo "===== $key | ${elapsed}s | exit=$exit_code ====="
                cat "$tmp_out"
              } >> "$OUT_LOG"

            done
          done
        done
      done
    done
  done
done

echo "[DONE] sweep finished"
