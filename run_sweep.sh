#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="0,1"
export HF_DATASETS_CACHE="/data1/LLM_models/dataset/corpus"
export HF_DATASETS_OFFLINE=1

PYTHON="/home/pxiaokang/miniconda3/envs/risk-prune/bin/python"
MAIN="/home/pxiaokang/risk-prune/main.py"
MODEL="/data1/LLM_models/LLM/Llama/llama2-7b-hf"

OUT_CSV="risk_sweep.csv"
OUT_LOG="risk_sweep.log"

# ===================== 🔧 搜索范围（随便改） =====================
RISK_K_LIST=(1 3 5)
RISK_PROBE_LIST=(0.01 0.03 0.05 0.07)
RISK_DECAY_LIST=(0.1 0.3 0.5 0.7 0.8 0.9)
RISK_DECAY_TYPE_LIST=("linear" "exponential")
RISK_DECAY_BETA_LIST=(0.01 0.019 0.001)
RISK_ALPHA_LIST=(0.1 0.2 0.3 0.4)
CONSIDER_CURRENT_LAYER_LIST=("False" "True")
# ================================================================

# ---------- 初始化 ----------
if [[ ! -f "$OUT_CSV" ]]; then
  echo "risk_k,risk_probe,risk_decay,risk_decay_type,risk_decay_beta,risk_alpha,consider_current_layer_in_risk,ppl,seconds,exit_code" > "$OUT_CSV"
fi

tmp_out="$(mktemp)"
done_keys="$(mktemp)"
trap 'rm -f "$tmp_out" "$done_keys"' EXIT

# ---------- 读取已完成配置 ----------
awk -F',' 'NR>1 {print $1"|"$2"|"$3"|"$4"|"$5"|"$6"|"$7}' "$OUT_CSV" | sort -u > "$done_keys"

echo "[INFO] Loaded $(wc -l < "$done_keys") completed runs"

# ---------- sweep ----------
for k in "${RISK_K_LIST[@]}"; do
  for probe in "${RISK_PROBE_LIST[@]}"; do
    for decay_type in "${RISK_DECAY_TYPE_LIST[@]}"; do
      for alpha in "${RISK_ALPHA_LIST[@]}"; do
        for consider_current in "${CONSIDER_CURRENT_LAYER_LIST[@]}"; do

          # exponential 类型使用 risk_decay，linear 类型使用 risk_decay_beta
          if [[ "$decay_type" == "exponential" ]]; then
            decay_values=("${RISK_DECAY_LIST[@]}")
            decay_beta_value=0.01  # 默认值，不使用
          else
            decay_values=(0.5)  # 默认值，不使用
            decay_beta_values=("${RISK_DECAY_BETA_LIST[@]}")
          fi

          # 根据 decay_type 选择遍历
          if [[ "$decay_type" == "exponential" ]]; then
            for decay in "${decay_values[@]}"; do
              decay_beta=0.01

              key="$k|$probe|$decay|$decay_type|$decay_beta|$alpha|$consider_current"
              if grep -qxF "$key" "$done_keys"; then
                echo "[SKIP] $key"
                continue
              fi

              echo
              echo "[RUN ] k=$k probe=$probe decay=$decay decay_type=$decay_type decay_beta=$decay_beta alpha=$alpha consider_current=$consider_current"
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
                --consider_current_layer_in_risk "$consider_current" \
                > "$tmp_out" 2>&1
              exit_code=$?
              set -e

              end_ts=$(date +%s)
              elapsed=$((end_ts - start_ts))

              ppl=$(awk '/ppl on wikitext/{x=$NF} END{if(x!="") print x; else print "NaN"}' "$tmp_out")

              echo "[TIME] ${elapsed}s = $(awk "BEGIN{printf \"%.2f\", $elapsed/60}") min"

              echo "$k,$probe,$decay,$decay_type,$decay_beta,$alpha,$consider_current,$ppl,$elapsed,$exit_code" >> "$OUT_CSV"
              echo "$key" >> "$done_keys"

              {
                echo "===== $key | ${elapsed}s | exit=$exit_code ====="
                cat "$tmp_out"
              } >> "$OUT_LOG"
            done
          else
            for decay_beta in "${decay_beta_values[@]}"; do
              decay=0.5

              key="$k|$probe|$decay|$decay_type|$decay_beta|$alpha|$consider_current"
              if grep -qxF "$key" "$done_keys"; then
                echo "[SKIP] $key"
                continue
              fi

              echo
              echo "[RUN ] k=$k probe=$probe decay=$decay decay_type=$decay_type decay_beta=$decay_beta alpha=$alpha consider_current=$consider_current"
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
                --consider_current_layer_in_risk "$consider_current" \
                > "$tmp_out" 2>&1
              exit_code=$?
              set -e

              end_ts=$(date +%s)
              elapsed=$((end_ts - start_ts))

              ppl=$(awk '/ppl on wikitext/{x=$NF} END{if(x!="") print x; else print "NaN"}' "$tmp_out")

              echo "[TIME] ${elapsed}s = $(awk "BEGIN{printf \"%.2f\", $elapsed/60}") min"

              echo "$k,$probe,$decay,$decay_type,$decay_beta,$alpha,$consider_current,$ppl,$elapsed,$exit_code" >> "$OUT_CSV"
              echo "$key" >> "$done_keys"

              {
                echo "===== $key | ${elapsed}s | exit=$exit_code ====="
                cat "$tmp_out"
              } >> "$OUT_LOG"
            done
          fi

        done
      done
    done
  done
done

echo "[DONE] sweep finished"
