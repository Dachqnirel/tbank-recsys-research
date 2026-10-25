#!/usr/bin/env bash
# Полный прогон эксперимента. Запуск из корня репозитория: bash run_all.sh
set -e
mkdir -p results
python src/01_prepare_sample.py | tee results/01_prepare_output.txt
python src/02_check_leakage.py  | tee results/02_leakage_output.txt
python src/03_train_manual.py   | tee results/03_manual_output.txt
# Шаги 1-3 (сетка, auto_all, перестановочная важность). На Python 3.14.8 + CatBoost 1.2.10
# дальнейший перебор K в одном процессе падал с segfault, поэтому он вынесен в 04b/04c.
python src/04_auto_features.py  | tee results/04_auto_output.txt || true
rm -f results/auto_k_sweep_runs.csv
for K in 10 20 40 90; do python src/04b_k_run.py --k $K; done
for S in 0 1 2; do python src/04c_final_selected.py --seed $S; done
python src/04d_breakdown.py     | tee results/04d_breakdown_output.txt
python src/05_report.py         | tee results/05_report_output.txt
# python src/06_scaling.py      | tee results/06_scaling_output.txt   # полный замер не выполнен
