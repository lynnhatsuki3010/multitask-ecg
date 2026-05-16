Write-Host "========================================="
Write-Host "   STARTING FINAL GOLDEN PIPELINE RUNS   "
Write-Host "========================================="

Write-Host "`n[1/3] Running Seed 42..."
python scripts/02_train.py --config configs/experiments/final/final_e01_seed42.yaml

Write-Host "`n[2/3] Running Seed 123..."
python scripts/02_train.py --config configs/experiments/final/final_e02_seed123.yaml

Write-Host "`n[3/3] Running Seed 2024..."
python scripts/02_train.py --config configs/experiments/final/final_e03_seed2024.yaml

Write-Host "`n========================================="
Write-Host "   ALL FINAL RUNS COMPLETED SUCCESSFULLY "
Write-Host "========================================="
