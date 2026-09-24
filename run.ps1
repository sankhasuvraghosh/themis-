$env:THEMIS_LLM_BACKEND = "local"
$env:THEMIS_LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

if (-not (Test-Path "data\index\embeddings.npy")) {
    Write-Host "Index not found, building it first..."
    python -m src.ingest.build_index
    if (-not (Test-Path "data\index\embeddings.npy")) {
        Write-Host "Index build failed. Fix the error above and rerun."
        exit 1
    }
}

python -m streamlit run src/app/main.py