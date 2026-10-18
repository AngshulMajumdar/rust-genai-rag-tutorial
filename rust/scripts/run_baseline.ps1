# Phase-1 baseline grid on the 100-question stratified subset. Run from the rust\ folder after `index` and `embed`.
$ErrorActionPreference = "Stop"
$B = ".\target\release\hr-rag.exe"
$Every = if ($env:EVERY) { $env:EVERY } else { 10 }
New-Item -ItemType Directory -Force runs | Out-Null
function Run($name, [string[]]$flags) {
    & $B eval --every $Every --out "runs\$name.jsonl" @flags
    & $B score "runs\$name.jsonl" --report "runs\$name.md" | Out-Null
}
Run "gemma3-4b_simple_bm25"    @("--model","gemma3:4b","--strategy","simple","--mode","bm25")
Run "gemma3-4b_simple_hybrid"  @("--model","gemma3:4b","--strategy","simple","--mode","hybrid")
Run "gemma3-4b_agentic_hybrid" @("--model","gemma3:4b","--strategy","agentic","--mode","hybrid")
Run "gemma3-1b_agentic_hybrid" @("--model","gemma3:1b","--strategy","agentic","--mode","hybrid")
Get-ChildItem runs\*.md | ForEach-Object { "## $($_.Name)"; Select-String -Path $_ -Pattern '^\| (ALL|Cross-language|group|---)' | ForEach-Object { $_.Line } }
