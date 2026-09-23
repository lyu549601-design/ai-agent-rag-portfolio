$listener = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    Stop-Process -Id $listener.OwningProcess -Force
    Write-Host "网页控制台已停止。"
} else {
    Write-Host "网页控制台当前没有运行。"
}