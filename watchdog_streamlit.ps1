$python  = "C:\Users\yaniv\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$script  = "C:\Users\yaniv\קלוד\trading-agent\signal_agent.py"
$port    = 8501
$logFile = "C:\Users\yaniv\קלוד\trading-agent\watchdog.log"

function Write-Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    Add-Content -Path $logFile -Value $line
}

function Is-Alive {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port" -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
        return ($r.StatusCode -eq 200)
    } catch { return $false }
}

function Start-Streamlit {
    # Kill anything on the port
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2

    $p = Start-Process `
        -FilePath $python `
        -ArgumentList @("-m", "streamlit", "run", $script, "--server.port=$port") `
        -WorkingDirectory "C:\Users\yaniv\קלוד\trading-agent" `
        -PassThru -WindowStyle Hidden
    Write-Log "Started PID $($p.Id)"
    return $p
}

Write-Log "Watchdog starting"
$proc = Start-Streamlit

while ($true) {
    Start-Sleep -Seconds 20

    if ($proc.HasExited -or -not (Is-Alive)) {
        Write-Log "Streamlit down — restarting"
        $proc = Start-Streamlit
    }
}
