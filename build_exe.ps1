$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
    python -m pip install pyinstaller
}

python -m pip install -r requirements.txt
python -m pip install pystray

$tools = Join-Path $root "tools"
New-Item -ItemType Directory -Force $tools | Out-Null

function Install-ZippedTool {
    param(
        [string]$OutputName,
        [string]$Url,
        [string]$Sha256,
        [string]$ExecutableName
    )
    $output = Join-Path $tools $OutputName
    if (Test-Path $output) {
        return
    }
    $archive = Join-Path $tools "$OutputName.zip"
    $extract = Join-Path $tools "$OutputName.extract"
    Invoke-WebRequest -Uri $Url -OutFile $archive
    $actual = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash
    if ($actual -ne $Sha256) {
        throw "SHA256 mismatch for $OutputName`: expected $Sha256, got $actual"
    }
    Expand-Archive -LiteralPath $archive -DestinationPath $extract -Force
    $candidate = Get-ChildItem -LiteralPath $extract -Recurse -Filter $ExecutableName | Select-Object -First 1
    if (-not $candidate) {
        throw "$ExecutableName not found in $archive"
    }
    Copy-Item -LiteralPath $candidate.FullName -Destination $output -Force
    Remove-Item -LiteralPath $archive -Force
    Remove-Item -LiteralPath $extract -Recurse -Force
}

Install-ZippedTool `
    -OutputName "aria2c.exe" `
    -Url "https://github.com/aria2/aria2/releases/download/release-1.37.0/aria2-1.37.0-win-64bit-build1.zip" `
    -Sha256 "67D015301EEF0B612191212D564C5BB0A14B5B9C4796B76454276A4D28D9B288" `
    -ExecutableName "aria2c.exe"

Install-ZippedTool `
    -OutputName "syncthing.exe" `
    -Url "https://github.com/syncthing/syncthing/releases/download/v2.1.0/syncthing-windows-amd64-v2.1.0.zip" `
    -Sha256 "33DA7C8371F4A70DCF7E5F9136D71DBF5EA280D06BB99DB0D1E979B14C324DEB" `
    -ExecutableName "syncthing.exe"

$runtimeSecrets = Join-Path $root "secrets.runtime.json"
if (-not (Test-Path $runtimeSecrets)) {
    if (-not $env:KRAKEN_API_KEY -or -not $env:KRAKEN_API_SECRET) {
        throw "Set KRAKEN_API_KEY and KRAKEN_API_SECRET env vars or create secrets.runtime.json before building."
    }
    @{
        kraken = @{
            api_key = $env:KRAKEN_API_KEY
            api_secret = $env:KRAKEN_API_SECRET
        }
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $runtimeSecrets -Encoding UTF8
}

Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue

python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --name MarketRecorder `
    --add-data "config.json;." `
    --add-data "secrets.runtime.json;." `
    --add-binary "tools\aria2c.exe;." `
    --add-binary "tools\syncthing.exe;." `
    --hidden-import pystray._win32 `
    --exclude-module pandas `
    --exclude-module matplotlib `
    --exclude-module scipy `
    --exclude-module sklearn `
    --exclude-module torch `
    --exclude-module numba `
    --exclude-module llvmlite `
    --exclude-module sympy `
    --exclude-module pytest `
    --exclude-module openpyxl `
    --exclude-module sqlalchemy `
    --exclude-module transformers `
    desktop_recorder.py

Write-Host "Built: $root\dist\MarketRecorder.exe"
