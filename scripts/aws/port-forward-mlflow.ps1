param(
    [string]$StackName = "all-tmd-v1-worker",
    [string]$Region = "ap-southeast-2",
    [string]$Profile = "",
    [int]$LocalPort = 5002
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
Initialize-AwsContext -Region $Region -Profile $Profile
$outputs = Get-AllTmdStackOutputs -StackName $StackName
$parametersFile = New-TemporaryFile
try {
    @{ portNumber = @("5002"); localPortNumber = @([string]$LocalPort) } |
        ConvertTo-Json -Compress |
        Set-Content -LiteralPath $parametersFile -Encoding utf8NoBOM
    $arguments = @(
        "ssm", "start-session",
        "--target", $outputs.InstanceId,
        "--document-name", "AWS-StartPortForwardingSession",
        "--parameters", "file://$parametersFile",
        "--region", $Region
    )
    if ($Profile) { $arguments += @("--profile", $Profile) }
    Write-Host "Forwarding http://localhost:$LocalPort to worker MLflow. Press Ctrl+C to stop."
    & aws @arguments
    if ($LASTEXITCODE -ne 0) { throw "Session Manager port forwarding failed." }
}
finally {
    Remove-Item -LiteralPath $parametersFile -Force -ErrorAction SilentlyContinue
}
