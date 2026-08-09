param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [string]$StackName = "all-tmd-v1-worker",
    [string]$Region = "ap-southeast-2",
    [string]$Profile = ""
)

$ErrorActionPreference = "Stop"
if ($RunId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$') {
    throw "RunId contains unsupported characters."
}
. (Join-Path $PSScriptRoot "common.ps1")
Initialize-AwsContext -Region $Region -Profile $Profile
$outputs = Get-AllTmdStackOutputs -StackName $StackName
$instanceId = $outputs.InstanceId
$bucket = $outputs.BucketName

Invoke-AllTmdAws -Arguments @(
    "s3api", "head-object",
    "--bucket", $bucket,
    "--key", "all-tmd-v1/config/$RunId/run-manifest.json",
    "--output", "json"
) | Out-Null
$state = Get-AllTmdEc2InstanceState -InstanceId $instanceId
if ($state -eq "stopped") {
    Invoke-AllTmdAws -Arguments @(
        "ec2", "start-instances", "--instance-ids", $instanceId, "--output", "json"
    ) | Out-Null
}
elseif ($state -in @("stopping", "shutting-down", "terminated")) {
    throw "Instance $instanceId cannot start a run while it is '$state'."
}
Wait-AllTmdSsmOnline -InstanceId $instanceId

$runnerUri = "s3://$bucket/all-tmd-v1/config/$RunId/run-trials-cloud.sh"
$commandId = Send-AllTmdSsmCommand -InstanceId $instanceId -Commands @(
    # AWS-RunShellScript executes this wrapper with /bin/sh, not Bash.
    "set -eu",
    "aws s3 cp '$runnerUri' /tmp/run-trials-cloud.sh --only-show-errors",
    "chmod 0755 /tmp/run-trials-cloud.sh",
    "/tmp/run-trials-cloud.sh install --bucket '$bucket' --run-id '$RunId'"
) -Comment "Start All-TMD run $RunId"
Wait-AllTmdSsmCommand -CommandId $commandId -InstanceId $instanceId | Out-Null
Write-Host "Run $RunId is active on $instanceId and will continue after disconnect."
Write-Host "Inspect it with: .\scripts\aws\status.ps1 -RunId $RunId"
