param(
    [Parameter(Mandatory = $true)]
    [string]$BudgetEmail,
    [string]$StackName = "all-tmd-v1-worker",
    [string]$Region = "ap-southeast-2",
    [string]$Profile = "",
    [string]$InstanceType = "c7i.4xlarge",
    [int]$DataVolumeSizeGiB = 200,
    [string]$BucketName = "",
    [string]$CollectorStackName = "transport-data-collector",
    [string]$NtfyTokenParameterName = "/all-tmd-v1/ntfy-token",
    [string]$ConsumerProjectName = "all-tmd-v2",
    [switch]$LeaveRunning,
    [switch]$NoExecuteChangeSet
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $PSScriptRoot "common.ps1")
Initialize-AwsContext -Region $Region -Profile $Profile
$stackSummary = Invoke-AllTmdAws -Arguments @(
    "cloudformation", "list-stacks",
    "--query", "StackSummaries[?StackName=='$StackName' && StackStatus!='DELETE_COMPLETE'] | [0].[StackId,StackStatus]",
    "--output", "text"
) -AllowEmpty
$stackSummaryText = ($stackSummary | Out-String).Trim()
$existingWorkerImageId = ""
if ($stackSummaryText -and $stackSummaryText -ne "None") {
    $stackFields = $stackSummaryText -split "\s+"
    $stackStatus = $stackFields[-1]
    if ($stackStatus -eq "UPDATE_ROLLBACK_FAILED") {
        throw "Stack $StackName is in UPDATE_ROLLBACK_FAILED. Run 'aws cloudformation continue-update-rollback --stack-name $StackName --region $Region', wait for UPDATE_ROLLBACK_COMPLETE, then retry."
    }
    if ($stackStatus -match '_IN_PROGRESS$') {
        throw "Stack $StackName is currently $stackStatus. Wait for that operation to finish before deploying."
    }
    $existingOutputs = Get-AllTmdStackOutputs -StackName $StackName
    $existingWorkerImageId = Invoke-AllTmdAws -Arguments @(
        "ec2", "describe-instances",
        "--instance-ids", $existingOutputs.InstanceId,
        "--query", "Reservations[0].Instances[0].ImageId",
        "--output", "text"
    )
    $existingWorkerImageId = ($existingWorkerImageId | Out-String).Trim()
    if ($existingWorkerImageId -notmatch '^ami-[0-9a-f]+$') {
        throw "Could not resolve the current AMI for worker $($existingOutputs.InstanceId); refusing an update that might replace it."
    }
    Write-Host "Pinning existing worker $($existingOutputs.InstanceId) to AMI $existingWorkerImageId for this stack update."
}
$collectorOutputs = Get-AllTmdStackOutputs -StackName $CollectorStackName
if (-not $collectorOutputs.SessionsBucketName -or -not $collectorOutputs.SessionsTableName) {
    throw "Collector stack $CollectorStackName does not expose SessionsBucketName and SessionsTableName."
}

$template = Join-Path $projectRoot "aws\cloudformation.yaml"
& python -c `
    "import sys; from pathlib import Path; import yaml; yaml.compose(Path(sys.argv[1]).read_text(encoding='utf-8'))" `
    $template
if ($LASTEXITCODE -ne 0) {
    throw "CloudFormation template is not well-formed YAML: $template"
}
Invoke-AllTmdAws -Arguments @(
    "cloudformation", "validate-template",
    "--template-body", "file://$template",
    "--output", "json"
) | Out-Null

$parameterOverrides = @(
    "InstanceType=$InstanceType",
    "DataVolumeSizeGiB=$DataVolumeSizeGiB",
    "BudgetNotificationEmail=$BudgetEmail",
    "NtfyTokenParameterName=$NtfyTokenParameterName"
    "ConsumerProjectName=$ConsumerProjectName"
    "CollectorSessionsBucketName=$($collectorOutputs.SessionsBucketName)"
    "CollectorSessionsTableName=$($collectorOutputs.SessionsTableName)"
)
if ($BucketName) {
    $parameterOverrides += "BucketName=$BucketName"
}
if ($existingWorkerImageId) {
    $parameterOverrides += "WorkerImageId=$existingWorkerImageId"
}

$deployArguments = @(
    "cloudformation", "deploy",
    "--stack-name", $StackName,
    "--template-file", $template,
    "--capabilities", "CAPABILITY_IAM",
    "--parameter-overrides"
) + $parameterOverrides
if ($NoExecuteChangeSet) {
    $deployArguments += "--no-execute-changeset"
}
Invoke-AllTmdAws -Arguments $deployArguments -AllowEmpty
if ($NoExecuteChangeSet) {
    Write-Host "Created a non-executed CloudFormation change set for $StackName. Review it in CloudFormation, then rerun without -NoExecuteChangeSet to apply it."
    return
}

$outputs = Get-AllTmdStackOutputs -StackName $StackName
$instanceId = $outputs.InstanceId
$instanceState = Get-AllTmdEc2InstanceState -InstanceId $instanceId
$startedWorker = $false
if ($instanceState -eq "stopping") {
    Invoke-AllTmdAws -Arguments @(
        "ec2", "wait", "instance-stopped", "--instance-ids", $instanceId
    ) -AllowEmpty
    $instanceState = "stopped"
}
if ($instanceState -eq "stopped") {
    Invoke-AllTmdAws -Arguments @(
        "ec2", "start-instances", "--instance-ids", $instanceId,
        "--output", "json"
    ) | Out-Null
    $startedWorker = $true
}
elseif ($instanceState -in @("shutting-down", "terminated")) {
    throw "Worker $instanceId cannot be validated because it is $instanceState."
}
Wait-AllTmdSsmOnline -InstanceId $instanceId

$ready = $false
for ($attempt = 0; $attempt -lt 60 -and -not $ready; $attempt++) {
    try {
        $commandId = Send-AllTmdSsmCommand -InstanceId $instanceId -Commands @(
            "test -f /var/lib/all-tmd-v1/bootstrap-complete",
            "docker compose version",
            "mountpoint -q /mnt/all-tmd-data"
        ) -Comment "Validate All-TMD worker bootstrap"
        Wait-AllTmdSsmCommand -CommandId $commandId -InstanceId $instanceId |
            Out-Null
        $ready = $true
    }
    catch {
        if ($attempt -eq 59) { throw }
        Start-Sleep -Seconds 10
    }
}

if ($startedWorker -and -not $LeaveRunning) {
    Invoke-AllTmdAws -Arguments @(
        "ec2", "stop-instances", "--instance-ids", $instanceId,
        "--output", "json"
    ) | Out-Null
    Write-Host "Worker $instanceId is initialized and stopping to avoid idle compute charges."
}
else {
    Write-Host "Worker $instanceId is initialized and remains running because it was already running or -LeaveRunning was supplied."
}
Write-Host "S3 bucket: $($outputs.BucketName)"
Write-Host "Data volume: $($outputs.DataVolumeId)"
