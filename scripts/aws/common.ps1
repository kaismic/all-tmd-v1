Set-StrictMode -Version Latest

function Initialize-AwsContext {
    param(
        [string]$Region = "ap-southeast-2",
        [string]$Profile = ""
    )
    $script:AllTmdAwsRegion = $Region
    $script:AllTmdAwsProfile = $Profile
    if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
        throw "AWS CLI was not found on PATH."
    }
}

function Invoke-AllTmdAws {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [switch]$AllowEmpty
    )
    $awsArguments = @($Arguments)
    if ($script:AllTmdAwsRegion) {
        $awsArguments += @("--region", $script:AllTmdAwsRegion)
    }
    if ($script:AllTmdAwsProfile) {
        $awsArguments += @("--profile", $script:AllTmdAwsProfile)
    }
    $output = & aws @awsArguments
    if ($LASTEXITCODE -ne 0) {
        throw "AWS CLI failed with exit code ${LASTEXITCODE}: aws $($Arguments -join ' ')"
    }
    if (-not $AllowEmpty -and $null -eq $output) {
        throw "AWS CLI returned no output: aws $($Arguments -join ' ')"
    }
    return $output
}

function Assert-AllTmdGitCommitAvailable {
    param(
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][string]$Commit
    )
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "Git was not found on PATH."
    }
    if (-not $Repository.Trim()) {
        throw "The run manifest does not specify a Git repository."
    }
    if ($Commit -notmatch '^[0-9a-f]{40}$') {
        throw "The run manifest contains an invalid Git commit."
    }

    $safeProjectRoot = $ProjectRoot.Replace('\', '/')
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        # A dry-run asks the remote for the exact object without changing the
        # local checkout or refs. This catches clean but unpushed commits.
        $ErrorActionPreference = "Continue"
        $fetchOutput = @(& git -c "safe.directory=$safeProjectRoot" `
            -C $ProjectRoot fetch --dry-run --depth 1 -- `
            $Repository $Commit 2>&1)
        $fetchExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($fetchExitCode -ne 0) {
        $detail = ($fetchOutput | Out-String).Trim()
        if ($detail) {
            Write-Warning $detail
        }
        throw "Git commit $Commit is not available from the run repository. Push it before preparing or starting an AWS run."
    }
}

function Get-AllTmdStackOutputs {
    param([Parameter(Mandatory = $true)][string]$StackName)
    $json = Invoke-AllTmdAws -Arguments @(
        "cloudformation", "describe-stacks",
        "--stack-name", $StackName,
        "--query", "Stacks[0].Outputs",
        "--output", "json"
    )
    $result = @{}
    foreach ($entry in (($json -join "`n") | ConvertFrom-Json)) {
        $result[$entry.OutputKey] = $entry.OutputValue
    }
    return $result
}

function Get-AllTmdEc2InstanceState {
    param([Parameter(Mandatory = $true)][string]$InstanceId)
    $state = Invoke-AllTmdAws -Arguments @(
        "ec2", "describe-instances",
        "--instance-ids", $InstanceId,
        "--query", "Reservations[0].Instances[0].State.Name",
        "--output", "text"
    )
    return ($state | Out-String).Trim()
}

function Wait-AllTmdSsmOnline {
    param(
        [Parameter(Mandatory = $true)][string]$InstanceId,
        [int]$TimeoutSeconds = 900
    )
    $deadline = [DateTimeOffset]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        $instanceState = Get-AllTmdEc2InstanceState -InstanceId $InstanceId
        if ($instanceState -in @("stopping", "stopped", "shutting-down", "terminated")) {
            throw "Instance $InstanceId entered EC2 state '$instanceState' while waiting for Systems Manager. It may have been stopped externally."
        }
        $status = Invoke-AllTmdAws -Arguments @(
            "ssm", "describe-instance-information",
            "--filters", "Key=InstanceIds,Values=$InstanceId",
            "--query", "InstanceInformationList[0].PingStatus",
            "--output", "text"
        ) -AllowEmpty
        if (($status | Out-String).Trim() -eq "Online") {
            return
        }
        Start-Sleep -Seconds 10
    } while ([DateTimeOffset]::UtcNow -lt $deadline)
    throw "Instance $InstanceId did not become available in Systems Manager."
}

function Send-AllTmdSsmCommand {
    param(
        [Parameter(Mandatory = $true)][string]$InstanceId,
        [Parameter(Mandatory = $true)][string[]]$Commands,
        [string]$Comment = "All-TMD operation"
    )
    $parameterFile = New-TemporaryFile
    try {
        @{ commands = $Commands } |
            ConvertTo-Json -Depth 4 |
            Set-Content -LiteralPath $parameterFile -Encoding utf8NoBOM
        $commandId = Invoke-AllTmdAws -Arguments @(
            "ssm", "send-command",
            "--instance-ids", $InstanceId,
            "--document-name", "AWS-RunShellScript",
            "--comment", $Comment,
            "--parameters", "file://$parameterFile",
            "--query", "Command.CommandId",
            "--output", "text"
        )
        return ($commandId | Out-String).Trim()
    }
    finally {
        Remove-Item -LiteralPath $parameterFile -Force -ErrorAction SilentlyContinue
    }
}

function Wait-AllTmdSsmCommand {
    param(
        [Parameter(Mandatory = $true)][string]$CommandId,
        [Parameter(Mandatory = $true)][string]$InstanceId,
        [int]$TimeoutSeconds = 600
    )
    $deadline = [DateTimeOffset]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        $raw = Invoke-AllTmdAws -Arguments @(
            "ssm", "get-command-invocation",
            "--command-id", $CommandId,
            "--instance-id", $InstanceId,
            "--output", "json"
        ) -AllowEmpty
        if ($raw) {
            $invocation = ($raw -join "`n") | ConvertFrom-Json
            if ($invocation.Status -in @("Success", "Failed", "TimedOut", "Cancelled")) {
                if ($invocation.StandardOutputContent) {
                    Write-Host $invocation.StandardOutputContent.TrimEnd()
                }
                if ($invocation.StandardErrorContent) {
                    Write-Warning $invocation.StandardErrorContent.TrimEnd()
                }
                if ($invocation.Status -ne "Success") {
                    throw "SSM command $CommandId finished with status $($invocation.Status)."
                }
                return $invocation
            }
        }
        Start-Sleep -Seconds 5
    } while ([DateTimeOffset]::UtcNow -lt $deadline)
    throw "SSM command $CommandId did not finish before the timeout."
}
