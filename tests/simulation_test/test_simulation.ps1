$packageName = "MRzeroCore"

$mainBranch = "main_branch"

function CleanFolder {
    param (
    )
    Remove-Item -Recurse -Force $mainBranch -ErrorAction SilentlyContinue
    Remove-Item -Path "tests\simulation_test\ref_files" -Recurse -Force
    Remove-Item -Path "tests\simulation_test\actual_files" -Recurse -Force
    Remove-Item -Path "tests\simulation_test\seq_files" -Recurse -Force
    if (Test-Path "numerical_brain_cropped.mat") {
        Remove-Item "numerical_brain_cropped.mat" -Force
    }
}

# Register cleanup for Ctrl+C
$null = Register-EngineEvent PowerShell.Exiting -Action { CleanFolder }

Write-Host "Cleaning up old directories..."
Remove-Item -Recurse -Force $mainBranch -ErrorAction SilentlyContinue
mkdir "tests\simulation_test\ref_files" -Force
mkdir "tests\simulation_test\actual_files" -Force
mkdir "tests\simulation_test\seq_files" -Force

Write-Host "=========================== Generate sequence files ==========================="
if ($args.Count -gt 0) {
    # Check if the specified notebook exists
    $notebookPath = $args[0]
    if (-not (Test-Path $notebookPath)) {
        Write-Host "=========================== Failed ==========================="
        Write-Host "Notebook not found: $notebookPath"
        CleanFolder
        exit 1
    }
    python tests\simulation_test\generate_seq_files.py @args
} else {
    python tests\simulation_test\generate_seq_files.py
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "=========================== Failed ==========================="
    Write-Host "Failed to generate sequence files. Exit code: $LASTEXITCODE"
    CleanFolder
    exit
} else {
    Write-Host "All sequence files generated successfully."
}

Write-Host "=========================== Cloning main branch ==========================="
git clone https://github.com/MRsources/MRzero-Core.git $mainBranch

Write-Host "=========================== Installing main branch version ==========================="
Set-Location $mainBranch
pip uninstall -y $packageName
pip install -e .
Set-Location ..

Write-Host "=========================== Generate reference data ==========================="
python tests\simulation_test\generate_ref_files.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "=========================== Failed ==========================="
    Write-Host "Failed to generate reference data. Exit code: $LASTEXITCODE"
    CleanFolder
    exit
} else {
    Write-Host "All reference files generated successfully."
}

Write-Host "=========================== Uninstall main branch version and install current version ==========================="
pip uninstall -y $packageName
pip install -e .

Write-Host "=========================== Generate actual data ==========================="
python tests\simulation_test\generate_actual_files.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "=========================== Failed ==========================="
    Write-Host "Failed to generate actual data. Exit code: $LASTEXITCODE"
    CleanFolder
    exit
} else {
    Write-Host "All actual files generated successfully."
}

Write-Host "=========================== Run tests ==========================="
python tests\simulation_test\test_simulation.py

Write-Host "Cleaning ..."
CleanFolder
