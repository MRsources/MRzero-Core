$packageName = "MRzeroCore"
$venvName = ".test_venv"
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

# Create virtual environment if it doesn't exist
if (-not (Test-Path "$venvName\Scripts\Activate.ps1")) {
    Write-Host "=========================== Creating virtual environment ==========================="
    python -m venv $venvName
    if ($LASTEXITCODE -ne 0) {
        Write-Host "=========================== Failed ==========================="
        Write-Host "Failed to create virtual environment. Exit code: $LASTEXITCODE"
        CleanFolder
        exit 1
    }
    
    # Activate virtual environment
    & "$venvName\Scripts\Activate.ps1"
    Write-Host "Environment activated."
    
    Write-Host "=========================== Installing dependencies in virtual environment ==========================="
    pip install pypulseq torchkbnufft nbformat --quiet
    Write-Host "Dependencies installed."
} else {
    Write-Host "=========================== Using existing virtual environment ==========================="
    # Activate existing virtual environment
    & "$venvName\Scripts\Activate.ps1"
    Write-Host "Environment activated."
}

Write-Host "=========================== Generate sequence files ==========================="
if ($args.Count -gt 0) {
    # Check if the specified notebook exists
    $notebookPath = $args[0]
    if (-not (Test-Path $notebookPath)) {
        Write-Host "=========================== Failed ==========================="
        Write-Host "Notebook not found: $notebookPath"
        deactivate
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
    deactivate
    CleanFolder
    exit
} else {
    Write-Host "All sequence files generated successfully."
}

Write-Host "=========================== Cloning main branch ==========================="
git clone https://github.com/MRsources/MRzero-Core.git $mainBranch

Write-Host "=========================== Installing main branch version ==========================="
Set-Location $mainBranch
pip install -e . --quiet
Set-Location ..
Write-Host "Main branch version installed."

Write-Host "=========================== Generate reference data ==========================="
python tests\simulation_test\generate_ref_files.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "=========================== Failed ==========================="
    Write-Host "Failed to generate reference data. Exit code: $LASTEXITCODE"
    deactivate
    CleanFolder
    exit
} else {
    Write-Host "All reference files generated successfully."
}

Write-Host "=========================== Installing current branch version ==========================="
pip install -e . --quiet
Write-Host "Current branch version installed."

Write-Host "=========================== Generate actual data ==========================="
python tests\simulation_test\generate_actual_files.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "=========================== Failed ==========================="
    Write-Host "Failed to generate actual data. Exit code: $LASTEXITCODE"
    deactivate
    CleanFolder
    exit
} else {
    Write-Host "All actual files generated successfully."
}

Write-Host "=========================== Run tests ==========================="
python tests\simulation_test\test_simulation.py

Write-Host "Cleaning ..."
# Deactivate virtual environment before cleanup
deactivate
CleanFolder
