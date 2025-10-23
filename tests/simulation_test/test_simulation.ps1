$packageName = "MRzeroCore"

$mainBranch = "main_branch"

function CleanFolder {
    param (
    )
    Remove-Item -Recurse -Force $mainBranch -ErrorAction SilentlyContinue
    Remove-Item -Path "tests\simulation_test\ref_files" -Recurse -Force
    Remove-Item -Path "tests\simulation_test\actual_files" -Recurse -Force
    Remove-Item numerical_brain_cropped.mat -Force
}

Write-Host "Cleaning up old directories..."
Remove-Item -Recurse -Force $mainBranch -ErrorAction SilentlyContinue
mkdir "tests\simulation_test\ref_files" -Force
mkdir "tests\simulation_test\actual_files" -Force

Write-Host "=========================== Cloning main branch ==========================="
git clone https://github.com/MRsources/MRzero-Core.git $mainBranch

Write-Host "=========================== Installing main branch version ==========================="
Set-Location $mainBranch
pip uninstall -y $packageName
pip install -e .

Write-Host "=========================== Generate reference data ==========================="
Set-Location ..
if ($args.Count -gt 0) {
    python tests\simulation_test\generate_ref_files.py @args
} else {
    python tests\simulation_test\generate_ref_files.py
}
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
if ($args.Count -gt 0) {
    python tests\simulation_test\generate_actual_files.py @args
} else {
    python tests\simulation_test\generate_actual_files.py
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "=========================== Failed ==========================="
    Write-Host "Failed to generate actual data. Exit code: $LASTEXITCODE"
    CleanFolder
    exit
} else {
    Write-Host "All actual files generated successfully."
}

Write-Host "=========================== Run tests ==========================="
if ($args.Count -gt 0) {
    python tests\simulation_test\test_simulation.py @args
} else {
    python tests\simulation_test\test_simulation.py
}

Write-Host "Cleaning ..."
CleanFolder
