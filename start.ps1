# Quick Start Script for AI Financial Intelligence Platform
# Run this script to install dependencies and launch the application

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AI Financial Intelligence Platform" -ForegroundColor Cyan
Write-Host "Quick Start Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = py -3.11 --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python 3.11 not found! Please install Python 3.11" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    py -3.11 -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
}

Write-Host ""

# Install dependencies
Write-Host "Installing dependencies (this may take a few minutes)..." -ForegroundColor Yellow
.\venv\Scripts\python.exe -m pip install --upgrade pip -q
.\venv\Scripts\python.exe -m pip install -r requirements.txt -q

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Error installing dependencies" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting Streamlit application..." -ForegroundColor Yellow
Write-Host ""
Write-Host "The application will open in your browser automatically." -ForegroundColor Cyan
Write-Host "If it doesn't, navigate to: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the application" -ForegroundColor Yellow
Write-Host ""

# Launch Streamlit
.\venv\Scripts\python.exe -m streamlit run app.py
