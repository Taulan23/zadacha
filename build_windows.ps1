# PowerShell скрипт для сборки Windows EXE файла
Write-Host "Сборка Windows EXE файла для анализа спектров..." -ForegroundColor Green
Write-Host ""

# Проверка Python
try {
    $pythonVersion = python --version
    Write-Host "Найден Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Ошибка: Python не найден!" -ForegroundColor Red
    exit 1
}

# Установка PyInstaller
Write-Host "Установка PyInstaller..." -ForegroundColor Yellow
pip install pyinstaller

# Создание EXE файла
Write-Host "Сборка EXE файла..." -ForegroundColor Yellow
pyinstaller --onefile `
    --console `
    --name "SpectrumAnalyzer" `
    --add-data "данные;данные" `
    --add-data "requirements.txt;." `
    --add-data "README.md;." `
    --hidden-import numpy `
    --hidden-import pandas `
    --hidden-import matplotlib `
    --hidden-import scipy `
    --hidden-import openpyxl `
    --hidden-import xlsxwriter `
    --hidden-import psutil `
    --hidden-import pygame `
    main.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Сборка успешно завершена!" -ForegroundColor Green
    Write-Host "EXE файл находится в папке dist/SpectrumAnalyzer.exe" -ForegroundColor Green
    Write-Host ""
    Write-Host "Для запуска: dist/SpectrumAnalyzer.exe" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "Ошибка при сборке!" -ForegroundColor Red
}

Write-Host ""
Read-Host "Нажмите Enter для выхода"
