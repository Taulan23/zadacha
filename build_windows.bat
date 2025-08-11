@echo off
echo Сборка Windows EXE файла для анализа спектров...
echo.

REM Установка зависимостей
echo Установка PyInstaller...
pip install pyinstaller

echo.
echo Сборка EXE файла...
pyinstaller --onefile --console --name "SpectrumAnalyzer" --add-data "данные;данные" --add-data "requirements.txt;." --add-data "README.md;." main.py

echo.
echo Сборка завершена!
echo EXE файл находится в папке dist/
echo.
pause
