# Инструкция по сборке Windows EXE файла

## Требования для сборки на Windows

1. **Python 3.8+** - скачать с [python.org](https://www.python.org/downloads/)
2. **Git** - для клонирования репозитория
3. **Достаточно места на диске** - минимум 2GB свободного места

## Быстрая сборка (рекомендуется)

### Вариант 1: Использование batch файла
1. Откройте командную строку (cmd) от имени администратора
2. Перейдите в папку проекта
3. Запустите:
```cmd
build_windows.bat
```

### Вариант 2: Использование PowerShell
1. Откройте PowerShell от имени администратора
2. Перейдите в папку проекта
3. Запустите:
```powershell
.\build_windows.ps1
```

## Ручная сборка

### Шаг 1: Установка зависимостей
```cmd
pip install -r requirements.txt
pip install pyinstaller
```

### Шаг 2: Сборка EXE файла
```cmd
pyinstaller --onefile --console --name "SpectrumAnalyzer" --add-data "данные;данные" --add-data "requirements.txt;." --add-data "README.md;." main.py
```

### Шаг 3: Проверка результата
После сборки EXE файл будет находиться в папке `dist/SpectrumAnalyzer.exe`

## Запуск программы

1. Скопируйте файл `SpectrumAnalyzer.exe` из папки `dist/`
2. Скопируйте папку `данные/` рядом с EXE файлом
3. Запустите `SpectrumAnalyzer.exe`

## Структура файлов для распространения

```
SpectrumAnalyzer/
├── SpectrumAnalyzer.exe
├── данные/
│   ├── DN 8-Group spectra.xlsx
│   └── DN Integral spectra -short and long irradiation.xlsx
├── requirements.txt
└── README.md
```

## Возможные проблемы и решения

### Проблема: "Python не найден"
**Решение:** Добавьте Python в PATH или используйте полный путь к python.exe

### Проблема: "Не удается найти модуль"
**Решение:** Убедитесь, что все зависимости установлены:
```cmd
pip install numpy pandas matplotlib scipy openpyxl xlsxwriter psutil pygame
```

### Проблема: "Access denied"
**Решение:** Запустите командную строку от имени администратора

### Проблема: Большой размер EXE файла
**Решение:** Это нормально, так как PyInstaller включает все зависимости

## Оптимизация размера (опционально)

Для уменьшения размера EXE файла можно исключить неиспользуемые модули:

```cmd
pyinstaller --onefile --console --name "SpectrumAnalyzer" --exclude-module matplotlib.tests --exclude-module numpy.tests --exclude-module scipy.tests main.py
```

## Проверка работоспособности

После сборки протестируйте EXE файл:
1. Запустите `SpectrumAnalyzer.exe`
2. Программа должна начать анализ данных
3. Результаты сохранятся в папке `results/`

## Распространение

Для распространения программы:
1. Создайте архив с EXE файлом и папкой `данные/`
2. Пользователям нужно только распаковать архив и запустить EXE файл
3. Python не требуется на компьютере пользователя
