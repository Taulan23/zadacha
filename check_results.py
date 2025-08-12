"""
Скрипт для проверки содержимого Excel файла с результатами анализа
"""

import pandas as pd
import numpy as np

def check_excel_file(filename: str):
    """Проверка содержимого Excel файла"""
    print(f"Проверка файла: {filename}")
    print("=" * 60)
    
    try:
        # Читаем Excel файл
        excel_file = pd.ExcelFile(filename)
        
        print(f"Листы в файле: {excel_file.sheet_names}")
        print()
        
        # Проверяем каждый лист
        for sheet_name in excel_file.sheet_names:
            print(f"Лист: {sheet_name}")
            print("-" * 40)
            
            df = pd.read_excel(filename, sheet_name=sheet_name)
            
            print(f"Размерность: {df.shape}")
            print(f"Колонки: {list(df.columns)}")
            
            # Проверяем на отрицательные значения
            if df.select_dtypes(include=[np.number]).shape[1] > 0:
                numeric_cols = df.select_dtypes(include=[np.number])
                negative_count = (numeric_cols < 0).sum().sum()
                print(f"Отрицательных значений: {negative_count}")
                
                if negative_count > 0:
                    print("ВНИМАНИЕ: Обнаружены отрицательные значения!")
                    for col in numeric_cols.columns:
                        neg_vals = numeric_cols[col][numeric_cols[col] < 0]
                        if len(neg_vals) > 0:
                            print(f"  {col}: {len(neg_vals)} отрицательных значений")
            
            # Показываем первые несколько строк
            print("Первые 3 строки:")
            print(df.head(3))
            print()
            
            # Проверяем на NaN значения
            nan_count = df.isna().sum().sum()
            print(f"NaN значений: {nan_count}")
            print()
    
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")

if __name__ == "__main__":
    # Проверяем самый свежий файл
    check_excel_file("results/final_analysis_20250812_113243.xlsx")
