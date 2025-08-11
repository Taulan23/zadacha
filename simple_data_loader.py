"""
Упрощенный загрузчик данных для работы с реальными Excel файлами
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging
import os

logger = logging.getLogger(__name__)

class SimpleDataLoader:
    """Упрощенный загрузчик данных"""
    
    def __init__(self, data_dir: str = "данные"):
        self.data_dir = data_dir
    
    def load_integral_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Загрузка интегральных данных
        
        Returns:
            Tuple: (long_data, short_data, energy_bins)
        """
        file_path = os.path.join(self.data_dir, "DN Integral spectra -short and long irradiation.xlsx")
        
        try:
            # Загружаем данные длинного облучения (120s)
            logger.info("Загрузка данных длинного облучения...")
            df_long = pd.read_excel(file_path, sheet_name='Spectra tirr=120s   ', header=1)
            
            # Загружаем данные короткого облучения (20s)
            logger.info("Загрузка данных короткого облучения...")
            df_short = pd.read_excel(file_path, sheet_name='Spectra tirr=20s ', header=1)
            
            # Извлекаем энергетические бины (первый столбец)
            energy_col = df_long.columns[0]
            energy_bins = df_long[energy_col].dropna().values
            
            # Извлекаем данные спектров (все столбцы кроме первого)
            long_columns = [col for col in df_long.columns[1:] if not pd.isna(col) and str(col).strip() != '']
            short_columns = [col for col in df_short.columns[1:] if not pd.isna(col) and str(col).strip() != '']
            
            logger.info(f"Найдено {len(long_columns)} столбцов длинного облучения")
            logger.info(f"Найдено {len(short_columns)} столбцов короткого облучения")
            
            # Создаем массивы данных
            long_data = df_long[long_columns].dropna().values.T
            short_data = df_short[short_columns].dropna().values.T
            
            # Проверяем размерности
            if long_data.shape[1] != len(energy_bins):
                logger.warning(f"Размерность данных длинного облучения не совпадает с энергетическими бинами")
                # Обрезаем до минимальной размерности
                min_bins = min(long_data.shape[1], len(energy_bins))
                long_data = long_data[:, :min_bins]
                energy_bins = energy_bins[:min_bins]
            
            if short_data.shape[1] != len(energy_bins):
                logger.warning(f"Размерность данных короткого облучения не совпадает с энергетическими бинами")
                # Обрезаем до минимальной размерности
                min_bins = min(short_data.shape[1], len(energy_bins))
                short_data = short_data[:, :min_bins]
                energy_bins = energy_bins[:min_bins]
            
            logger.info(f"Загружено {len(energy_bins)} энергетических бинов")
            logger.info(f"Данные длинного облучения: {long_data.shape}")
            logger.info(f"Данные короткого облучения: {short_data.shape}")
            
            return long_data, short_data, energy_bins
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке интегральных данных: {e}")
            # Создаем пустые данные в случае ошибки
            energy_bins = np.arange(10, 1610, 10)
            long_data = np.zeros((18, len(energy_bins)))
            short_data = np.zeros((18, len(energy_bins)))
            return long_data, short_data, energy_bins
    
    def load_group_spectra(self) -> Dict[str, np.ndarray]:
        """
        Загрузка групповых спектров
        
        Returns:
            Dict: групповые спектры
        """
        file_path = os.path.join(self.data_dir, "DN 8-Group spectra.xlsx")
        
        logger.info("Загрузка групповых спектров...")
        df = pd.read_excel(file_path, sheet_name='DN Group spectra', header=1)
        
        # Извлекаем энергетические бины
        energy_col = df.columns[0]
        energy_bins = df[energy_col].dropna().values
        
        # Извлекаем данные групп (столбцы с χ(En))
        group_spectra = {}
        
        # Ищем столбцы с данными групп (пропускаем заголовки)
        data_columns = []
        for col in df.columns:
            if 'χ(En)' in str(col) and 'n/10keV' in str(col):
                data_columns.append(col)
        
        logger.info(f"Найдено {len(data_columns)} столбцов с данными групп")
        
        # Берем первые 8 столбцов с данными
        for i in range(min(8, len(data_columns))):
            col = data_columns[i]
            try:
                # Преобразуем в числовой формат
                group_data = pd.to_numeric(df[col], errors='coerce').dropna().values
                group_spectra[f'group_{i+1}'] = group_data
                logger.info(f"Загружена группа {i+1}: {len(group_data)} точек")
            except Exception as e:
                logger.warning(f"Ошибка при загрузке группы {i+1}: {e}")
                group_spectra[f'group_{i+1}'] = np.zeros(160)  # Заполняем нулями
        
        return group_spectra
    
    def load_all_data(self) -> Dict:
        """
        Загрузка всех данных
        
        Returns:
            Dict: все данные
        """
        long_data, short_data, energy_bins = self.load_integral_data()
        group_spectra = self.load_group_spectra()
        
        return {
            'long_irradiation_data': long_data,
            'short_irradiation_data': short_data,
            'energy_bins': energy_bins,
            'group_spectra': group_spectra
        }
