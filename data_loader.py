"""
Модуль для загрузки реальных данных из Excel файлов
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
import os

logger = logging.getLogger(__name__)

class RealDataLoader:
    """Класс для загрузки реальных данных из Excel файлов"""
    
    def __init__(self, data_dir: str = "данные"):
        """
        Инициализация загрузчика данных
        
        Args:
            data_dir: директория с данными
        """
        self.data_dir = data_dir
        self.integral_file = "DN Integral spectra -short and long irradiation.xlsx"
        self.group_file = "DN 8-Group spectra.xlsx"
        
    def load_integral_spectra(self) -> Dict[str, np.ndarray]:
        """
        Загрузка интегральных спектров из Excel файла
        
        Returns:
            Dict: словарь с данными интегральных спектров
        """
        file_path = os.path.join(self.data_dir, self.integral_file)
        
        try:
            # Читаем все листы из файла
            excel_file = pd.ExcelFile(file_path)
            logger.info(f"Доступные листы: {excel_file.sheet_names}")
            
            data = {}
            
            for sheet_name in excel_file.sheet_names:
                logger.info(f"Загрузка листа: {sheet_name}")
                
                # Пропускаем служебные листы
                if 'Description' in sheet_name or 'Лист1' in sheet_name:
                    continue
                
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Ищем строку с заголовками (обычно вторая строка)
                header_row = None
                for i in range(min(5, len(df))):
                    row = df.iloc[i]
                    if 'En, keV' in str(row.values) or 'keV' in str(row.values):
                        header_row = i
                        break
                
                if header_row is None:
                    logger.warning(f"Не найдена строка с заголовками в листе {sheet_name}")
                    continue
                
                # Читаем данные с правильными заголовками
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
                
                # Ищем столбец с энергией
                energy_col = None
                for col in df.columns:
                    if 'keV' in str(col) or 'En' in str(col):
                        energy_col = col
                        break
                
                if energy_col is None:
                    logger.warning(f"Не найден столбец с энергией в листе {sheet_name}")
                    continue
                
                # Очищаем данные от NaN значений
                df = df.dropna(subset=[energy_col])
                
                # Извлекаем энергетические бины
                energy_bins = df[energy_col].values
                
                # Извлекаем данные спектров (все столбцы кроме энергии и служебных)
                spectrum_columns = []
                for col in df.columns:
                    if col != energy_col and not pd.isna(col) and str(col).strip() != '':
                        # Проверяем, что столбец содержит числовые данные
                        if df[col].dtype in ['float64', 'int64'] or df[col].dtype == 'object':
                            try:
                                pd.to_numeric(df[col], errors='coerce')
                                spectrum_columns.append(col)
                            except:
                                continue
                
                if len(spectrum_columns) == 0:
                    logger.warning(f"Не найдены данные спектров в листе {sheet_name}")
                    continue
                
                # Создаем массив данных
                spectra_data = df[spectrum_columns].values
                
                # Преобразуем в числовой формат
                spectra_data = pd.DataFrame(spectra_data, columns=spectrum_columns).apply(pd.to_numeric, errors='coerce').values
                
                data[sheet_name] = {
                    'energy_bins': energy_bins,
                    'spectra': spectra_data,
                    'column_names': spectrum_columns
                }
                
                logger.info(f"Загружено {len(energy_bins)} энергетических бинов и {len(spectrum_columns)} спектров из листа {sheet_name}")
            
            return data
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке интегральных спектров: {e}")
            return {}
    
    def load_group_spectra(self) -> Dict[str, np.ndarray]:
        """
        Загрузка групповых спектров из Excel файла
        
        Returns:
            Dict: словарь с данными групповых спектров
        """
        file_path = os.path.join(self.data_dir, self.group_file)
        
        try:
            # Читаем все листы из файла
            excel_file = pd.ExcelFile(file_path)
            logger.info(f"Доступные листы: {excel_file.sheet_names}")
            
            data = {}
            
            for sheet_name in excel_file.sheet_names:
                logger.info(f"Загрузка листа: {sheet_name}")
                
                # Пропускаем служебные листы
                if 'Description' in sheet_name or 'Relative abundances' in sheet_name:
                    continue
                
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Ищем строку с заголовками (обычно вторая строка)
                header_row = None
                for i in range(min(5, len(df))):
                    row = df.iloc[i]
                    if 'En, keV' in str(row.values) or 'keV' in str(row.values):
                        header_row = i
                        break
                
                if header_row is None:
                    logger.warning(f"Не найдена строка с заголовками в листе {sheet_name}")
                    continue
                
                # Читаем данные с правильными заголовками
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
                
                # Ищем столбец с энергией
                energy_col = None
                for col in df.columns:
                    if 'keV' in str(col) or 'En' in str(col):
                        energy_col = col
                        break
                
                if energy_col is None:
                    logger.warning(f"Не найден столбец с энергией в листе {sheet_name}")
                    continue
                
                # Очищаем данные от NaN значений
                df = df.dropna(subset=[energy_col])
                
                # Извлекаем энергетические бины
                energy_bins = df[energy_col].values
                
                # Извлекаем данные спектров (столбцы с χ(En))
                spectrum_columns = []
                for col in df.columns:
                    if col != energy_col and not pd.isna(col) and str(col).strip() != '':
                        if 'χ(En)' in str(col) or 'n/10keV' in str(col):
                            # Проверяем, что столбец содержит числовые данные
                            try:
                                pd.to_numeric(df[col], errors='coerce')
                                spectrum_columns.append(col)
                            except:
                                continue
                
                if len(spectrum_columns) == 0:
                    logger.warning(f"Не найдены данные спектров в листе {sheet_name}")
                    continue
                
                # Создаем массив данных
                spectra_data = df[spectrum_columns].values
                
                # Преобразуем в числовой формат
                spectra_data = pd.DataFrame(spectra_data, columns=spectrum_columns).apply(pd.to_numeric, errors='coerce').values
                
                data[sheet_name] = {
                    'energy_bins': energy_bins,
                    'spectra': spectra_data,
                    'column_names': spectrum_columns
                }
                
                logger.info(f"Загружено {len(energy_bins)} энергетических бинов и {len(spectrum_columns)} спектров из листа {sheet_name}")
            
            return data
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке групповых спектров: {e}")
            return {}
    
    def extract_experimental_data(self, integral_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Извлечение экспериментальных данных для анализа
        
        Args:
            integral_data: данные интегральных спектров
            
        Returns:
            Tuple: (long_irradiation_data, short_irradiation_data)
        """
        long_data = None
        short_data = None
        
        # Ищем данные длинного и короткого облучения
        for sheet_name, sheet_data in integral_data.items():
            sheet_name_lower = sheet_name.lower()
            
            if 'long' in sheet_name_lower or '120' in sheet_name_lower:
                logger.info(f"Найдены данные длинного облучения в листе: {sheet_name}")
                long_data = sheet_data['spectra'].T  # Транспонируем для соответствия формату
                
            elif 'short' in sheet_name_lower or '20' in sheet_name_lower:
                logger.info(f"Найдены данные короткого облучения в листе: {sheet_name}")
                short_data = sheet_data['spectra'].T  # Транспонируем для соответствия формату
        
        if long_data is None or short_data is None:
            logger.warning("Не удалось найти данные длинного или короткого облучения")
            # Возвращаем первые доступные данные
            for sheet_name, sheet_data in integral_data.items():
                if long_data is None:
                    long_data = sheet_data['spectra'].T
                    logger.info(f"Используем данные из листа {sheet_name} как длинное облучение")
                elif short_data is None:
                    short_data = sheet_data['spectra'].T
                    logger.info(f"Используем данные из листа {sheet_name} как короткое облучение")
                    break
        
        return long_data, short_data
    
    def get_energy_bins(self, integral_data: Dict) -> np.ndarray:
        """
        Получение энергетических бинов из данных
        
        Args:
            integral_data: данные интегральных спектров
            
        Returns:
            np.ndarray: энергетические бины
        """
        # Берем энергетические бины из первого доступного листа
        for sheet_name, sheet_data in integral_data.items():
            return sheet_data['energy_bins']
        
        # Если данных нет, возвращаем стандартные бины
        return np.arange(0, 1600, 10)
    
    def load_all_data(self) -> Dict:
        """
        Загрузка всех данных
        
        Returns:
            Dict: все загруженные данные
        """
        logger.info("Загрузка интегральных спектров...")
        integral_data = self.load_integral_spectra()
        
        logger.info("Загрузка групповых спектров...")
        group_data = self.load_group_spectra()
        
        # Извлекаем экспериментальные данные
        long_data, short_data = self.extract_experimental_data(integral_data)
        energy_bins = self.get_energy_bins(integral_data)
        
        return {
            'integral_data': integral_data,
            'group_data': group_data,
            'long_irradiation_data': long_data,
            'short_irradiation_data': short_data,
            'energy_bins': energy_bins
        }
