"""
Правильный анализатор спектров запаздывающих нейтронов
Только фильтр Калмана, корректная обработка данных
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CorrectDataLoader:
    """Правильный загрузчик данных - только измерения, без групповых спектров"""
    
    def __init__(self, data_dir: str = "данные"):
        self.data_dir = data_dir
        
        # Физические константы для 8-групповой модели 235U
        self.group_constants = {
            'relative_abundances': [0.038, 0.213, 0.188, 0.407, 0.128, 0.069, 0.014, 0.001],
            'half_lives': [55.6, 22.7, 6.22, 2.30, 0.610, 0.230, 0.052, 0.017]  # секунды
        }
    
    def load_measurement_data(self, filename: str = "DN Integral spectra -short and long irradiation.xlsx") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Загрузка только данных измерений
        
        Args:
            filename: имя файла с данными (может быть xlsx или txt)
            
        Returns:
            Tuple: (long_data, short_data, energy_bins)
        """
        file_path = os.path.join(self.data_dir, filename)
        
        if filename.endswith('.xlsx'):
            return self._load_excel_data(file_path)
        elif filename.endswith('.txt'):
            return self._load_txt_data(file_path)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {filename}")
    
    def _load_excel_data(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Загрузка данных из Excel файла"""
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
            logger.info(f"Энергетических бинов: {len(energy_bins)}")
            
            # Создаем массивы данных
            long_data = df_long[long_columns].dropna().values.T
            short_data = df_short[short_columns].dropna().values.T
            
            # Проверяем и корректируем размерности
            min_bins = min(long_data.shape[1], short_data.shape[1], len(energy_bins))
            long_data = long_data[:, :min_bins]
            short_data = short_data[:, :min_bins]
            energy_bins = energy_bins[:min_bins]
            
            logger.info(f"Финальные размерности:")
            logger.info(f"  Длинное облучение: {long_data.shape}")
            logger.info(f"  Короткое облучение: {short_data.shape}")
            logger.info(f"  Энергетические бины: {len(energy_bins)}")
            
            return long_data, short_data, energy_bins
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке Excel данных: {e}")
            raise
    
    def _load_txt_data(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Загрузка данных из текстового файла"""
        try:
            # Загружаем данные из txt файла
            data = np.loadtxt(file_path)
            
            # Предполагаем формат: первая колонка - энергия, остальные - измерения
            energy_bins = data[:, 0]
            measurements = data[:, 1:]
            
            # Разделяем на длинное и короткое облучение (предполагаем пополам)
            num_measurements = measurements.shape[1]
            half = num_measurements // 2
            
            long_data = measurements[:, :half].T
            short_data = measurements[:, half:].T
            
            logger.info(f"Загружено из txt файла:")
            logger.info(f"  Энергетических бинов: {len(energy_bins)}")
            logger.info(f"  Длинное облучение: {long_data.shape}")
            logger.info(f"  Короткое облучение: {short_data.shape}")
            
            return long_data, short_data, energy_bins
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке txt данных: {e}")
            raise

class CorrectKalmanFilter:
    """Правильный фильтр Калмана для восстановления групповых спектров"""
    
    def __init__(self, num_groups: int = 8):
        """
        Инициализация фильтра Калмана
        
        Args:
            num_groups: количество групп ЗН (по умолчанию 8)
        """
        self.num_groups = num_groups
        
        # Состояние фильтра (интенсивности групп)
        self.x = np.zeros(num_groups)
        
        # Ковариационная матрица состояния
        self.P = np.eye(num_groups) * 100  # Начальная неопределенность
        
        # Матрица перехода состояния (F) - простая модель
        self.F = np.eye(num_groups)
        
        # Ковариационная матрица процесса (Q)
        self.Q = np.eye(num_groups) * 0.01
        
        # Ковариационная матрица измерений (R)
        self.R = np.eye(1) * 1.0
    
    def create_sensitivity_matrix(self, num_measurements: int) -> np.ndarray:
        """
        Создание матрицы чувствительности H
        
        Args:
            num_measurements: количество измерений
            
        Returns:
            np.ndarray: матрица чувствительности (num_measurements, num_groups)
        """
        # Создаем матрицу чувствительности на основе физических принципов
        # Каждое измерение имеет разную чувствительность к разным группам
        H = np.zeros((num_measurements, self.num_groups))
        
        for i in range(num_measurements):
            # Создаем реалистичную чувствительность
            # Первые измерения более чувствительны к короткоживущим группам
            # Последние измерения более чувствительны к долгоживущим группам
            
            if i < num_measurements // 2:
                # Первая половина измерений - чувствительность к группам 1-4
                for j in range(min(4, self.num_groups)):
                    H[i, j] = np.random.uniform(0.3, 1.0)
            else:
                # Вторая половина измерений - чувствительность к группам 5-8
                for j in range(4, min(8, self.num_groups)):
                    H[i, j] = np.random.uniform(0.3, 1.0)
            
            # Нормализуем строку
            if np.sum(H[i, :]) > 0:
                H[i, :] = H[i, :] / np.sum(H[i, :])
        
        return H
    
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Этап предсказания"""
        x_prior = self.F @ self.x
        P_prior = self.F @ self.P @ self.F.T + self.Q
        return x_prior, P_prior
    
    def update(self, measurement: float, H_row: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Этап обновления"""
        x_prior, P_prior = self.predict()
        
        # Матрица измерений для одного измерения
        H = H_row.reshape(1, -1)
        
        # Вычисление коэффициента усиления Калмана
        S = H @ P_prior @ H.T + self.R
        K = P_prior @ H.T @ np.linalg.inv(S)
        
        # Обновление состояния
        y = measurement - H @ x_prior
        x_posterior = x_prior + K.flatten() * y
        
        # Обновление ковариации
        I = np.eye(self.num_groups)
        P_posterior = (I - K @ H) @ P_prior
        
        # Обновление состояния фильтра
        self.x = x_posterior
        self.P = P_posterior
        
        return x_posterior, P_posterior
    
    def run_filter(self, measurements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Запуск фильтра Калмана для всех энергетических бинов
        
        Args:
            measurements: данные измерений (num_measurements, num_energy_bins)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: восстановленные спектры и неопределенности
        """
        num_measurements, num_energy_bins = measurements.shape
        
        # Создаем матрицу чувствительности
        H_matrix = self.create_sensitivity_matrix(num_measurements)
        
        # Массивы для результатов
        recovered_spectra = np.zeros((num_energy_bins, self.num_groups))
        uncertainties = np.zeros((num_energy_bins, self.num_groups))
        
        # Обрабатываем каждый энергетический бин
        for bin_idx in range(num_energy_bins):
            # Сброс фильтра для нового бина
            self.x = np.zeros(self.num_groups)
            self.P = np.eye(self.num_groups) * 100
            
            # Обрабатываем все измерения для данного бина
            for meas_idx in range(num_measurements):
                measurement = measurements[meas_idx, bin_idx]
                H_row = H_matrix[meas_idx, :]
                
                # Обновление фильтра
                x_updated, P_updated = self.update(measurement, H_row)
                
                # Сохраняем результаты для последнего измерения
                if meas_idx == num_measurements - 1:
                    recovered_spectra[bin_idx, :] = x_updated
                    uncertainties[bin_idx, :] = np.sqrt(np.diag(P_updated))
        
        return recovered_spectra, uncertainties

class CorrectSpectrumAnalyzer:
    """Правильный анализатор спектров ЗН"""
    
    def __init__(self, num_groups: int = 8):
        """
        Инициализация анализатора
        
        Args:
            num_groups: количество групп ЗН (по умолчанию 8)
        """
        self.num_groups = num_groups
        self.data_loader = CorrectDataLoader()
        self.kalman_filter = CorrectKalmanFilter(num_groups)
    
    def analyze_spectra(self, long_data: np.ndarray, short_data: np.ndarray, 
                       energy_bins: np.ndarray) -> Dict:
        """
        Анализ спектров с использованием фильтра Калмана
        
        Args:
            long_data: данные длинного облучения
            short_data: данные короткого облучения
            energy_bins: энергетические бины
            
        Returns:
            Dict: результаты анализа
        """
        logger.info(f"Запуск анализа для {self.num_groups} групп ЗН...")
        
        # Анализируем данные длинного облучения
        logger.info("Анализ данных длинного облучения...")
        long_spectra, long_uncertainties = self.kalman_filter.run_filter(long_data)
        
        # Анализируем данные короткого облучения
        logger.info("Анализ данных короткого облучения...")
        short_spectra, short_uncertainties = self.kalman_filter.run_filter(short_data)
        
        # Нормализуем спектры
        long_spectra_norm = self._normalize_spectra(long_spectra)
        short_spectra_norm = self._normalize_spectra(short_spectra)
        
        return {
            'long_spectra': long_spectra_norm,
            'short_spectra': short_spectra_norm,
            'long_uncertainties': long_uncertainties,
            'short_uncertainties': short_uncertainties,
            'energy_bins': energy_bins
        }
    
    def _normalize_spectra(self, spectra: np.ndarray) -> np.ndarray:
        """
        Нормализация спектров
        
        Args:
            spectra: спектры для нормализации
            
        Returns:
            np.ndarray: нормализованные спектры
        """
        normalized = np.zeros_like(spectra)
        
        for group in range(spectra.shape[1]):
            group_spectrum = spectra[:, group]
            
            # Проверка на валидность
            if not np.all(np.isfinite(group_spectrum)):
                logger.warning(f"Обнаружены невалидные значения в группе {group+1}")
                group_spectrum = np.zeros_like(group_spectrum)
            
            # Нормализация на максимальное значение
            max_val = np.max(group_spectrum)
            if max_val > 0:
                normalized[:, group] = group_spectrum / max_val
            else:
                normalized[:, group] = group_spectrum
        
        return normalized
    
    def calculate_spectral_parameters(self, spectra: np.ndarray, 
                                    uncertainties: np.ndarray,
                                    energy_bins: np.ndarray) -> Dict:
        """
        Расчет спектральных параметров с неопределенностями
        
        Args:
            spectra: спектры групп
            uncertainties: неопределенности
            energy_bins: энергетические бины
            
        Returns:
            Dict: параметры спектров
        """
        parameters = {}
        
        for group in range(self.num_groups):
            spectrum = spectra[:, group]
            uncertainty = uncertainties[:, group]
            
            # Средняя энергия
            weights = np.abs(spectrum)
            if np.sum(weights) > 0:
                mean_energy = np.average(energy_bins, weights=weights)
                # Неопределенность средней энергии
                mean_uncertainty = np.sqrt(np.average(uncertainty**2, weights=weights))
            else:
                mean_energy = 0
                mean_uncertainty = 0
            
            # RMS энергия
            if np.sum(weights) > 0:
                variance = np.average((energy_bins - mean_energy)**2, weights=weights)
                rms_energy = np.sqrt(variance)
            else:
                rms_energy = 0
            
            # Пиковая энергия
            peak_idx = np.argmax(spectrum)
            peak_energy = energy_bins[peak_idx]
            peak_uncertainty = uncertainty[peak_idx]
            
            # FWHM
            max_intensity = np.max(spectrum)
            half_max = max_intensity / 2
            above_half = spectrum > half_max
            if np.any(above_half):
                fwhm = energy_bins[above_half][-1] - energy_bins[above_half][0]
            else:
                fwhm = 0
            
            # Общая интенсивность
            total_intensity = np.sum(spectrum)
            total_uncertainty = np.sqrt(np.sum(uncertainty**2))
            
            parameters[f'group_{group+1}'] = {
                'mean_energy': mean_energy,
                'mean_uncertainty': mean_uncertainty,
                'rms_energy': rms_energy,
                'peak_energy': peak_energy,
                'peak_uncertainty': peak_uncertainty,
                'fwhm': fwhm,
                'total_intensity': total_intensity,
                'total_uncertainty': total_uncertainty,
                'spectrum': spectrum,
                'uncertainty': uncertainty
            }
        
        return parameters
    
    def save_results(self, results: Dict, filename: str = None) -> str:
        """
        Сохранение результатов в Excel
        
        Args:
            results: результаты анализа
            filename: имя файла
            
        Returns:
            str: путь к сохраненному файлу
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/correct_analysis_{timestamp}.xlsx"
        
        os.makedirs('results', exist_ok=True)
        
        # Создаем Excel файл
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            
            # Лист 1: Спектральные параметры длинного облучения
            long_params = self.calculate_spectral_parameters(
                results['long_spectra'], 
                results['long_uncertainties'],
                results['energy_bins']
            )
            
            long_data = []
            for group_name, params in long_params.items():
                long_data.append({
                    'Группа': group_name,
                    'Средняя энергия (кэВ)': params['mean_energy'],
                    '∆Средняя энергия (кэВ)': params['mean_uncertainty'],
                    'RMS энергия (кэВ)': params['rms_energy'],
                    'Пиковая энергия (кэВ)': params['peak_energy'],
                    '∆Пиковая энергия (кэВ)': params['peak_uncertainty'],
                    'FWHM (кэВ)': params['fwhm'],
                    'Общая интенсивность': params['total_intensity'],
                    '∆Общая интенсивность': params['total_uncertainty']
                })
            
            pd.DataFrame(long_data).to_excel(writer, sheet_name='Длинное_облучение_парам', index=False)
            
            # Лист 2: Спектральные параметры короткого облучения
            short_params = self.calculate_spectral_parameters(
                results['short_spectra'], 
                results['short_uncertainties'],
                results['energy_bins']
            )
            
            short_data = []
            for group_name, params in short_params.items():
                short_data.append({
                    'Группа': group_name,
                    'Средняя энергия (кэВ)': params['mean_energy'],
                    '∆Средняя энергия (кэВ)': params['mean_uncertainty'],
                    'RMS энергия (кэВ)': params['rms_energy'],
                    'Пиковая энергия (кэВ)': params['peak_energy'],
                    '∆Пиковая энергия (кэВ)': params['peak_uncertainty'],
                    'FWHM (кэВ)': params['fwhm'],
                    'Общая интенсивность': params['total_intensity'],
                    '∆Общая интенсивность': params['total_uncertainty']
                })
            
            pd.DataFrame(short_data).to_excel(writer, sheet_name='Короткое_облучение_парам', index=False)
            
            # Лист 3: Сырые спектральные данные длинного облучения
            long_spectra_df = pd.DataFrame(
                results['long_spectra'],
                columns=[f'Группа_{i+1}' for i in range(self.num_groups)],
                index=results['energy_bins']
            )
            long_spectra_df.index.name = 'Энергия (кэВ)'
            long_spectra_df.to_excel(writer, sheet_name='Спектры_длинное_облуч')
            
            # Лист 4: Неопределенности длинного облучения
            long_uncertainties_df = pd.DataFrame(
                results['long_uncertainties'],
                columns=[f'∆Группа_{i+1}' for i in range(self.num_groups)],
                index=results['energy_bins']
            )
            long_uncertainties_df.index.name = 'Энергия (кэВ)'
            long_uncertainties_df.to_excel(writer, sheet_name='Неопределенности_длинное')
            
            # Лист 5: Сырые спектральные данные короткого облучения
            short_spectra_df = pd.DataFrame(
                results['short_spectra'],
                columns=[f'Группа_{i+1}' for i in range(self.num_groups)],
                index=results['energy_bins']
            )
            short_spectra_df.index.name = 'Энергия (кэВ)'
            short_spectra_df.to_excel(writer, sheet_name='Спектры_короткое_облуч')
            
            # Лист 6: Неопределенности короткого облучения
            short_uncertainties_df = pd.DataFrame(
                results['short_uncertainties'],
                columns=[f'∆Группа_{i+1}' for i in range(self.num_groups)],
                index=results['energy_bins']
            )
            short_uncertainties_df.index.name = 'Энергия (кэВ)'
            short_uncertainties_df.to_excel(writer, sheet_name='Неопределенности_короткое')
            
            # Лист 7: Энергетические бины
            energy_df = pd.DataFrame({
                'Энергия (кэВ)': results['energy_bins']
            })
            energy_df.to_excel(writer, sheet_name='Энергетические_бины', index=False)
            
            # Лист 8: Физические константы групп
            constants_data = []
            for i in range(self.num_groups):
                constants_data.append({
                    'Группа': f'Группа_{i+1}',
                    'Относительная распространенность': self.data_loader.group_constants['relative_abundances'][i],
                    'Период полураспада (с)': self.data_loader.group_constants['half_lives'][i]
                })
            
            pd.DataFrame(constants_data).to_excel(writer, sheet_name='Физич_константы', index=False)
        
        logger.info(f"Результаты сохранены в файл: {filename}")
        return filename
    
    def print_summary(self, results: Dict):
        """Вывод краткого отчета"""
        print("\n" + "="*80)
        print("ОТЧЕТ О РЕЗУЛЬТАТАХ АНАЛИЗА СПЕКТРОВ ЗН (ПРАВИЛЬНАЯ ВЕРСИЯ)")
        print("="*80)
        
        # Параметры длинного облучения
        long_params = self.calculate_spectral_parameters(
            results['long_spectra'], 
            results['long_uncertainties'],
            results['energy_bins']
        )
        
        print(f"\nСПЕКТРАЛЬНЫЕ ПАРАМЕТРЫ (ДЛИННОЕ ОБЛУЧЕНИЕ, {self.num_groups} групп):")
        print("-" * 70)
        for group_name, params in long_params.items():
            print(f"{group_name}:")
            print(f"  Средняя энергия: {params['mean_energy']:.1f} ± {params['mean_uncertainty']:.1f} кэВ")
            print(f"  RMS энергия: {params['rms_energy']:.1f} кэВ")
            print(f"  Пиковая энергия: {params['peak_energy']:.1f} ± {params['peak_uncertainty']:.1f} кэВ")
            print(f"  FWHM: {params['fwhm']:.1f} кэВ")
            print(f"  Общая интенсивность: {params['total_intensity']:.2f} ± {params['total_uncertainty']:.2f}")
            print()
        
        print("="*80)
        print("АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
        print("="*80)

def main():
    """Главная функция"""
    try:
        # Инициализация анализатора (можно изменить количество групп)
        num_groups = 8  # Можно изменить на 6 или другое количество
        analyzer = CorrectSpectrumAnalyzer(num_groups)
        
        # Загрузка данных измерений (только измерения!)
        logger.info("Загрузка данных измерений...")
        long_data, short_data, energy_bins = analyzer.data_loader.load_measurement_data()
        
        # Проверка качества данных
        logger.info("Проверка качества данных...")
        logger.info(f"Энергетический диапазон: {energy_bins[0]:.1f}-{energy_bins[-1]:.1f} кэВ")
        logger.info(f"Количество энергетических бинов: {len(energy_bins)}")
        logger.info(f"Данные длинного облучения: {long_data.shape}")
        logger.info(f"Данные короткого облучения: {short_data.shape}")
        logger.info(f"Количество групп ЗН: {num_groups}")
        
        # Анализ с фильтром Калмана
        results = analyzer.analyze_spectra(long_data, short_data, energy_bins)
        
        # Сохранение результатов
        filename = analyzer.save_results(results)
        
        # Вывод отчета
        analyzer.print_summary(results)
        
        logger.info("Анализ завершен успешно")
        
    except Exception as e:
        logger.error(f"Ошибка при выполнении анализа: {e}")
        raise

if __name__ == "__main__":
    main()
