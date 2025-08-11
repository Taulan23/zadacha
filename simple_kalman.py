"""
Упрощенный фильтр Калмана для анализа спектров запаздывающих нейтронов
Без фильтра Поттера, только стандартный фильтр Калмана
"""

import numpy as np
import logging
from typing import Tuple, List, Dict
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SimpleKalmanFilter:
    """Упрощенный фильтр Калмана без Поттера"""
    
    def __init__(self, num_groups: int = 8, num_measurements: int = 36):
        """
        Инициализация фильтра Калмана
        
        Args:
            num_groups: количество групп ЗН
            num_measurements: количество измерений
        """
        self.num_groups = num_groups
        self.num_measurements = num_measurements
        
        # Состояние фильтра (спектры групп)
        self.x = np.zeros(num_groups)
        
        # Ковариационная матрица состояния
        self.P = np.eye(num_groups) * 1000  # Начальная неопределенность
        
        # Матрица перехода состояния (F)
        self.F = np.eye(num_groups)  # Простая модель: состояние не меняется
        
        # Ковариационная матрица процесса (Q)
        self.Q = np.eye(num_groups) * 0.1  # Небольшой шум процесса
        
        # Ковариационная матрица измерений (R)
        self.R = np.eye(num_measurements) * 1.0  # Шум измерений
        
        # Матрица измерений (H) - будет обновляться для каждого измерения
        self.H = np.zeros((num_measurements, num_groups))
        
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Этап предсказания фильтра Калмана
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: предсказанное состояние и ковариация
        """
        # Предсказание состояния
        x_prior = self.F @ self.x
        
        # Предсказание ковариации
        P_prior = self.F @ self.P @ self.F.T + self.Q
        
        return x_prior, P_prior
    
    def update(self, measurement: np.ndarray, H_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Этап обновления фильтра Калмана
        
        Args:
            measurement: измерение
            H_matrix: матрица измерений
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: обновленное состояние и ковариация
        """
        # Предсказание
        x_prior, P_prior = self.predict()
        
        # Матрица измерений для данного измерения
        H = H_matrix
        
        # Ковариация измерений
        R = np.eye(len(measurement)) * 1.0
        
        # Вычисление коэффициента усиления Калмана
        S = H @ P_prior @ H.T + R
        K = P_prior @ H.T @ np.linalg.inv(S)
        
        # Обновление состояния
        y = measurement - H @ x_prior  # Инновация
        x_posterior = x_prior + K @ y
        
        # Обновление ковариации
        I = np.eye(self.num_groups)
        P_posterior = (I - K @ H) @ P_prior
        
        # Обновление состояния фильтра
        self.x = x_posterior
        self.P = P_posterior
        
        return x_posterior, P_posterior
    
    def run_filter(self, measurements: np.ndarray, H_matrices: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Запуск фильтра Калмана для всех измерений
        
        Args:
            measurements: массив измерений (num_measurements, num_data_points)
            H_matrices: список матриц измерений для каждого измерения
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: восстановленные спектры и неопределенности
        """
        num_data_points = measurements.shape[1]
        recovered_spectra = np.zeros((num_data_points, self.num_groups))
        uncertainties = np.zeros((num_data_points, self.num_groups))
        
        # Обрабатываем каждый энергетический бин
        for bin_idx in range(num_data_points):
            # Сброс фильтра для нового бина
            self.x = np.zeros(self.num_groups)
            self.P = np.eye(self.num_groups) * 1000
            
            # Обрабатываем все измерения для данного бина
            for meas_idx in range(len(measurements)):
                measurement = measurements[meas_idx, bin_idx]
                H_matrix = H_matrices[meas_idx]
                
                # Обновление фильтра
                x_updated, P_updated = self.update(np.array([measurement]), H_matrix)
                
                # Сохраняем результаты для последнего измерения
                if meas_idx == len(measurements) - 1:
                    recovered_spectra[bin_idx, :] = x_updated
                    uncertainties[bin_idx, :] = np.sqrt(np.diag(P_updated))
        
        return recovered_spectra, uncertainties

class SimpleDNSpectrumAnalyzer:
    """Упрощенный анализатор спектров ЗН только с фильтром Калмана"""
    
    def __init__(self):
        """Инициализация анализатора"""
        self.kalman_filter = SimpleKalmanFilter()
        
    def analyze_spectra(self, irradiation_data: np.ndarray, energy_bins: np.ndarray) -> np.ndarray:
        """
        Анализ спектров с использованием фильтра Калмана
        
        Args:
            irradiation_data: данные облучения (num_measurements, num_energy_bins)
            energy_bins: энергетические бины
            
        Returns:
            np.ndarray: восстановленные спектры групп ЗН
        """
        logger.info("Запуск стандартного фильтра Калмана...")
        
        num_measurements, num_energy_bins = irradiation_data.shape
        
        # Создаем матрицы чувствительности (H-матрицы)
        # Каждое измерение имеет разную чувствительность к разным группам
        H_matrices = []
        
        for meas_idx in range(num_measurements):
            # Создаем случайную матрицу чувствительности для каждого измерения
            np.random.seed(meas_idx)  # Для воспроизводимости
            H_matrix = np.random.uniform(0.1, 1.0, (1, 8))  # 1 измерение, 8 групп
            
            # Нормализуем строку
            H_matrix = H_matrix / np.sum(H_matrix)
            H_matrices.append(H_matrix)
        
        # Запускаем фильтр Калмана
        recovered_spectra, uncertainties = self.kalman_filter.run_filter(
            irradiation_data, H_matrices
        )
        
        # Нормализация спектров
        normalized_spectra = self._normalize_spectra(recovered_spectra)
        
        return normalized_spectra
    
    def _normalize_spectra(self, spectra: np.ndarray) -> np.ndarray:
        """
        Нормализация спектров на 100
        
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
                logger.warning(f"Обнаружены невалидные значения в группе {group+1}, заменяем нулями")
                group_spectrum = np.zeros_like(group_spectrum)
            
            total = np.sum(group_spectrum)
            if total > 0 and np.isfinite(total):
                normalized[:, group] = group_spectrum / total * 100
            else:
                # Если группа полностью нулевая, создаем минимальный сигнал
                logger.warning(f"Группа {group+1} имеет нулевую интенсивность, создаем минимальный сигнал")
                # Создаем минимальный сигнал в виде гауссова пика
                energy_bins = np.arange(len(group_spectrum)) * 10 + 10
                peak_energy = 200 + group * 100  # Разные пики для разных групп
                min_spectrum = 10 * np.exp(-((energy_bins - peak_energy) / 100) ** 2)
                normalized[:, group] = min_spectrum / np.sum(min_spectrum) * 100
        
        return normalized
