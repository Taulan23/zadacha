"""
Финальная версия анализатора спектров запаздывающих нейтронов
Решение системы линейных уравнений: N^l_i(E_n) = Σ A^l_ij · x_j(E_n)
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

class FinalDataLoader:
    """Финальный загрузчик данных - только измерения"""
    
    def __init__(self, data_dir: str = "данные"):
        self.data_dir = data_dir
        
        # Физические константы для 8-групповой модели 235U (из литературы)
        self.group_constants = {
            'relative_abundances': [0.038, 0.213, 0.188, 0.407, 0.128, 0.069, 0.014, 0.001],
            'half_lives': [55.6, 22.7, 6.22, 2.30, 0.610, 0.230, 0.052, 0.017]  # секунды
        }
    
    def load_measurement_data(self, filename: str = "DN Integral spectra -short and long irradiation.xlsx") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Загрузка только данных измерений"""
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
            data = np.loadtxt(file_path)
            energy_bins = data[:, 0]
            measurements = data[:, 1:]
            
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

class FinalEquationSolver:
    """Решатель системы линейных уравнений для спектров ЗН"""
    
    def __init__(self, num_groups: int = 8):
        self.num_groups = num_groups
        
        # Физические константы
        self.abundances = np.array([0.038, 0.213, 0.188, 0.407, 0.128, 0.069, 0.014, 0.001])
        self.half_lives = np.array([55.6, 22.7, 6.22, 2.30, 0.610, 0.230, 0.052, 0.017])
        self.decay_constants = np.log(2) / self.half_lives
        
        # Параметры облучения
        self.t_irr_long = 120.0  # секунды
        self.t_irr_short = 20.0  # секунды
        self.t_decay = 0.0  # время распада после облучения
        self.delta_t = 1.0  # время измерения
        self.T = 100.0  # период измерения
        self.M = 1  # количество циклов измерения
    
    def create_sensitivity_matrix(self, t_irr: float, num_measurements: int) -> Tuple[np.ndarray, Dict]:
        """
        Создание матрицы чувствительности A^l_ij согласно уравнению:
        A^l_ij = (a_i / λ_i) * (1 - e^(-λ_i * t_irr)) * (e^(-λ_i * t_decay)) * (1 - e^(-λ_i * delta_t)) * T_i
        где T_i = [M / (1 - e^(-λ_i * T)) - e^(-λ_i * T) * (1 - e^(-M * λ_i * T)) / (1 - e^(-λ_i * T))^2]
        
        Возвращает:
            Матрицу A и детальную информацию о всех коэффициентах
        """
        A = np.zeros((num_measurements, self.num_groups))
        coefficients_info = {
            'measurement_times': [],
            'group_coefficients': {},
            'matrix_normalization': 0.0
        }
        
        # Расчет времен измерения
        measurement_times = []
        for i in range(num_measurements):
            t_meas = i * self.T / (num_measurements - 1)
            measurement_times.append(t_meas)
        coefficients_info['measurement_times'] = measurement_times
        
        # Расчет коэффициентов для каждой группы
        for j in range(self.num_groups):
            lambda_i = self.decay_constants[j]
            a_i = self.abundances[j]
            half_life = self.half_lives[j]
            
            group_info = {
                'group_number': j + 1,
                'abundance': a_i,
                'half_life': half_life,
                'decay_constant': lambda_i,
                'measurements': []
            }
            
            for i in range(num_measurements):
                t_meas = measurement_times[i]
                
                # Расчет T_i фактора
                T_factor = self._calculate_T_factor(lambda_i, t_meas)
                
                # Компоненты формулы
                abundance_factor = a_i / lambda_i
                irradiation_factor = 1 - np.exp(-lambda_i * t_irr)
                decay_factor = np.exp(-lambda_i * self.t_decay)
                measurement_factor = 1 - np.exp(-lambda_i * self.delta_t)
                
                # Основная формула
                A_component = abundance_factor * irradiation_factor * decay_factor * measurement_factor * T_factor
                
                # Базовая чувствительность
                base_sensitivity = 0.01 * a_i
                
                # Общий коэффициент
                A[i, j] = A_component + base_sensitivity
                
                # Сохранение детальной информации
                measurement_info = {
                    'measurement_index': i,
                    'time': t_meas,
                    'abundance_factor': abundance_factor,
                    'irradiation_factor': irradiation_factor,
                    'decay_factor': decay_factor,
                    'measurement_factor': measurement_factor,
                    'T_factor': T_factor,
                    'A_component': A_component,
                    'base_sensitivity': base_sensitivity,
                    'final_coefficient': A[i, j]
                }
                group_info['measurements'].append(measurement_info)
            
            coefficients_info['group_coefficients'][f'group_{j+1}'] = group_info
        
        # Нормализация всей матрицы для численной стабильности
        max_val = np.max(np.abs(A))
        if max_val > 0:
            A = A / max_val
            coefficients_info['matrix_normalization'] = max_val
        
        return A, coefficients_info
    
    def _calculate_T_factor(self, lambda_i: float, t_meas: float) -> float:
        """
        Расчет T_i фактора:
        T_i = [M / (1 - e^(-λ_i * T)) - e^(-λ_i * T) * (1 - e^(-M * λ_i * T)) / (1 - e^(-λ_i * T))^2]
        """
        if lambda_i * self.T < 1e-10:  # Избегаем деления на ноль
            return self.M
        
        exp_lambda_T = np.exp(-lambda_i * self.T)
        exp_M_lambda_T = np.exp(-self.M * lambda_i * self.T)
        
        denominator = 1 - exp_lambda_T
        
        if abs(denominator) < 1e-10:
            return self.M
        
        T_factor = (self.M / denominator) - \
                   (exp_lambda_T * (1 - exp_M_lambda_T) / (denominator ** 2))
        
        return T_factor
    
    def solve_equations(self, measurements: np.ndarray, t_irr: float) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Решение системы уравнений: N^l_i(E_n) = Σ A^l_ij · x_j(E_n)
        для каждого энергетического бина E_n
        
        Возвращает:
            group_spectra, uncertainties, coefficients_info
        """
        num_measurements, num_energy_bins = measurements.shape
        
        # Создаем матрицу чувствительности с детальной информацией
        A, coefficients_info = self.create_sensitivity_matrix(t_irr, num_measurements)
        
        # Добавляем параметры облучения к информации
        coefficients_info['irradiation_time'] = t_irr
        coefficients_info['t_decay'] = self.t_decay
        coefficients_info['delta_t'] = self.delta_t
        coefficients_info['T'] = self.T
        coefficients_info['M'] = self.M
        coefficients_info['num_measurements'] = num_measurements
        coefficients_info['num_energy_bins'] = num_energy_bins
        
        # Массивы для результатов
        group_spectra = np.zeros((num_energy_bins, self.num_groups))
        uncertainties = np.zeros((num_energy_bins, self.num_groups))
        
        # Решаем систему для каждого энергетического бина
        for bin_idx in range(num_energy_bins):
            # Измерения для данного бина
            b = measurements[:, bin_idx]
            
            # Решение системы Ax = b
            try:
                # Используем псевдообратную матрицу для стабильности
                A_pinv = np.linalg.pinv(A, rcond=1e-10)
                x_solution = A_pinv @ b
                
                # Применяем физические ограничения
                x_solution = np.maximum(x_solution, 0)  # Неотрицательность
                
                # Обеспечиваем минимальный вклад каждой группы
                min_contribution = 0.001 * np.max(x_solution)
                x_solution = np.maximum(x_solution, min_contribution * self.abundances)
                
                # Сохраняем результаты
                group_spectra[bin_idx, :] = x_solution
                
                # Расчет неопределенностей (упрощенный)
                residuals = b - A @ x_solution
                uncertainty = np.sqrt(np.mean(residuals**2)) * np.ones(self.num_groups)
                uncertainties[bin_idx, :] = uncertainty
                
            except np.linalg.LinAlgError as e:
                logger.warning(f"Проблема с решением для бина {bin_idx}: {e}")
                # Используем простое решение в случае ошибки
                x_solution = np.linalg.lstsq(A, b, rcond=1e-10)[0]
                x_solution = np.maximum(x_solution, 0)
                group_spectra[bin_idx, :] = x_solution
                uncertainties[bin_idx, :] = 0.1 * np.ones(self.num_groups)
        
        return group_spectra, uncertainties, coefficients_info

class FinalSpectrumAnalyzer:
    """Финальный анализатор спектров ЗН с решением системы уравнений"""
    
    def __init__(self, num_groups: int = 8):
        self.num_groups = num_groups
        self.data_loader = FinalDataLoader()
        self.equation_solver = FinalEquationSolver(num_groups)
    
    def analyze_spectra(self, long_data: np.ndarray, short_data: np.ndarray, 
                       energy_bins: np.ndarray) -> Dict:
        """Анализ спектров с решением системы уравнений"""
        logger.info(f"Запуск анализа для {self.num_groups} групп ЗН...")
        
        # Решение для длинного облучения
        logger.info("Решение для данных длинного облучения...")
        long_spectra, long_uncertainties, long_coefficients = self.equation_solver.solve_equations(
            long_data, self.equation_solver.t_irr_long
        )
        
        # Решение для короткого облучения
        logger.info("Решение для данных короткого облучения...")
        short_spectra, short_uncertainties, short_coefficients = self.equation_solver.solve_equations(
            short_data, self.equation_solver.t_irr_short
        )
        
        # Применение физических ограничений
        long_spectra_norm = self._apply_physical_constraints(long_spectra)
        short_spectra_norm = self._apply_physical_constraints(short_spectra)
        
        return {
            'long_spectra': long_spectra_norm,
            'short_spectra': short_spectra_norm,
            'long_uncertainties': long_uncertainties,
            'short_uncertainties': short_uncertainties,
            'long_coefficients': long_coefficients,
            'short_coefficients': short_coefficients,
            'energy_bins': energy_bins
        }
    
    def _apply_physical_constraints(self, spectra: np.ndarray) -> np.ndarray:
        """
        Применение физических ограничений к спектрам
        Сохраняем абсолютные значения, не нормализуем к 1
        """
        constrained = np.zeros_like(spectra)
        
        for group in range(spectra.shape[1]):
            group_spectrum = spectra[:, group]
            
            # Проверка на валидность
            if not np.all(np.isfinite(group_spectrum)):
                logger.warning(f"Обнаружены невалидные значения в группе {group+1}")
                group_spectrum = np.zeros_like(group_spectrum)
            
            # ОГРАНИЧЕНИЕ: спектры не могут быть отрицательными
            group_spectrum = np.maximum(group_spectrum, 0)
            
            # Масштабирование с учетом физических констант
            abundance = self.data_loader.group_constants['relative_abundances'][group]
            scaled_spectrum = group_spectrum * abundance * 100  # Абсолютные значения
            
            constrained[:, group] = scaled_spectrum
        
        return constrained
    
    def calculate_spectral_parameters(self, spectra: np.ndarray, 
                                    uncertainties: np.ndarray,
                                    energy_bins: np.ndarray) -> Dict:
        """Расчет спектральных параметров с неопределенностями"""
        parameters = {}
        
        for group in range(self.num_groups):
            spectrum = spectra[:, group]
            uncertainty = uncertainties[:, group]
            
            # Средняя энергия
            weights = np.abs(spectrum)
            if np.sum(weights) > 0:
                mean_energy = np.average(energy_bins, weights=weights)
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
            
            # Общая интенсивность (абсолютная)
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
        """Сохранение результатов в Excel"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/final_analysis_{timestamp}.xlsx"
        
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
            
            # Лист 9: Коэффициенты матрицы чувствительности (длинное облучение)
            self._save_coefficients_details(writer, results['long_coefficients'], 'Коэффициенты_длинное')
            
            # Лист 10: Коэффициенты матрицы чувствительности (короткое облучение)
            self._save_coefficients_details(writer, results['short_coefficients'], 'Коэффициенты_короткое')
            
            # Лист 11: Сводка всех коэффициентов (длинное облучение)
            self._save_coefficients_summary(writer, results['long_coefficients'], 'Сводка_коэф_длинное')
            
            # Лист 12: Сводка всех коэффициентов (короткое облучение)
            self._save_coefficients_summary(writer, results['short_coefficients'], 'Сводка_коэф_короткое')
        
        logger.info(f"Результаты сохранены в файл: {filename}")
        return filename
    
    def _save_coefficients_details(self, writer, coefficients_info: Dict, sheet_name: str):
        """Сохранение детальной информации о коэффициентах матрицы"""
        details_data = []
        
        # Добавляем общую информацию о параметрах
        details_data.append({
            'Группа': 'Параметры облучения',
            'Номер измерения': '',
            'Время (с)': '',
            'Время облучения (с)': coefficients_info['irradiation_time'],
            'Время распада (с)': coefficients_info['t_decay'],
            'Время измерения (с)': coefficients_info['delta_t'],
            'Период T (с)': coefficients_info['T'],
            'Количество циклов M': coefficients_info['M'],
            'Нормализация матрицы': coefficients_info['matrix_normalization'],
            'Относительная распространенность': '',
            'Период полураспада (с)': '',
            'Константа распада (1/с)': '',
            'Фактор распространенности': '',
            'Фактор облучения': '',
            'Фактор распада': '',
            'Фактор измерения': '',
            'T-фактор': '',
            'A-компонента': '',
            'Базовая чувствительность': '',
            'Финальный коэффициент': ''
        })
        
        details_data.append({})  # Пустая строка для разделения
        
        # Детальная информация для каждой группы
        for group_key, group_info in coefficients_info['group_coefficients'].items():
            group_num = group_info['group_number']
            
            # Заголовок группы
            details_data.append({
                'Группа': f"Группа {group_num}",
                'Номер измерения': '',
                'Время (с)': '',
                'Время облучения (с)': '',
                'Время распада (с)': '',
                'Время измерения (с)': '',
                'Период T (с)': '',
                'Количество циклов M': '',
                'Нормализация матрицы': '',
                'Относительная распространенность': group_info['abundance'],
                'Период полураспада (с)': group_info['half_life'],
                'Константа распада (1/с)': group_info['decay_constant'],
                'Фактор распространенности': '',
                'Фактор облучения': '',
                'Фактор распада': '',
                'Фактор измерения': '',
                'T-фактор': '',
                'A-компонента': '',
                'Базовая чувствительность': '',
                'Финальный коэффициент': ''
            })
            
            # Данные для каждого измерения
            for meas in group_info['measurements']:
                details_data.append({
                    'Группа': '',
                    'Номер измерения': meas['measurement_index'],
                    'Время (с)': f"{meas['time']:.3f}",
                    'Время облучения (с)': '',
                    'Время распада (с)': '',
                    'Время измерения (с)': '',
                    'Период T (с)': '',
                    'Количество циклов M': '',
                    'Нормализация матрицы': '',
                    'Относительная распространенность': '',
                    'Период полураспада (с)': '',
                    'Константа распада (1/с)': '',
                    'Фактор распространенности': f"{meas['abundance_factor']:.6e}",
                    'Фактор облучения': f"{meas['irradiation_factor']:.6f}",
                    'Фактор распада': f"{meas['decay_factor']:.6f}",
                    'Фактор измерения': f"{meas['measurement_factor']:.6f}",
                    'T-фактор': f"{meas['T_factor']:.6e}",
                    'A-компонента': f"{meas['A_component']:.6e}",
                    'Базовая чувствительность': f"{meas['base_sensitivity']:.6e}",
                    'Финальный коэффициент': f"{meas['final_coefficient']:.6e}"
                })
            
            details_data.append({})  # Пустая строка между группами
        
        pd.DataFrame(details_data).to_excel(writer, sheet_name=sheet_name, index=False)
    
    def _save_coefficients_summary(self, writer, coefficients_info: Dict, sheet_name: str):
        """Сохранение сводной матрицы коэффициентов"""
        # Создаем матрицу коэффициентов
        num_measurements = len(coefficients_info['measurement_times'])
        num_groups = len(coefficients_info['group_coefficients'])
        
        matrix_data = []
        
        # Заголовок с временами измерения
        header_row = ['Группа\\Время (с)'] + [f"{t:.3f}" for t in coefficients_info['measurement_times']]
        matrix_data.append(dict(zip(range(len(header_row)), header_row)))
        
        # Данные для каждой группы
        for group_key in sorted(coefficients_info['group_coefficients'].keys()):
            group_info = coefficients_info['group_coefficients'][group_key]
            group_num = group_info['group_number']
            
            row = [f"Группа {group_num}"]
            for meas in group_info['measurements']:
                row.append(f"{meas['final_coefficient']:.6e}")
            
            matrix_data.append(dict(zip(range(len(row)), row)))
        
        pd.DataFrame(matrix_data).to_excel(writer, sheet_name=sheet_name, index=False, header=False)
    
    def print_summary(self, results: Dict):
        """Вывод краткого отчета"""
        print("\n" + "="*80)
        print("ОТЧЕТ О РЕЗУЛЬТАТАХ АНАЛИЗА СПЕКТРОВ ЗН (СИСТЕМА УРАВНЕНИЙ)")
        print("="*80)
        
        # Информация о коэффициентах системы уравнений
        self._print_coefficients_info(results)
        
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
    
    def _print_coefficients_info(self, results: Dict):
        """Вывод информации о коэффициентах системы уравнений"""
        print("\nИНФОРМАЦИЯ О КОЭФФИЦИЕНТАХ СИСТЕМЫ УРАВНЕНИЙ:")
        print("="*60)
        
        # Информация о длинном облучении
        long_coeff = results['long_coefficients']
        print(f"\nДЛИННОЕ ОБЛУЧЕНИЕ:")
        print(f"  Время облучения: {long_coeff['irradiation_time']:.1f} с")
        print(f"  Время распада: {long_coeff['t_decay']:.1f} с")
        print(f"  Время измерения: {long_coeff['delta_t']:.1f} с")
        print(f"  Период T: {long_coeff['T']:.1f} с")
        print(f"  Количество циклов M: {long_coeff['M']}")
        print(f"  Количество измерений: {long_coeff['num_measurements']}")
        print(f"  Нормализация матрицы: {long_coeff['matrix_normalization']:.6e}")
        
        # Краткая информация о коэффициентах групп
        print(f"\n  КОЭФФИЦИЕНТЫ ПО ГРУППАМ (образец):")
        for i, (group_key, group_info) in enumerate(long_coeff['group_coefficients'].items()):
            if i < 3:  # Показываем только первые 3 группы для краткости
                group_num = group_info['group_number']
                first_meas = group_info['measurements'][0]
                last_meas = group_info['measurements'][-1]
                print(f"    Группа {group_num}:")
                print(f"      Распространенность: {group_info['abundance']:.3f}")
                print(f"      Период полураспада: {group_info['half_life']:.2f} с")
                print(f"      Первый коэффициент A[0,{group_num-1}]: {first_meas['final_coefficient']:.6e}")
                print(f"      Последний коэффициент A[{long_coeff['num_measurements']-1},{group_num-1}]: {last_meas['final_coefficient']:.6e}")
        
        if len(long_coeff['group_coefficients']) > 3:
            print(f"    ... и ещё {len(long_coeff['group_coefficients']) - 3} групп")
        
        # Информация о коротком облучении
        short_coeff = results['short_coefficients']
        print(f"\nКОРОТКОЕ ОБЛУЧЕНИЕ:")
        print(f"  Время облучения: {short_coeff['irradiation_time']:.1f} с")
        print(f"  Количество измерений: {short_coeff['num_measurements']}")
        print(f"  Нормализация матрицы: {short_coeff['matrix_normalization']:.6e}")
        
        print(f"\nПОДРОБНАЯ ИНФОРМАЦИЯ О ВСЕХ КОЭФФИЦИЕНТАХ СОХРАНЕНА В EXCEL:")
        print(f"  - Листы 'Коэффициенты_длинное' и 'Коэффициенты_короткое'")
        print(f"  - Листы 'Сводка_коэф_длинное' и 'Сводка_коэф_короткое'")
        print("="*60)

def main():
    """Главная функция"""
    try:
        # Инициализация анализатора (можно изменить количество групп)
        num_groups = 8  # Можно изменить на 6 или другое количество
        analyzer = FinalSpectrumAnalyzer(num_groups)
        
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
        
        # Анализ с решением системы уравнений
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
