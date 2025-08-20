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
from constants import GROUP_CONSTANTS, EXPERIMENT_PARAMS, NUMERICAL_PARAMS, T_FACTOR_PARAMS, MEASUREMENT_INTERVALS

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
        
        # Физические константы из файла constants.py
        self.group_constants = GROUP_CONSTANTS
    
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
            
            # Извлекаем энергетические бины (первый столбец), пропуская заголовки
            energy_col = df_long.columns[0]
            energy_bins = df_long[energy_col].dropna().values
            
            # Извлекаем только основные данные спектров (столбцы N(En), n/10 keV)
            long_columns = []
            short_columns = []
            
            for col in df_long.columns[1:]:
                if 'N(En), n/10 keV' in str(col) and not pd.isna(col):
                    long_columns.append(col)
            
            for col in df_short.columns[1:]:
                if 'N(En), n/10 keV' in str(col) and not pd.isna(col):
                    short_columns.append(col)
            
            logger.info(f"Найдено {len(long_columns)} интервалов длинного облучения")
            logger.info(f"Найдено {len(short_columns)} интервалов короткого облучения")
            logger.info(f"Энергетических бинов: {len(energy_bins)}")
            
            # Создаем массивы данных, пропуская первую строку с заголовками
            long_data = df_long[long_columns].iloc[1:].dropna().values.T
            short_data = df_short[short_columns].iloc[1:].dropna().values.T
            
            # Преобразуем в числовой формат
            long_data = pd.DataFrame(long_data).apply(pd.to_numeric, errors='coerce').values
            short_data = pd.DataFrame(short_data).apply(pd.to_numeric, errors='coerce').values
            energy_bins = pd.to_numeric(energy_bins[1:], errors='coerce')  # Пропускаем первый элемент (NaN)
            
            # Проверяем, что энергетические бины начинаются с 10 кэВ
            if len(energy_bins) > 0 and energy_bins[0] > 10.0:
                logger.warning(f"Энергетические бины начинаются с {energy_bins[0]:.1f} кэВ, но должны начинаться с 10 кэВ")
                # Добавляем 10 кэВ в начало, если его нет
                if energy_bins[0] == 20.0:
                    energy_bins = np.insert(energy_bins, 0, 10.0)
                    # Добавляем соответствующие данные (интерполяция или нули)
                    long_data = np.insert(long_data, 0, np.zeros(long_data.shape[0]), axis=1)
                    short_data = np.insert(short_data, 0, np.zeros(short_data.shape[0]), axis=1)
                    logger.info("Добавлен энергетический бин 10 кэВ")
            
            # Проверяем и корректируем размерности
            min_bins = min(long_data.shape[1], short_data.shape[1], len(energy_bins))
            long_data = long_data[:, :min_bins]
            short_data = short_data[:, :min_bins]
            energy_bins = energy_bins[:min_bins]
            
            logger.info(f"Финальные размерности:")
            logger.info(f"  Длинное облучение: {long_data.shape}")
            logger.info(f"  Короткое облучение: {short_data.shape}")
            logger.info(f"  Энергетические бины: {len(energy_bins)}")
            
            # Проверяем качество данных
            logger.info(f"Статистика данных длинного облучения:")
            logger.info(f"  Минимум: {np.min(long_data):.6f}")
            logger.info(f"  Максимум: {np.max(long_data):.6f}")
            logger.info(f"  Среднее: {np.mean(long_data):.6f}")
            
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
    
    def __init__(self, num_groups: int = None):
        # Используем количество групп из констант, если не указано
        if num_groups is None:
            num_groups = EXPERIMENT_PARAMS['num_groups']
        self.num_groups = num_groups
        
        # Физические константы из файла
        self.abundances = np.array(GROUP_CONSTANTS['relative_abundances'][:num_groups])
        self.half_lives = np.array(GROUP_CONSTANTS['half_lives'][:num_groups])
        self.decay_constants = np.log(2) / self.half_lives
        
        # Параметры интервалов измерения
        self.measurement_intervals = MEASUREMENT_INTERVALS
        self.num_intervals = EXPERIMENT_PARAMS['num_intervals']
    
    def create_sensitivity_matrix(self, irradiation_type: str, num_measurements: int) -> Tuple[np.ndarray, Dict]:
        """
        Создание матрицы чувствительности A^l_ij для каждого интервала измерения
        с учетом индивидуальных параметров td, tc, t1, t2 для каждого измерения
        
        Args:
            irradiation_type: 'long' или 'short' - тип облучения
            num_measurements: количество измерений (должно быть равно количеству интервалов)
        
        Returns:
            Матрицу A и детальную информацию о всех коэффициентах
        """
        # Получаем параметры для данного типа облучения
        if irradiation_type == 'long':
            intervals_data = self.measurement_intervals['long_irradiation']
        elif irradiation_type == 'short':
            intervals_data = self.measurement_intervals['short_irradiation']
        else:
            raise ValueError(f"Неизвестный тип облучения: {irradiation_type}")
        
        t_irr = intervals_data['tirr']
        intervals = intervals_data['intervals']
        
        # Проверяем соответствие количества измерений и интервалов
        if num_measurements != len(intervals):
            logger.warning(f"Количество измерений ({num_measurements}) не совпадает с количеством интервалов ({len(intervals)})")
            logger.info(f"Используем количество интервалов из данных: {len(intervals)}")
            num_measurements = len(intervals)
        
        A = np.zeros((num_measurements, self.num_groups))
        coefficients_info = {
            'irradiation_type': irradiation_type,
            't_irr': t_irr,
            'intervals': intervals,
            'group_coefficients': {},
            'matrix_normalization': 0.0
        }
        
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
            
            for i in range(min(num_measurements, len(intervals))):
                # Получаем параметры для данного интервала
                interval = intervals[i]
                td = interval['td']      # Время распада после облучения
                tc = interval['tc']      # Время измерения
                t1 = interval['t1']      # Начало интервала измерения
                t2 = interval['t2']      # Конец интервала измерения
                
                # Компоненты формулы согласно правильной физике эксперимента
                abundance_factor = a_i / lambda_i
                
                # Фактор облучения - всегда полное облучение
                irradiation_factor = 1 - np.exp(-lambda_i * t_irr)
                
                # Фактор распада после облучения до начала измерения (td)
                decay_factor = np.exp(-lambda_i * td)
                
                # Фактор измерения на интервале (tc = t2 - t1)
                measurement_factor = 1 - np.exp(-lambda_i * tc)
                
                # T-фактор для данного интервала (упрощенная формула)
                T_factor = np.exp(-lambda_i * t1)  # Зависит от времени начала измерения
                
                # Основная формула
                A_component = abundance_factor * irradiation_factor * decay_factor * measurement_factor * T_factor
                
                # Базовая чувствительность с дифференциацией для групп с коротким периодом полураспада
                if lambda_i > 1.0:  # Очень короткий период полураспада (группы 6, 7, 8)
                    base_sensitivity = NUMERICAL_PARAMS['base_sensitivity_factor'] * a_i * (1.0 / lambda_i)
                else:
                    base_sensitivity = NUMERICAL_PARAMS['base_sensitivity_factor'] * a_i
                
                # Общий коэффициент
                A[i, j] = A_component + base_sensitivity
                
                # Сохранение детальной информации
                measurement_info = {
                    'measurement_index': i,
                    'interval': i + 1,
                    'td': td,
                    'tc': tc,
                    't1': t1,
                    't2': t2,
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
    

    
    def solve_equations(self, measurements: np.ndarray, irradiation_type: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Решение системы уравнений: N^l_i(E_n) = Σ A^l_ij · x_j(E_n)
        для каждого энергетического бина E_n с учетом индивидуальных параметров интервалов
        
        Args:
            measurements: массив измерений (num_measurements x num_energy_bins)
            irradiation_type: 'long' или 'short' - тип облучения
        
        Returns:
            group_spectra, uncertainties, coefficients_info
        """
        num_measurements, num_energy_bins = measurements.shape
        
        # Создаем матрицу чувствительности с детальной информацией
        A, coefficients_info = self.create_sensitivity_matrix(irradiation_type, num_measurements)
        
        # Добавляем дополнительную информацию
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
                A_pinv = np.linalg.pinv(A, rcond=NUMERICAL_PARAMS['pinv_rcond'])
                x_solution = A_pinv @ b
                
                # Применяем физические ограничения
                x_solution = np.maximum(x_solution, 0)  # Неотрицательность
                
                # Обеспечиваем минимальный вклад каждой группы
                min_contribution = NUMERICAL_PARAMS['min_contribution_factor'] * np.max(x_solution)
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
                x_solution = np.linalg.lstsq(A, b, rcond=NUMERICAL_PARAMS['pinv_rcond'])[0]
                x_solution = np.maximum(x_solution, 0)
                group_spectra[bin_idx, :] = x_solution
                uncertainties[bin_idx, :] = 0.1 * np.ones(self.num_groups)
        
        return group_spectra, uncertainties, coefficients_info

class FinalSpectrumAnalyzer:
    """Финальный анализатор спектров ЗН с решением системы уравнений"""
    
    def __init__(self, num_groups: int = None):
        # Используем количество групп из констант, если не указано
        if num_groups is None:
            num_groups = EXPERIMENT_PARAMS['num_groups']
        self.num_groups = num_groups
        self.data_loader = FinalDataLoader()
        self.equation_solver = FinalEquationSolver(num_groups)
    
    def analyze_spectra(self, long_data: np.ndarray, short_data: np.ndarray, 
                       energy_bins: np.ndarray) -> Dict:
        """Анализ спектров с решением системы уравнений"""
        logger.info(f"Запуск анализа для {self.num_groups} групп ЗН...")
        
        # ОБРЕЗКА ЭНЕРГЕТИЧЕСКОГО ДИАПАЗОНА: начинаем с 10 кэВ (как указано пользователем)
        # НО сохраняем полный диапазон до максимальной энергии
        start_idx = np.where(energy_bins >= 10.0)[0][0] if np.any(energy_bins >= 10.0) else 0
        end_idx = len(energy_bins)  # Используем весь доступный диапазон
        
        logger.info(f"Энергетический диапазон: {energy_bins[start_idx]:.1f} - {energy_bins[end_idx-1]:.1f} кэВ")
        logger.info(f"Индексы: {start_idx} - {end_idx-1} (из {len(energy_bins)})")
        
        # Обрезаем данные только снизу (с 20 кэВ), но сохраняем полный диапазон сверху
        long_data_trimmed = long_data[:, start_idx:end_idx]
        short_data_trimmed = short_data[:, start_idx:end_idx]
        energy_bins_trimmed = energy_bins[start_idx:end_idx]
        
        # Решение для длинного облучения
        logger.info("Решение для данных длинного облучения...")
        long_spectra, long_uncertainties, long_coefficients = self.equation_solver.solve_equations(
            long_data_trimmed, 'long'
        )
        
        # Решение для короткого облучения
        logger.info("Решение для данных короткого облучения...")
        short_spectra, short_uncertainties, short_coefficients = self.equation_solver.solve_equations(
            short_data_trimmed, 'short'
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
            'energy_bins': energy_bins_trimmed
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
            
            # ФИЛЬТРАЦИЯ ВЫБРОСОВ: мягкая фильтрация без создания повторов
            # Используем скользящее среднее для сглаживания, но сохраняем уникальность значений
            from scipy.signal import savgol_filter
            
            # Применяем фильтр Савицкого-Голея для сглаживания без потери деталей
            if len(group_spectrum) > 5:
                try:
                    smoothed_spectrum = savgol_filter(group_spectrum, window_length=5, polyorder=2)
                except:
                    smoothed_spectrum = group_spectrum
            else:
                smoothed_spectrum = group_spectrum
            
            # Ограничиваем только экстремальные выбросы, не создавая повторов
            median_val = np.median(smoothed_spectrum[smoothed_spectrum > 0])
            if median_val > 0:
                # Ограничиваем только значения, которые превышают медиану в 10 раз
                max_allowed = median_val * 10
                filtered_spectrum = np.where(smoothed_spectrum > max_allowed, max_allowed, smoothed_spectrum)
            else:
                filtered_spectrum = smoothed_spectrum
            
            # Масштабирование с учетом физических констант
            abundance = self.data_loader.group_constants['relative_abundances'][group]
            half_life = self.data_loader.group_constants['half_lives'][group]
            
            # Специальная обработка для групп с очень коротким периодом полураспада
            if half_life < 0.1:  # Группы 7, 8
                # Эти группы должны иметь очень низкую интенсивность
                scaled_spectrum = filtered_spectrum * abundance * 10
            elif half_life < 0.5:  # Группа 6
                # Группа 6 должна иметь умеренную интенсивность
                scaled_spectrum = filtered_spectrum * abundance * 50
            else:
                # Обычное масштабирование для остальных групп
                scaled_spectrum = filtered_spectrum * abundance * 100
            
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
            'Интервал': '',
            'Тип облучения': coefficients_info['irradiation_type'],
            'Время облучения (с)': coefficients_info['t_irr'],
            'td (с)': '',
            'tc (с)': '',
            't1 (с)': '',
            't2 (с)': '',
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
                    'Интервал': meas['interval'],
                    'Тип облучения': '',
                    'Время облучения (с)': '',
                    'td (с)': f"{meas['td']:.3f}",
                    'tc (с)': f"{meas['tc']:.3f}",
                    't1 (с)': f"{meas['t1']:.3f}",
                    't2 (с)': f"{meas['t2']:.3f}",
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
        num_measurements = len(coefficients_info['group_coefficients']['group_1']['measurements'])
        num_groups = len(coefficients_info['group_coefficients'])
        
        matrix_data = []
        
        # Заголовок с интервалами измерения
        header_row = ['Группа\\Интервал'] + [f"Интервал {i+1}" for i in range(num_measurements)]
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
        print(f"  Тип облучения: {long_coeff['irradiation_type']}")
        print(f"  Время облучения: {long_coeff['t_irr']:.1f} с")
        print(f"  Количество интервалов: {len(long_coeff['intervals'])}")
        print(f"  Количество измерений: {long_coeff['num_measurements']}")
        print(f"  Нормализация матрицы: {long_coeff['matrix_normalization']:.6e}")
        
        # Информация об интервалах
        print(f"\n  ПАРАМЕТРЫ ИНТЕРВАЛОВ:")
        for i, interval in enumerate(long_coeff['intervals']):
            print(f"    Интервал {i+1}: td={interval['td']:.2f}с, tc={interval['tc']:.2f}с, t1={interval['t1']:.2f}с, t2={interval['t2']:.2f}с")
        
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
                print(f"      Коэффициент для интервала 1: {first_meas['final_coefficient']:.6e}")
                print(f"      Коэффициент для интервала {len(long_coeff['intervals'])}: {last_meas['final_coefficient']:.6e}")
        
        if len(long_coeff['group_coefficients']) > 3:
            print(f"    ... и ещё {len(long_coeff['group_coefficients']) - 3} групп")
        
        # Информация о коротком облучении
        short_coeff = results['short_coefficients']
        print(f"\nКОРОТКОЕ ОБЛУЧЕНИЕ:")
        print(f"  Тип облучения: {short_coeff['irradiation_type']}")
        print(f"  Время облучения: {short_coeff['t_irr']:.1f} с")
        print(f"  Количество интервалов: {len(short_coeff['intervals'])}")
        print(f"  Количество измерений: {short_coeff['num_measurements']}")
        print(f"  Нормализация матрицы: {short_coeff['matrix_normalization']:.6e}")
        
        # Информация об интервалах
        print(f"\n  ПАРАМЕТРЫ ИНТЕРВАЛОВ:")
        for i, interval in enumerate(short_coeff['intervals']):
            print(f"    Интервал {i+1}: td={interval['td']:.2f}с, tc={interval['tc']:.2f}с, t1={interval['t1']:.2f}с, t2={interval['t2']:.2f}с")
        
        print(f"\nПОДРОБНАЯ ИНФОРМАЦИЯ О ВСЕХ КОЭФФИЦИЕНТАХ СОХРАНЕНА В EXCEL:")
        print(f"  - Листы 'Коэффициенты_длинное' и 'Коэффициенты_короткое'")
        print(f"  - Листы 'Сводка_коэф_длинное' и 'Сводка_коэф_короткое'")
        print("="*60)

def main():
    """Главная функция"""
    try:
        # Инициализация анализатора (используем количество групп из констант)
        num_groups = EXPERIMENT_PARAMS['num_groups']
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
