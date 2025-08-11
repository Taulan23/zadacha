"""
Расширенный анализатор спектров запаздывающих нейтронов
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy.optimize import minimize
from scipy.stats import chi2
import pandas as pd

logger = logging.getLogger(__name__)

class ExtendedDNSpectrumAnalyzer:
    """Расширенный анализатор спектров ЗН с дополнительными методами"""
    
    def __init__(self, data_processor):
        """
        Инициализация расширенного анализатора
        
        Args:
            data_processor: процессор данных ЗН
        """
        self.data_processor = data_processor
        
    def calculate_spectral_parameters(self, spectra: np.ndarray, 
                                    energy_bins: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Вычисление спектральных параметров
        
        Args:
            spectra: спектры групп ЗН
            energy_bins: энергетические бины
            
        Returns:
            Dict: спектральные параметры
        """
        num_groups = spectra.shape[1]
        parameters = {
            'mean_energy': np.zeros(num_groups),
            'rms_energy': np.zeros(num_groups),
            'peak_energy': np.zeros(num_groups),
            'fwhm': np.zeros(num_groups),
            'total_intensity': np.zeros(num_groups)
        }
        
        for group in range(num_groups):
            spectrum = spectra[:, group]
            
            # Общая интенсивность
            total_intensity = np.sum(spectrum)
            parameters['total_intensity'][group] = total_intensity
            
            # Проверяем, есть ли данные в спектре
            if total_intensity <= 0 or np.all(spectrum == 0):
                # Если спектр пустой, устанавливаем разумные значения по умолчанию
                parameters['mean_energy'][group] = 0.0
                parameters['rms_energy'][group] = 0.0
                parameters['peak_energy'][group] = 0.0
                parameters['fwhm'][group] = 0.0
                continue
            
            # Средняя энергия
            parameters['mean_energy'][group] = np.sum(spectrum * energy_bins) / total_intensity
            
            # RMS энергия (исправляем вычисление)
            if total_intensity > 0:
                variance = np.sum(spectrum * (energy_bins - parameters['mean_energy'][group])**2) / total_intensity
                parameters['rms_energy'][group] = np.sqrt(variance) if variance > 0 else 0.0
            else:
                parameters['rms_energy'][group] = 0.0
            
            # Пиковая энергия
            peak_idx = np.argmax(spectrum)
            parameters['peak_energy'][group] = energy_bins[peak_idx]
            
            # FWHM (полная ширина на половине максимума) - улучшенный алгоритм
            max_intensity = np.max(spectrum)
            if max_intensity <= 0:
                parameters['fwhm'][group] = 0.0
                continue
                
            half_max = max_intensity / 2
            
            # Находим индексы, где интенсивность равна половине максимума
            above_half = spectrum >= half_max
            if np.any(above_half):
                # Используем интерполяцию для более точного определения FWHM
                left_indices = np.where(above_half)[0]
                right_indices = np.where(above_half)[0]
                
                if len(left_indices) > 0 and len(right_indices) > 0:
                    left_idx = left_indices[0]
                    right_idx = right_indices[-1]
                    
                    # Интерполяция для более точного определения границ
                    if left_idx > 0:
                        # Линейная интерполяция для левой границы
                        y1, y2 = spectrum[left_idx-1], spectrum[left_idx]
                        x1, x2 = energy_bins[left_idx-1], energy_bins[left_idx]
                        left_energy = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
                    else:
                        left_energy = energy_bins[left_idx]
                    
                    if right_idx < len(energy_bins) - 1:
                        # Линейная интерполяция для правой границы
                        y1, y2 = spectrum[right_idx], spectrum[right_idx+1]
                        x1, x2 = energy_bins[right_idx], energy_bins[right_idx+1]
                        right_energy = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
                    else:
                        right_energy = energy_bins[right_idx]
                    
                    parameters['fwhm'][group] = right_energy - left_energy
                else:
                    parameters['fwhm'][group] = 0.0
            else:
                parameters['fwhm'][group] = 0.0
        
        return parameters
    
    def fit_spectrum_model(self, observed_spectrum: np.ndarray, 
                          energy_bins: np.ndarray,
                          initial_guess: Optional[np.ndarray] = None) -> Dict:
        """
        Подгонка модели спектра к наблюдаемым данным
        
        Args:
            observed_spectrum: наблюдаемый спектр
            energy_bins: энергетические бины
            initial_guess: начальное приближение параметров
            
        Returns:
            Dict: результаты подгонки
        """
        if initial_guess is None:
            # Начальное приближение: экспоненциальный спектр
            initial_guess = np.array([100.0, 200.0, 1.0])  # [A, E0, alpha]
        
        def model_spectrum(params, energy):
            A, E0, alpha = params
            return A * np.exp(-energy / E0) * (energy / E0)**alpha
        
        def chi_square(params):
            model = model_spectrum(params, energy_bins)
            # Добавляем небольшой шум для избежания деления на ноль
            noise = np.max(observed_spectrum) * 0.01
            chi2 = np.sum(((observed_spectrum - model) / (observed_spectrum + noise))**2)
            return chi2
        
        # Минимизация хи-квадрат
        result = minimize(chi_square, initial_guess, method='Nelder-Mead')
        
        fitted_params = result.x
        fitted_spectrum = model_spectrum(fitted_params, energy_bins)
        
        # Вычисление качества подгонки
        residuals = observed_spectrum - fitted_spectrum
        rms_residual = np.sqrt(np.mean(residuals**2))
        chi2_value = chi_square(fitted_params)
        
        return {
            'fitted_params': fitted_params,
            'fitted_spectrum': fitted_spectrum,
            'residuals': residuals,
            'rms_residual': rms_residual,
            'chi2_value': chi2_value,
            'success': result.success
        }
    
    def calculate_correlation_matrix(self, spectra: np.ndarray) -> np.ndarray:
        """
        Вычисление корреляционной матрицы между группами ЗН
        
        Args:
            spectra: спектры групп ЗН
            
        Returns:
            np.ndarray: корреляционная матрица
        """
        num_groups = spectra.shape[1]
        correlation_matrix = np.zeros((num_groups, num_groups))
        
        for i in range(num_groups):
            for j in range(num_groups):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    # Вычисляем корреляцию Пирсона
                    spectrum_i = spectra[:, i]
                    spectrum_j = spectra[:, j]
                    
                    mean_i = np.mean(spectrum_i)
                    mean_j = np.mean(spectrum_j)
                    
                    numerator = np.sum((spectrum_i - mean_i) * (spectrum_j - mean_j))
                    denominator = np.sqrt(np.sum((spectrum_i - mean_i)**2) * 
                                        np.sum((spectrum_j - mean_j)**2))
                    
                    if denominator > 0:
                        correlation_matrix[i, j] = numerator / denominator
                    else:
                        correlation_matrix[i, j] = 0
        
        return correlation_matrix
    
    def analyze_energy_resolution(self, spectra: np.ndarray, 
                                energy_bins: np.ndarray) -> Dict[str, float]:
        """
        Анализ энергетического разрешения спектров
        
        Args:
            spectra: спектры групп ЗН
            energy_bins: энергетические бины
            
        Returns:
            Dict: параметры энергетического разрешения
        """
        resolution_params = {}
        
        # Среднее энергетическое разрешение
        bin_width = energy_bins[1] - energy_bins[0]
        resolution_params['bin_width'] = float(bin_width)
        
        # Эффективное энергетическое разрешение
        total_spectrum = np.sum(spectra, axis=1)
        if np.sum(total_spectrum) > 0:
            mean_energy = np.sum(total_spectrum * energy_bins) / np.sum(total_spectrum)
            variance = np.sum(total_spectrum * (energy_bins - mean_energy)**2) / np.sum(total_spectrum)
            effective_resolution = np.sqrt(variance)
            resolution_params['effective_resolution'] = effective_resolution
        else:
            resolution_params['effective_resolution'] = 0.0
        
        # Разрешение по FWHM
        fwhm_values = []
        for group in range(spectra.shape[1]):
            spectrum = spectra[:, group]
            max_intensity = np.max(spectrum)
            if max_intensity > 0:
                half_max = max_intensity / 2
                above_half = spectrum >= half_max
                if np.any(above_half):
                    left_idx = np.where(above_half)[0][0]
                    right_idx = np.where(above_half)[0][-1]
                    fwhm = energy_bins[right_idx] - energy_bins[left_idx]
                    fwhm_values.append(fwhm)
        
        if fwhm_values:
            resolution_params['mean_fwhm'] = float(np.mean(fwhm_values))
            resolution_params['std_fwhm'] = float(np.std(fwhm_values))
        else:
            resolution_params['mean_fwhm'] = 0.0
            resolution_params['std_fwhm'] = 0.0
        
        return resolution_params
    
    def calculate_uncertainty_propagation(self, spectra: np.ndarray, 
                                        uncertainties: np.ndarray,
                                        energy_bins: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Вычисление распространения неопределенностей
        
        Args:
            spectra: спектры групп ЗН
            uncertainties: неопределенности
            energy_bins: энергетические бины
            
        Returns:
            Dict: результаты распространения неопределенностей
        """
        # Неопределенности для интегральных параметров
        total_spectrum = np.sum(spectra, axis=1)
        total_uncertainty = np.sqrt(np.sum(uncertainties**2, axis=1))
        
        # Неопределенности для средних энергий
        mean_energy_uncertainties = np.zeros(spectra.shape[1])
        for group in range(spectra.shape[1]):
            spectrum = spectra[:, group]
            uncertainty = uncertainties[:, group]
            
            if np.sum(spectrum) > 0:
                mean_energy = np.sum(spectrum * energy_bins) / np.sum(spectrum)
                
                # Частные производные
                d_mean_d_spectrum = energy_bins / np.sum(spectrum)
                d_mean_d_spectrum -= np.sum(spectrum * energy_bins) / (np.sum(spectrum)**2)
                
                # Неопределенность средней энергии
                mean_energy_uncertainty = np.sqrt(np.sum((d_mean_d_spectrum * uncertainty)**2))
                mean_energy_uncertainties[group] = mean_energy_uncertainty
        
        return {
            'total_spectrum_uncertainty': total_uncertainty,
            'mean_energy_uncertainties': mean_energy_uncertainties
        }
    
    def generate_quality_report(self, results: Dict) -> str:
        """
        Генерация отчета о качестве результатов
        
        Args:
            results: результаты анализа
            
        Returns:
            str: текстовый отчет
        """
        report = []
        report.append("="*60)
        report.append("ОТЧЕТ О КАЧЕСТВЕ РЕЗУЛЬТАТОВ АНАЛИЗА СПЕКТРОВ ЗН")
        report.append("="*60)
        
        # Анализ спектральных параметров
        kalman_spectra = results['kalman_spectra']
        energy_bins = results['energy_bins']
        
        spectral_params = self.calculate_spectral_parameters(kalman_spectra, energy_bins)
        
        report.append("\nСПЕКТРАЛЬНЫЕ ПАРАМЕТРЫ:")
        report.append("-" * 40)
        for group in range(8):
            report.append(f"Группа {group+1}:")
            report.append(f"  Средняя энергия: {spectral_params['mean_energy'][group]:.1f} кэВ")
            report.append(f"  RMS энергия: {spectral_params['rms_energy'][group]:.1f} кэВ")
            report.append(f"  Пиковая энергия: {spectral_params['peak_energy'][group]:.1f} кэВ")
            report.append(f"  FWHM: {spectral_params['fwhm'][group]:.1f} кэВ")
            report.append(f"  Общая интенсивность: {spectral_params['total_intensity'][group]:.2f}")
        
        # Анализ энергетического разрешения
        resolution_params = self.analyze_energy_resolution(kalman_spectra, energy_bins)
        
        report.append("\nЭНЕРГЕТИЧЕСКОЕ РАЗРЕШЕНИЕ:")
        report.append("-" * 40)
        report.append(f"Ширина бина: {resolution_params['bin_width']:.1f} кэВ")
        report.append(f"Эффективное разрешение: {resolution_params['effective_resolution']:.1f} кэВ")
        report.append(f"Средний FWHM: {resolution_params['mean_fwhm']:.1f} кэВ")
        report.append(f"Стандартное отклонение FWHM: {resolution_params['std_fwhm']:.1f} кэВ")
        
        # Корреляционный анализ
        correlation_matrix = self.calculate_correlation_matrix(kalman_spectra)
        
        report.append("\nКОРРЕЛЯЦИОННЫЙ АНАЛИЗ:")
        report.append("-" * 40)
        report.append("Корреляционная матрица между группами:")
        for i in range(8):
            row = [f"{correlation_matrix[i, j]:.3f}" for j in range(8)]
            report.append(f"Группа {i+1}: {' '.join(row)}")
        
        # Анализ неопределенностей
        uncertainty_prop = self.calculate_uncertainty_propagation(
            kalman_spectra, results['kalman_uncertainties'], energy_bins
        )
        
        report.append("\nАНАЛИЗ НЕОПРЕДЕЛЕННОСТЕЙ:")
        report.append("-" * 40)
        report.append("Неопределенности средних энергий по группам:")
        for group in range(8):
            report.append(f"Группа {group+1}: {uncertainty_prop['mean_energy_uncertainties'][group]:.3f} кэВ")
        
        report.append("\n" + "="*60)
        report.append("ОТЧЕТ ЗАВЕРШЕН")
        report.append("="*60)
        
        return "\n".join(report)
    
    def save_detailed_results(self, results: Dict, filename: str = "detailed_analysis_results.xlsx"):
        """
        Сохранение детальных результатов анализа
        
        Args:
            results: результаты анализа
            filename: имя файла
        """
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Основные спектры
            kalman_df = pd.DataFrame(results['kalman_spectra'], 
                                   columns=[f'Группа_{i+1}' for i in range(8)])
            kalman_df['Энергия_кэВ'] = results['energy_bins']
            kalman_df.to_excel(writer, sheet_name='Спектры_Калман', index=False)
            
            # Спектральные параметры
            spectral_params = self.calculate_spectral_parameters(
                results['kalman_spectra'], results['energy_bins']
            )
            
            params_df = pd.DataFrame({
                'Группа': [f'Группа_{i+1}' for i in range(8)],
                'Средняя_энергия_кэВ': spectral_params['mean_energy'],
                'RMS_энергия_кэВ': spectral_params['rms_energy'],
                'Пиковая_энергия_кэВ': spectral_params['peak_energy'],
                'FWHM_кэВ': spectral_params['fwhm'],
                'Общая_интенсивность': spectral_params['total_intensity']
            })
            params_df.to_excel(writer, sheet_name='Спектральные_параметры', index=False)
            
            # Корреляционная матрица
            correlation_matrix = self.calculate_correlation_matrix(results['kalman_spectra'])
            corr_df = pd.DataFrame(correlation_matrix, 
                                 columns=[f'Группа_{i+1}' for i in range(8)],
                                 index=[f'Группа_{i+1}' for i in range(8)])
            corr_df.to_excel(writer, sheet_name='Корреляционная_матрица')
            
            # Неопределенности
            uncertainty_prop = self.calculate_uncertainty_propagation(
                results['kalman_spectra'], results['kalman_uncertainties'], results['energy_bins']
            )
            
            unc_df = pd.DataFrame({
                'Группа': [f'Группа_{i+1}' for i in range(8)],
                'Неопределенность_средней_энергии_кэВ': uncertainty_prop['mean_energy_uncertainties']
            })
            unc_df.to_excel(writer, sheet_name='Неопределенности', index=False)
        
        print(f"Детальные результаты сохранены в файл: {filename}")
