"""
Упрощенный модуль для анализа энергетических спектров запаздывающих нейтронов
Только стандартный фильтр Калмана, без графиков и фильтра Поттера
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple
import os
from datetime import datetime

from simple_data_loader import SimpleDataLoader
from simple_kalman import SimpleDNSpectrumAnalyzer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleSpectrumAnalyzer:
    """Упрощенный анализатор спектров только с фильтром Калмана"""
    
    def __init__(self):
        self.data_loader = SimpleDataLoader()
        self.kalman_analyzer = SimpleDNSpectrumAnalyzer()
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Загрузка данных из Excel файлов"""
        logger.info("Загрузка реальных данных из Excel файлов...")
        
        # Загружаем данные облучения
        long_data, short_data, energy_bins = self.data_loader.load_integral_data()
        
        # Загружаем групповые спектры (если есть)
        group_spectra = self.data_loader.load_group_spectra()
        
        return long_data, short_data, group_spectra
    
    def analyze_with_kalman(self, long_data: np.ndarray, short_data: np.ndarray) -> Dict:
        """Анализ данных только с фильтром Калмана"""
        logger.info("Запуск анализа с использованием фильтра Калмана...")
        
        # Получаем энергетические бины
        energy_bins = np.linspace(10, 1600, 160)
        
        # Анализируем данные длинного облучения
        logger.info("Анализ данных длинного облучения...")
        kalman_long = self.kalman_analyzer.analyze_spectra(long_data, energy_bins)
        
        # Анализируем данные короткого облучения
        logger.info("Анализ данных короткого облучения...")
        kalman_short = self.kalman_analyzer.analyze_spectra(short_data, energy_bins)
        
        return {
            'long_irradiation': kalman_long,
            'short_irradiation': kalman_short,
            'energy_bins': energy_bins
        }
    
    def calculate_spectral_parameters(self, spectra: np.ndarray, energy_bins: np.ndarray) -> Dict:
        """Расчет спектральных параметров"""
        num_groups = spectra.shape[1]
        parameters = {}
        
        for group in range(num_groups):
            spectrum = spectra[:, group]
            
            # Средняя энергия
            mean_energy = np.average(energy_bins, weights=spectrum)
            
            # RMS энергия
            variance = np.average((energy_bins - mean_energy)**2, weights=spectrum)
            rms_energy = np.sqrt(variance)
            
            # Пиковая энергия
            peak_energy = energy_bins[np.argmax(spectrum)]
            
            # FWHM (приблизительно)
            max_intensity = np.max(spectrum)
            half_max = max_intensity / 2
            above_half = spectrum > half_max
            if np.any(above_half):
                fwhm = energy_bins[above_half][-1] - energy_bins[above_half][0]
            else:
                fwhm = 0
            
            # Общая интенсивность
            total_intensity = np.sum(spectrum)
            
            parameters[f'group_{group+1}'] = {
                'mean_energy': mean_energy,
                'rms_energy': rms_energy,
                'peak_energy': peak_energy,
                'fwhm': fwhm,
                'total_intensity': total_intensity,
                'spectrum': spectrum
            }
        
        return parameters
    
    def save_results(self, results: Dict, filename: str = None):
        """Сохранение результатов в Excel"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/precise_data_{timestamp}.xlsx"
        
        os.makedirs('results', exist_ok=True)
        
        # Создаем Excel файл с несколькими листами
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            
            # Лист 1: Спектральные параметры длинного облучения
            long_params = self.calculate_spectral_parameters(
                results['long_irradiation'], results['energy_bins']
            )
            
            long_data = []
            for group_name, params in long_params.items():
                long_data.append({
                    'Группа': group_name,
                    'Средняя энергия (кэВ)': params['mean_energy'],
                    'RMS энергия (кэВ)': params['rms_energy'],
                    'Пиковая энергия (кэВ)': params['peak_energy'],
                    'FWHM (кэВ)': params['fwhm'],
                    'Общая интенсивность': params['total_intensity']
                })
            
            pd.DataFrame(long_data).to_excel(writer, sheet_name='Длинное облучение', index=False)
            
            # Лист 2: Спектральные параметры короткого облучения
            short_params = self.calculate_spectral_parameters(
                results['short_irradiation'], results['energy_bins']
            )
            
            short_data = []
            for group_name, params in short_params.items():
                short_data.append({
                    'Группа': group_name,
                    'Средняя энергия (кэВ)': params['mean_energy'],
                    'RMS энергия (кэВ)': params['rms_energy'],
                    'Пиковая энергия (кэВ)': params['peak_energy'],
                    'FWHM (кэВ)': params['fwhm'],
                    'Общая интенсивность': params['total_intensity']
                })
            
            pd.DataFrame(short_data).to_excel(writer, sheet_name='Короткое облучение', index=False)
            
            # Лист 3: Сырые спектральные данные длинного облучения
            long_spectra_df = pd.DataFrame(
                results['long_irradiation'],
                columns=[f'Группа_{i+1}' for i in range(results['long_irradiation'].shape[1])],
                index=results['energy_bins']
            )
            long_spectra_df.index.name = 'Энергия (кэВ)'
            long_spectra_df.to_excel(writer, sheet_name='Спектры_длинное_облучение')
            
            # Лист 4: Сырые спектральные данные короткого облучения
            short_spectra_df = pd.DataFrame(
                results['short_irradiation'],
                columns=[f'Группа_{i+1}' for i in range(results['short_irradiation'].shape[1])],
                index=results['energy_bins']
            )
            short_spectra_df.index.name = 'Энергия (кэВ)'
            short_spectra_df.to_excel(writer, sheet_name='Спектры_короткое_облучение')
            
            # Лист 5: Энергетические бины
            energy_df = pd.DataFrame({
                'Энергия (кэВ)': results['energy_bins']
            })
            energy_df.to_excel(writer, sheet_name='Энергетические_бины', index=False)
        
        logger.info(f"Результаты сохранены в файл: {filename}")
        return filename
    
    def print_summary(self, results: Dict):
        """Вывод краткого отчета"""
        print("\n" + "="*60)
        print("ОТЧЕТ О РЕЗУЛЬТАТАХ АНАЛИЗА СПЕКТРОВ ЗН")
        print("="*60)
        
        # Параметры длинного облучения
        long_params = self.calculate_spectral_parameters(
            results['long_irradiation'], results['energy_bins']
        )
        
        print("\nСПЕКТРАЛЬНЫЕ ПАРАМЕТРЫ (ДЛИННОЕ ОБЛУЧЕНИЕ):")
        print("-" * 50)
        for group_name, params in long_params.items():
            print(f"{group_name}:")
            print(f"  Средняя энергия: {params['mean_energy']:.1f} кэВ")
            print(f"  RMS энергия: {params['rms_energy']:.1f} кэВ")
            print(f"  Пиковая энергия: {params['peak_energy']:.1f} кэВ")
            print(f"  FWHM: {params['fwhm']:.1f} кэВ")
            print(f"  Общая интенсивность: {params['total_intensity']:.2f}")
            print()
        
        print("="*60)
        print("АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
        print("="*60)

def main():
    """Главная функция"""
    try:
        # Инициализация анализатора
        analyzer = SimpleSpectrumAnalyzer()
        
        # Загрузка данных
        long_data, short_data, group_spectra = analyzer.load_data()
        
        # Проверка качества данных
        logger.info("Проверка качества данных...")
        logger.info(f"Энергетический диапазон: 10.0-1600.0 кэВ")
        logger.info(f"Количество энергетических бинов: {long_data.shape[1]}")
        logger.info(f"Данные длинного облучения: {long_data.shape}")
        logger.info(f"Данные короткого облучения: {short_data.shape}")
        
        # Анализ с фильтром Калмана
        results = analyzer.analyze_with_kalman(long_data, short_data)
        
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
