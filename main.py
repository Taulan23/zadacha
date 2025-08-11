"""
Главный модуль для анализа энергетических спектров запаздывающих нейтронов
с использованием фильтра Калмана
"""

import numpy as np
import logging
from typing import Dict
import os

from data_processor import DNDataProcessor
from kalman_filter import DNSpectrumAnalyzer
from visualization import DNSpectrumVisualizer
from spectrum_analyzer import ExtendedDNSpectrumAnalyzer
from simple_data_loader import SimpleDataLoader

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_synthetic_true_spectra(energy_bins: np.ndarray) -> np.ndarray:
    """
    Создание синтетических истинных спектров для тестирования
    
    Args:
        energy_bins: энергетические бины
        
    Returns:
        np.ndarray: истинные спектры групп ЗН
    """
    num_bins = len(energy_bins)
    true_spectra = np.zeros((num_bins, 8))
    
    # Параметры для каждой группы ЗН (на основе литературы по ЗН)
    group_params = [
        {'mean_energy': 250, 'sigma': 80, 'amplitude': 100, 'decay_time': 55.6},    # Группа 1
        {'mean_energy': 460, 'sigma': 120, 'amplitude': 80, 'decay_time': 22.7},    # Группа 2
        {'mean_energy': 300, 'sigma': 100, 'amplitude': 60, 'decay_time': 6.22},    # Группа 3
        {'mean_energy': 550, 'sigma': 150, 'amplitude': 40, 'decay_time': 2.30},    # Группа 4
        {'mean_energy': 420, 'sigma': 110, 'amplitude': 70, 'decay_time': 0.610},   # Группа 5
        {'mean_energy': 180, 'sigma': 60, 'amplitude': 90, 'decay_time': 0.230},    # Группа 6
        {'mean_energy': 380, 'sigma': 95, 'amplitude': 50, 'decay_time': 0.052},    # Группа 7
        {'mean_energy': 320, 'sigma': 85, 'amplitude': 30, 'decay_time': 0.017}     # Группа 8
    ]
    
    # Создаем реалистичные спектры для каждой группы
    for group in range(8):
        params = group_params[group]
        
        # Основной пик (гауссов)
        spectrum = params['amplitude'] * np.exp(-((energy_bins - params['mean_energy']) / params['sigma']) ** 2)
        
        # Добавляем экспоненциальный хвост для высоких энергий (характерно для ЗН)
        tail = 0.15 * params['amplitude'] * np.exp(-energy_bins / 600)
        spectrum += tail
        
        # Добавляем низкоэнергетический компонент для некоторых групп
        if group in [0, 2, 5]:  # Группы с низкоэнергетическими пиками
            low_energy = 0.25 * params['amplitude'] * np.exp(-((energy_bins - 50) / 25) ** 2)
            spectrum += low_energy
        
        # Добавляем второй пик для некоторых групп (характерно для ЗН)
        if group in [1, 4, 7]:
            second_peak = 0.3 * params['amplitude'] * np.exp(-((energy_bins - params['mean_energy'] * 1.5) / params['sigma']) ** 2)
            spectrum += second_peak
        
        # Добавляем случайные флуктуации для реалистичности (уменьшаем шум)
        noise = np.random.normal(0, 0.01 * params['amplitude'], num_bins)
        spectrum += noise
        
        # Убираем отрицательные значения
        spectrum = np.maximum(spectrum, 0)
        
        # Нормализуем
        if np.sum(spectrum) > 0:
            spectrum = spectrum / np.sum(spectrum) * 100
        
        true_spectra[:, group] = spectrum
    
    return true_spectra

def create_synthetic_measurements(true_spectra: np.ndarray, num_measurements: int = 18) -> tuple:
    """
    Создание синтетических измерений на основе истинных спектров
    
    Args:
        true_spectra: истинные спектры групп ЗН
        num_measurements: количество измерений
        
    Returns:
        tuple: (long_irradiation_data, short_irradiation_data)
    """
    num_bins, num_groups = true_spectra.shape
    
    # Создаем матрицу чувствительности (H-матрица) с более реалистичными значениями
    # Каждое измерение имеет разную чувствительность к разным группам
    np.random.seed(42)  # Для воспроизводимости результатов
    
    # Создаем базовую матрицу чувствительности
    H_matrix = np.zeros((num_measurements, num_groups))
    
    # Первые 6 измерений - длинное облучение (120s)
    for i in range(6):
        # Длинное облучение более чувствительно к группам с большими временами жизни
        weights = np.array([0.25, 0.20, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02])  # Убывающие веса
        H_matrix[i, :] = weights * np.random.uniform(0.8, 1.2, num_groups)
    
    # Следующие 6 измерений - среднее облучение (60s)
    for i in range(6, 12):
        # Среднее облучение - равномерная чувствительность
        weights = np.array([0.15, 0.18, 0.20, 0.15, 0.12, 0.08, 0.06, 0.06])
        H_matrix[i, :] = weights * np.random.uniform(0.7, 1.3, num_groups)
    
    # Последние 6 измерений - короткое облучение (20s)
    for i in range(12, 18):
        # Короткое облучение более чувствительно к группам с малыми временами жизни
        weights = np.array([0.02, 0.04, 0.06, 0.08, 0.12, 0.15, 0.20, 0.25])  # Возрастающие веса
        H_matrix[i, :] = weights * np.random.uniform(0.6, 1.4, num_groups)
    
    # Нормализуем строки матрицы H
    H_matrix = H_matrix / np.sum(H_matrix, axis=1, keepdims=True)
    
    # Создаем измерения для длинного облучения
    long_measurements = np.zeros((num_measurements, num_bins))
    for i in range(num_measurements):
        # Линейная комбинация групповых спектров
        measurement = np.zeros(num_bins)
        for j in range(num_groups):
            measurement += H_matrix[i, j] * true_spectra[:, j]
        
        # Добавляем шум измерений (уменьшаем шум)
        noise = np.random.normal(0, 0.02 * np.max(measurement), num_bins)
        measurement += noise
        measurement = np.maximum(measurement, 0)
        
        long_measurements[i, :] = measurement
    
    # Создаем измерения для короткого облучения (с другими весами)
    short_measurements = np.zeros((num_measurements, num_bins))
    
    # Для короткого облучения используем обратные веса
    H_matrix_short = H_matrix.copy()
    for i in range(num_measurements):
        if i < 6:  # Длинное облучение
            H_matrix_short[i, :] = H_matrix[i, :] * np.random.uniform(0.3, 0.7, num_groups)
        elif i < 12:  # Среднее облучение
            H_matrix_short[i, :] = H_matrix[i, :] * np.random.uniform(0.5, 0.9, num_groups)
        else:  # Короткое облучение
            H_matrix_short[i, :] = H_matrix[i, :] * np.random.uniform(0.8, 1.2, num_groups)
    
    H_matrix_short = H_matrix_short / np.sum(H_matrix_short, axis=1, keepdims=True)
    
    for i in range(num_measurements):
        measurement = np.zeros(num_bins)
        for j in range(num_groups):
            measurement += H_matrix_short[i, j] * true_spectra[:, j]
        
        noise = np.random.normal(0, 0.02 * np.max(measurement), num_bins)
        measurement += noise
        measurement = np.maximum(measurement, 0)
        
        short_measurements[i, :] = measurement
    
    return long_measurements.T, short_measurements.T

def main():
    """Главная функция для запуска анализа"""
    
    logger.info("Запуск анализа спектров запаздывающих нейтронов")
    
    # Загрузка реальных данных из Excel файлов
    logger.info("Загрузка реальных данных из Excel файлов...")
    data_loader = SimpleDataLoader()
    all_data = data_loader.load_all_data()
    
    # Извлечение данных
    long_irradiation_data = all_data['long_irradiation_data']
    short_irradiation_data = all_data['short_irradiation_data']
    energy_bins = all_data['energy_bins']
    group_spectra = all_data['group_spectra']
    
    # Преобразуем в numpy массивы с правильным типом
    long_irradiation_data = np.array(long_irradiation_data, dtype=np.float64)
    short_irradiation_data = np.array(short_irradiation_data, dtype=np.float64)
    
    # Проверка качества данных
    logger.info("Проверка качества данных...")
    if np.any(np.isnan(long_irradiation_data)) or np.any(np.isnan(short_irradiation_data)):
        logger.warning("Обнаружены NaN значения в данных, заменяем на нули")
        long_irradiation_data = np.nan_to_num(long_irradiation_data, nan=0.0)
        short_irradiation_data = np.nan_to_num(short_irradiation_data, nan=0.0)
    
    if np.any(long_irradiation_data < 0) or np.any(short_irradiation_data < 0):
        logger.warning("Обнаружены отрицательные значения, заменяем на абсолютные")
        long_irradiation_data = np.abs(long_irradiation_data)
        short_irradiation_data = np.abs(short_irradiation_data)
    
    logger.info(f"Энергетический диапазон: {energy_bins[0]}-{energy_bins[-1]} кэВ")
    logger.info(f"Количество энергетических бинов: {len(energy_bins)}")
    logger.info(f"Данные длинного облучения: {long_irradiation_data.shape}")
    logger.info(f"Данные короткого облучения: {short_irradiation_data.shape}")
    
    # Создание процессора данных
    logger.info("Инициализация процессора данных...")
    data_processor = DNDataProcessor()
    
    # Создание анализатора спектров
    logger.info("Инициализация анализатора спектров...")
    spectrum_analyzer = DNSpectrumAnalyzer(data_processor)
    extended_analyzer = ExtendedDNSpectrumAnalyzer(data_processor)
    
    # Анализ спектров с использованием фильтра Калмана
    logger.info("Запуск анализа с использованием фильтра Калмана...")
    results = spectrum_analyzer.analyze_spectra(long_irradiation_data, short_irradiation_data)
    
    # Добавляем энергетические бины в результаты
    results['energy_bins'] = energy_bins
    
    # Загрузка данных JEFF для сравнения
    logger.info("Загрузка данных JEFF-3.1.1...")
    jeff_spectra = data_processor.load_jeff_data()
    
    # Создание визуализатора
    logger.info("Инициализация визуализатора...")
    visualizer = DNSpectrumVisualizer()
    
    # Создание папки для результатов
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Визуализация результатов
    logger.info("Создание визуализаций...")
    
    # 1. Индивидуальные спектры Калмана
    visualizer.plot_individual_spectra(
        results['kalman_spectra'],
        results['kalman_uncertainties'],
        results['energy_bins'],
        title="Спектры групп ЗН (фильтр Калмана) - Реальные данные",
        save_path=f"{results_dir}/kalman_individual_spectra.png"
    )
    
    # 2. Индивидуальные спектры Поттера
    visualizer.plot_individual_spectra(
        results['potter_spectra'],
        results['potter_uncertainties'],
        results['energy_bins'],
        title="Спектры групп ЗН (алгоритм Поттера) - Реальные данные",
        save_path=f"{results_dir}/potter_individual_spectra.png"
    )
    
    # 3. Сравнение методов
    visualizer.plot_comparison_spectra(
        results['kalman_spectra'],
        results['potter_spectra'],
        jeff_spectra,
        results['energy_bins'],
        save_path=f"{results_dir}/comparison_spectra.png"
    )
    
    # 4. Распределение по энергиям
    visualizer.plot_energy_distribution(
        results['kalman_spectra'],
        results['energy_bins'],
        title="Распределение по энергиям (Калман) - Реальные данные",
        save_path=f"{results_dir}/energy_distribution.png"
    )
    
    # 5. Анализ неопределенностей
    visualizer.plot_uncertainty_analysis(
        results['kalman_uncertainties'],
        results['potter_uncertainties'],
        results['energy_bins'],
        save_path=f"{results_dir}/uncertainty_analysis.png"
    )
    
    # 6. Сравнение с JEFF
    comparison_results = spectrum_analyzer.compare_with_jeff(
        results['kalman_spectra'], jeff_spectra
    )
    visualizer.plot_comparison_statistics(
        comparison_results,
        save_path=f"{results_dir}/jeff_comparison.png"
    )
    
    # 7. Сводный отчет
    visualizer.create_summary_report(
        results,
        save_path=f"{results_dir}/summary_report.png"
    )
    
    # Сохранение результатов в Excel
    logger.info("Сохранение результатов в Excel...")
    visualizer.save_results_to_excel(results, f"{results_dir}/dn_analysis_results.xlsx")
    
    # Расширенный анализ
    logger.info("Выполнение расширенного анализа...")
    extended_analyzer.save_detailed_results(results, f"{results_dir}/detailed_analysis_results.xlsx")
    
    # Генерация отчета о качестве
    quality_report = extended_analyzer.generate_quality_report(results)
    with open(f"{results_dir}/quality_report.txt", 'w', encoding='utf-8') as f:
        f.write(quality_report)
    print("\n" + quality_report)
    
    # Вывод статистики
    logger.info("Вывод статистики результатов...")
    print("\n" + "="*60)
    print("СТАТИСТИКА РЕЗУЛЬТАТОВ АНАЛИЗА СПЕКТРОВ ЗН (РЕАЛЬНЫЕ ДАННЫЕ)")
    print("="*60)
    
    # Статистика по группам
    print("\nСредние значения интенсивности по группам (Калман):")
    for group in range(8):
        mean_intensity = np.mean(results['kalman_spectra'][:, group])
        std_intensity = np.std(results['kalman_spectra'][:, group])
        print(f"Группа {group+1}: {mean_intensity:.2f} ± {std_intensity:.2f}")
    
    # Сравнение с JEFF
    print("\nСравнение с данными JEFF-3.1.1:")
    for group_name, relative_diff in comparison_results.items():
        print(f"{group_name}: {relative_diff:.1f}%")
    
    # Средние неопределенности
    print("\nСредние неопределенности по группам (Калман):")
    for group in range(8):
        mean_uncertainty = np.mean(results['kalman_uncertainties'][:, group])
        print(f"Группа {group+1}: {mean_uncertainty:.3f}")
    
    # Сравнение методов
    print("\nСравнение методов Калман vs Поттер:")
    for group in range(8):
        kalman_mean = np.mean(results['kalman_spectra'][:, group])
        potter_mean = np.mean(results['potter_spectra'][:, group])
        if kalman_mean > 0:
            diff_percent = abs(kalman_mean - potter_mean) / kalman_mean * 100
            print(f"Группа {group+1}: разность {diff_percent:.1f}%")
        else:
            print(f"Группа {group+1}: нет данных")
    
    print("\n" + "="*60)
    print("АНАЛИЗ РЕАЛЬНЫХ ДАННЫХ ЗАВЕРШЕН УСПЕШНО!")
    print(f"Результаты сохранены в папке: {results_dir}")
    print("="*60)
    
    return results

if __name__ == "__main__":
    try:
        results = main()
        logger.info("Анализ завершен успешно")
    except Exception as e:
        logger.error(f"Ошибка при выполнении анализа: {e}")
        raise
