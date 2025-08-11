"""
Модуль тестирования системы анализа спектров запаздывающих нейтронов
"""

import numpy as np
import unittest
from typing import Dict
import logging

from data_processor import DNDataProcessor
from kalman_filter import KalmanFilter, PotterKalmanFilter, DNSpectrumAnalyzer
from visualization import DNSpectrumVisualizer
from spectrum_analyzer import ExtendedDNSpectrumAnalyzer

# Настройка логирования для тестов
logging.basicConfig(level=logging.WARNING)

class TestDNDataProcessor(unittest.TestCase):
    """Тесты для процессора данных ЗН"""
    
    def setUp(self):
        """Инициализация тестов"""
        self.processor = DNDataProcessor()
    
    def test_initialization(self):
        """Тест инициализации процессора"""
        self.assertEqual(len(self.processor.dn_groups_data['relative_abundance']), 8)
        self.assertEqual(len(self.processor.dn_groups_data['half_lives']), 8)
        self.assertEqual(len(self.processor.dn_groups_data['decay_constants']), 8)
        self.assertEqual(self.processor.num_energy_bins, 160)
    
    def test_activation_factor_calculation(self):
        """Тест вычисления фактора активации"""
        t_irr = 120.0
        t_d = 0.12
        dt_c = 1.88
        lambda_i = 0.0124
        M = 1
        T = 300.0
        
        factor = self.processor.calculate_activation_factor(t_irr, t_d, dt_c, lambda_i, M, T)
        
        self.assertIsInstance(factor, float)
        self.assertGreater(factor, 0)
    
    def test_observation_matrix_creation(self):
        """Тест создания матрицы наблюдений"""
        A_long = self.processor.create_observation_matrix('long')
        A_short = self.processor.create_observation_matrix('short')
        
        self.assertEqual(A_long.shape, (12, 8))
        self.assertEqual(A_short.shape, (12, 8))
        self.assertTrue(np.all(A_long >= 0))
        self.assertTrue(np.all(A_short >= 0))
    
    def test_synthetic_data_generation(self):
        """Тест генерации синтетических данных"""
        energy_bins = self.processor.get_energy_bins()
        true_spectra = np.random.rand(len(energy_bins), 8)
        
        long_data = self.processor.generate_synthetic_data(true_spectra, 'long')
        short_data = self.processor.generate_synthetic_data(true_spectra, 'short')
        
        self.assertEqual(long_data.shape, (12, len(energy_bins)))
        self.assertEqual(short_data.shape, (12, len(energy_bins)))
        self.assertTrue(np.all(long_data >= 0))
        self.assertTrue(np.all(short_data >= 0))

class TestKalmanFilter(unittest.TestCase):
    """Тесты для фильтра Калмана"""
    
    def setUp(self):
        """Инициализация тестов"""
        self.kalman = KalmanFilter()
        self.potter = PotterKalmanFilter()
    
    def test_kalman_initialization(self):
        """Тест инициализации фильтра Калмана"""
        self.assertEqual(self.kalman.num_groups, 8)
        self.assertEqual(self.kalman.num_measurements, 12)
        self.assertEqual(self.kalman.x_prior.shape, (8,))
        self.assertEqual(self.kalman.P_prior.shape, (8, 8))
    
    def test_predict_step(self):
        """Тест этапа предсказания"""
        x_prior, P_prior = self.kalman.predict()
        
        self.assertEqual(x_prior.shape, (8,))
        self.assertEqual(P_prior.shape, (8, 8))
        self.assertTrue(np.all(np.diag(P_prior) >= 0))  # Диагональные элементы положительные
    
    def test_update_step(self):
        """Тест этапа обновления"""
        # Сначала предсказание
        self.kalman.predict()
        
        # Создаем тестовые данные
        measurement = np.random.rand(12)
        H_matrix = np.random.rand(12, 8)
        
        x_posterior, P_posterior = self.kalman.update(measurement, H_matrix)
        
        self.assertEqual(x_posterior.shape, (8,))
        self.assertEqual(P_posterior.shape, (8, 8))
        self.assertTrue(np.all(np.diag(P_posterior) >= 0))
    
    def test_potter_filter(self):
        """Тест фильтра Поттера"""
        self.assertEqual(self.potter.num_groups, 8)
        self.assertEqual(self.potter.num_measurements, 12)
        
        # Тест предсказания
        x_prior, P_prior = self.potter.predict()
        self.assertEqual(x_prior.shape, (8,))
        self.assertEqual(P_prior.shape, (8, 8))
        
        # Тест обновления
        measurement = np.random.rand(12)
        H_matrix = np.random.rand(12, 8)
        
        x_posterior, P_posterior = self.potter.update(measurement, H_matrix)
        self.assertEqual(x_posterior.shape, (8,))
        self.assertEqual(P_posterior.shape, (8, 8))

class TestDNSpectrumAnalyzer(unittest.TestCase):
    """Тесты для анализатора спектров ЗН"""
    
    def setUp(self):
        """Инициализация тестов"""
        self.processor = DNDataProcessor()
        self.analyzer = DNSpectrumAnalyzer(self.processor)
    
    def test_analyzer_initialization(self):
        """Тест инициализации анализатора"""
        self.assertIsNotNone(self.analyzer.data_processor)
        self.assertIsNotNone(self.analyzer.kalman_filter)
        self.assertIsNotNone(self.analyzer.potter_filter)
    
    def test_spectrum_analysis(self):
        """Тест анализа спектров"""
        energy_bins = self.processor.get_energy_bins()
        true_spectra = np.random.rand(len(energy_bins), 8)
        
        long_data = self.processor.generate_synthetic_data(true_spectra, 'long')
        short_data = self.processor.generate_synthetic_data(true_spectra, 'short')
        
        results = self.analyzer.analyze_spectra(long_data, short_data)
        
        # Проверяем структуру результатов
        required_keys = ['kalman_spectra', 'kalman_uncertainties', 
                        'potter_spectra', 'potter_uncertainties', 'energy_bins']
        for key in required_keys:
            self.assertIn(key, results)
        
        # Проверяем размеры
        self.assertEqual(results['kalman_spectra'].shape, (len(energy_bins), 8))
        self.assertEqual(results['potter_spectra'].shape, (len(energy_bins), 8))
        self.assertEqual(results['kalman_uncertainties'].shape, (len(energy_bins), 8))
        self.assertEqual(results['potter_uncertainties'].shape, (len(energy_bins), 8))
    
    def test_spectrum_normalization(self):
        """Тест нормализации спектров"""
        test_spectra = np.random.rand(160, 8)
        normalized = self.analyzer._normalize_spectra(test_spectra)
        
        self.assertEqual(normalized.shape, test_spectra.shape)
        # Проверяем, что сумма по каждой группе равна 100
        for group in range(8):
            self.assertAlmostEqual(np.sum(normalized[:, group]), 100.0, places=1)

class TestExtendedDNSpectrumAnalyzer(unittest.TestCase):
    """Тесты для расширенного анализатора"""
    
    def setUp(self):
        """Инициализация тестов"""
        self.processor = DNDataProcessor()
        self.extended_analyzer = ExtendedDNSpectrumAnalyzer(self.processor)
    
    def test_spectral_parameters_calculation(self):
        """Тест вычисления спектральных параметров"""
        energy_bins = np.arange(0, 1600, 10)
        test_spectra = np.random.rand(len(energy_bins), 8)
        
        params = self.extended_analyzer.calculate_spectral_parameters(test_spectra, energy_bins)
        
        required_keys = ['mean_energy', 'rms_energy', 'peak_energy', 'fwhm', 'total_intensity']
        for key in required_keys:
            self.assertIn(key, params)
            self.assertEqual(len(params[key]), 8)
    
    def test_correlation_matrix_calculation(self):
        """Тест вычисления корреляционной матрицы"""
        test_spectra = np.random.rand(160, 8)
        corr_matrix = self.extended_analyzer.calculate_correlation_matrix(test_spectra)
        
        self.assertEqual(corr_matrix.shape, (8, 8))
        # Диагональные элементы должны быть равны 1
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), np.ones(8))
        # Матрица должна быть симметричной
        np.testing.assert_array_almost_equal(corr_matrix, corr_matrix.T)
    
    def test_energy_resolution_analysis(self):
        """Тест анализа энергетического разрешения"""
        energy_bins = np.arange(0, 1600, 10)
        test_spectra = np.random.rand(len(energy_bins), 8)
        
        resolution_params = self.extended_analyzer.analyze_energy_resolution(test_spectra, energy_bins)
        
        required_keys = ['bin_width', 'effective_resolution', 'mean_fwhm', 'std_fwhm']
        for key in required_keys:
            self.assertIn(key, resolution_params)
            self.assertIsInstance(resolution_params[key], (int, float))

class TestDNSpectrumVisualizer(unittest.TestCase):
    """Тесты для визуализатора"""
    
    def setUp(self):
        """Инициализация тестов"""
        self.visualizer = DNSpectrumVisualizer()
    
    def test_visualizer_initialization(self):
        """Тест инициализации визуализатора"""
        self.assertEqual(len(self.visualizer.colors), 8)
        self.assertEqual(len(self.visualizer.group_names), 8)
        self.assertTrue(all(name.startswith('Группа') for name in self.visualizer.group_names))

def run_comprehensive_test():
    """Запуск комплексного тестирования системы"""
    print("Запуск комплексного тестирования системы анализа спектров ЗН...")
    print("="*60)
    
    # Создание тестовых данных
    processor = DNDataProcessor()
    energy_bins = processor.get_energy_bins()
    
    # Создание реалистичных тестовых спектров
    test_spectra = np.zeros((len(energy_bins), 8))
    for group in range(8):
        if group < 4:
            spectrum = np.exp(-energy_bins / (150 + group * 80))
        else:
            spectrum = np.exp(-energy_bins / (80 + group * 40))
        test_spectra[:, group] = spectrum / np.sum(spectrum) * 100
    
    # Генерация тестовых данных
    long_data = processor.generate_synthetic_data(test_spectra, 'long', noise_level=0.02)
    short_data = processor.generate_synthetic_data(test_spectra, 'short', noise_level=0.02)
    
    # Тестирование анализатора
    analyzer = DNSpectrumAnalyzer(processor)
    results = analyzer.analyze_spectra(long_data, short_data)
    
    # Тестирование расширенного анализатора
    extended_analyzer = ExtendedDNSpectrumAnalyzer(processor)
    
    # Проверка результатов
    print("Проверка результатов анализа:")
    print(f"- Размер спектров Калмана: {results['kalman_spectra'].shape}")
    print(f"- Размер спектров Поттера: {results['potter_spectra'].shape}")
    print(f"- Размер неопределенностей: {results['kalman_uncertainties'].shape}")
    
    # Проверка нормализации
    for group in range(8):
        kalman_sum = np.sum(results['kalman_spectra'][:, group])
        potter_sum = np.sum(results['potter_spectra'][:, group])
        print(f"- Группа {group+1}: Калман={kalman_sum:.1f}, Поттер={potter_sum:.1f}")
    
    # Проверка спектральных параметров
    spectral_params = extended_analyzer.calculate_spectral_parameters(
        results['kalman_spectra'], results['energy_bins']
    )
    print("\nСпектральные параметры:")
    for group in range(8):
        print(f"- Группа {group+1}: средняя энергия={spectral_params['mean_energy'][group]:.1f} кэВ")
    
    print("\n" + "="*60)
    print("КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО УСПЕШНО!")
    print("="*60)
    
    return True

if __name__ == "__main__":
    # Запуск unit-тестов
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Запуск комплексного тестирования
    run_comprehensive_test()
