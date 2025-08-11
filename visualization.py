"""
Модуль для визуализации результатов анализа спектров запаздывающих нейтронов
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional
import pandas as pd

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DNSpectrumVisualizer:
    """Класс для визуализации спектров ЗН"""
    
    def __init__(self):
        """Инициализация визуализатора"""
        self.colors = plt.cm.Set3(np.linspace(0, 1, 8))
        self.group_names = [f'Группа {i+1}' for i in range(8)]
        
    def plot_individual_spectra(self, spectra: np.ndarray, uncertainties: np.ndarray,
                              energy_bins: np.ndarray, title: str = "Спектры групп ЗН",
                              save_path: Optional[str] = None):
        """
        Построение индивидуальных спектров для каждой группы ЗН
        
        Args:
            spectra: спектры групп ЗН
            uncertainties: неопределенности
            energy_bins: энергетические бины
            title: заголовок графика
            save_path: путь для сохранения
        """
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for group in range(8):
            ax = axes[group]
            
            # Основной спектр
            ax.plot(energy_bins, spectra[:, group], 
                   color=self.colors[group], linewidth=2, label=self.group_names[group])
            
            # Область неопределенности
            ax.fill_between(energy_bins, 
                          spectra[:, group] - uncertainties[:, group],
                          spectra[:, group] + uncertainties[:, group],
                          alpha=0.3, color=self.colors[group])
            
            ax.set_xlabel('Энергия (кэВ)')
            ax.set_ylabel('Интенсивность (норм. на 100)')
            ax.set_title(f'{self.group_names[group]}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Установка пределов по оси Y
            max_val = np.max(spectra[:, group])
            if np.isfinite(max_val) and max_val > 0:
                ax.set_ylim(0, max_val * 1.1)
            else:
                ax.set_ylim(0, 1.0)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_comparison_spectra(self, kalman_spectra: np.ndarray, potter_spectra: np.ndarray,
                              jeff_spectra: Dict[str, np.ndarray], energy_bins: np.ndarray,
                              save_path: Optional[str] = None):
        """
        Сравнение спектров, полученных разными методами
        
        Args:
            kalman_spectra: спектры фильтра Калмана
            potter_spectra: спектры алгоритма Поттера
            jeff_spectra: спектры JEFF-3.1.1
            energy_bins: энергетические бины
            save_path: путь для сохранения
        """
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for group in range(8):
            ax = axes[group]
            group_name = f'group_{group+1}'
            
            # Спектр Калмана
            ax.plot(energy_bins, kalman_spectra[:, group], 
                   color='blue', linewidth=2, label='Калман', alpha=0.8)
            
            # Спектр Поттера
            ax.plot(energy_bins, potter_spectra[:, group], 
                   color='red', linewidth=2, label='Поттер', alpha=0.8)
            
            # Спектр JEFF
            if group_name in jeff_spectra:
                jeff_data = jeff_spectra[group_name]
                # Преобразуем в числовой формат
                if isinstance(jeff_data, (list, np.ndarray)):
                    jeff_data = np.array(jeff_data, dtype=np.float64)
                    # Обрезаем до нужной длины
                    if len(jeff_data) > len(energy_bins):
                        jeff_data = jeff_data[:len(energy_bins)]
                    elif len(jeff_data) < len(energy_bins):
                        jeff_data = np.pad(jeff_data, (0, len(energy_bins) - len(jeff_data)), 'constant')
                    
                    ax.plot(energy_bins, jeff_data, 
                           color='green', linewidth=2, label='JEFF-3.1.1', alpha=0.8)
            
            ax.set_xlabel('Энергия (кэВ)')
            ax.set_ylabel('Интенсивность (норм. на 100)')
            ax.set_title(f'{self.group_names[group]}')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.suptitle('Сравнение спектров ЗН: Калман vs Поттер vs JEFF-3.1.1', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_energy_distribution(self, spectra: np.ndarray, energy_bins: np.ndarray,
                               title: str = "Распределение по энергиям",
                               save_path: Optional[str] = None):
        """
        Построение распределения по энергиям для всех групп
        
        Args:
            spectra: спектры групп ЗН
            energy_bins: энергетические бины
            title: заголовок графика
            save_path: путь для сохранения
        """
        plt.figure(figsize=(12, 8))
        
        for group in range(8):
            plt.plot(energy_bins, spectra[:, group], 
                    color=self.colors[group], linewidth=2, 
                    label=self.group_names[group])
        
        plt.xlabel('Энергия (кэВ)')
        plt.ylabel('Интенсивность (норм. на 100)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(0, 1600)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_uncertainty_analysis(self, kalman_uncertainties: np.ndarray, 
                                potter_uncertainties: np.ndarray,
                                energy_bins: np.ndarray,
                                save_path: Optional[str] = None):
        """
        Анализ неопределенностей
        
        Args:
            kalman_uncertainties: неопределенности Калмана
            potter_uncertainties: неопределенности Поттера
            energy_bins: энергетические бины
            save_path: путь для сохранения
        """
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for group in range(8):
            ax = axes[group]
            
            # Неопределенности Калмана
            ax.plot(energy_bins, kalman_uncertainties[:, group], 
                   color='blue', linewidth=2, label='Калман', alpha=0.8)
            
            # Неопределенности Поттера
            ax.plot(energy_bins, potter_uncertainties[:, group], 
                   color='red', linewidth=2, label='Поттер', alpha=0.8)
            
            ax.set_xlabel('Энергия (кэВ)')
            ax.set_ylabel('Неопределенность')
            ax.set_title(f'{self.group_names[group]}')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.suptitle('Анализ неопределенностей: Калман vs Поттер', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_comparison_statistics(self, comparison_results: Dict[str, float],
                                 save_path: Optional[str] = None):
        """
        Построение статистики сравнения с JEFF
        
        Args:
            comparison_results: результаты сравнения
            save_path: путь для сохранения
        """
        groups = list(comparison_results.keys())
        values = list(comparison_results.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(groups, values, color=self.colors[:len(groups)])
        
        plt.xlabel('Группы ЗН')
        plt.ylabel('Относительная разность (%)')
        plt.title('Сравнение с данными JEFF-3.1.1')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Добавление значений на столбцы
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_summary_report(self, results: Dict, save_path: Optional[str] = None):
        """
        Создание сводного отчета
        
        Args:
            results: результаты анализа
            save_path: путь для сохранения
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Создаем сетку графиков
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # График 1: Общий спектр Калмана
        ax1 = fig.add_subplot(gs[0, :2])
        energy_bins = results['energy_bins']
        for group in range(8):
            ax1.plot(energy_bins, results['kalman_spectra'][:, group], 
                    color=self.colors[group], linewidth=1.5, 
                    label=self.group_names[group], alpha=0.8)
        ax1.set_xlabel('Энергия (кэВ)')
        ax1.set_ylabel('Интенсивность')
        ax1.set_title('Спектры ЗН (Калман)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # График 2: Сравнение методов для группы 1
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(energy_bins, results['kalman_spectra'][:, 0], 
                color='blue', label='Калман', linewidth=2)
        ax2.plot(energy_bins, results['potter_spectra'][:, 0], 
                color='red', label='Поттер', linewidth=2)
        ax2.set_xlabel('Энергия (кэВ)')
        ax2.set_ylabel('Интенсивность')
        ax2.set_title('Группа 1: Калман vs Поттер')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # График 3: Неопределенности
        ax3 = fig.add_subplot(gs[1, :])
        for group in range(8):
            ax3.plot(energy_bins, results['kalman_uncertainties'][:, group], 
                    color=self.colors[group], linewidth=1.5, 
                    label=self.group_names[group], alpha=0.8)
        ax3.set_xlabel('Энергия (кэВ)')
        ax3.set_ylabel('Неопределенность')
        ax3.set_title('Неопределенности (Калман)')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # График 4: Статистика групп
        ax4 = fig.add_subplot(gs[2, :])
        group_means = np.mean(results['kalman_spectra'], axis=0)
        group_stds = np.std(results['kalman_spectra'], axis=0)
        
        bars = ax4.bar(self.group_names, group_means, 
                      yerr=group_stds, capsize=5, color=self.colors)
        ax4.set_xlabel('Группы ЗН')
        ax4.set_ylabel('Средняя интенсивность')
        ax4.set_title('Статистика по группам')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Сводный отчет: Анализ спектров ЗН с использованием фильтра Калмана', 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_results_to_excel(self, results: Dict, filename: str = "dn_analysis_results.xlsx"):
        """
        Сохранение результатов в Excel файл
        
        Args:
            results: результаты анализа
            filename: имя файла
        """
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Спектры Калмана
            kalman_df = pd.DataFrame(results['kalman_spectra'], 
                                   columns=[f'Группа_{i+1}' for i in range(8)])
            kalman_df['Энергия_кэВ'] = results['energy_bins']
            kalman_df.to_excel(writer, sheet_name='Спектры_Калман', index=False)
            
            # Спектры Поттера
            potter_df = pd.DataFrame(results['potter_spectra'], 
                                   columns=[f'Группа_{i+1}' for i in range(8)])
            potter_df['Энергия_кэВ'] = results['energy_bins']
            potter_df.to_excel(writer, sheet_name='Спектры_Поттер', index=False)
            
            # Неопределенности Калмана
            kalman_unc_df = pd.DataFrame(results['kalman_uncertainties'], 
                                       columns=[f'Группа_{i+1}' for i in range(8)])
            kalman_unc_df['Энергия_кэВ'] = results['energy_bins']
            kalman_unc_df.to_excel(writer, sheet_name='Неопределенности_Калман', index=False)
            
            # Неопределенности Поттера
            potter_unc_df = pd.DataFrame(results['potter_uncertainties'], 
                                       columns=[f'Группа_{i+1}' for i in range(8)])
            potter_unc_df['Энергия_кэВ'] = results['energy_bins']
            potter_unc_df.to_excel(writer, sheet_name='Неопределенности_Поттер', index=False)
        
        print(f"Результаты сохранены в файл: {filename}")
