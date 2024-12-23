import numpy as np
from django.shortcuts import render
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')  # Включаем "безголовый" режим
import matplotlib.pyplot as plt
import io
import base64

# Функция для вычисления суммы квадратов отклонений
def sum_of_squares(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

# Гиперболическая функция: y = c / x
def hyperbolic_model(x, c):
    return c / x

# Степенная функция: y = a * x^b
def power_model(x, a, b):
    return a * x**b

# Список таблиц с исходными данными
tables = [
    {
        'name': 'Кубическая',
        'x': np.array([3, 3.3, 3.6, 3.9, 4.2, 4.5, 4.8, 5.1, 5.4, 5.7]),
        'y': np.array([0.88434, 0.81406, 0.73332, 0.69874, 0.68276, 0.65688, 0.62317, 0.57424, 0.5643, 0.56202]),
    },
    {
        'name': 'Степенная',

        'x': np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]),
        'y': np.array([0.2, 0.6, 1.3, 2.4, 4.0, 6.0, 8.5, 11.5])
    },
    {
        'name': 'Линейная',
        'x': np.array([1, 1.1, 1.3, 1.4, 1.7, 1.8, 1.9, 2.1, 2.3, 2.5, 2.7]),
        'y': np.array([2, 2.1, 2.3, 2.4, 2.7, 2.8, 2.9, 3.1, 3.3, 3.5, 3.7]),
    },
    {
        'name': 'Гиперболическая',
        'x': np.array([-10, -5, -2, -1, 1, 2, 5, 8, 10]),
        'y': np.array([1 / -10, 1 / -5, 1 / -2, 1 / -1, 1 / 1, 1 / 2, 1 / 5, 1 / 8, 1 / 10]),
    },
    {
        'name': 'Квадратичная',
        'x': np.array([2.21, 2.24, 2.25, 2.3, 2.35, 2.37, 2.4, 2.41, 2.45, 2.47]),
        'y': np.array([5.7816, 6.15292, 6.23877, 6.81036, 6.99732, 7.3326, 7.7662, 7.4108, 8.2008, 8.0025]),
    }
]

def table_view(request):
    current_index = int(request.GET.get('index', 0))  # Получаем индекс таблицы из GET-запроса
    table = tables[current_index]
    x = table['x']
    y = table['y']
    name = table['name']

    # Вычисления для линейной функции
    try:
        coefficients_linear = np.polyfit(x, y, 1)
        k, b = coefficients_linear
        y_linear = k * x + b
        ss_linear = sum_of_squares(y, y_linear)
    except Exception as e:
        k, b = None, None
        y_linear = None
        ss_linear = None

    # Вычисления для квадратичной функции
    try:
        coefficients_quadratic = np.polyfit(x, y, 2)
        a, b, c = coefficients_quadratic
        y_quadratic = a * x**2 + b * x + c
        ss_quadratic = sum_of_squares(y, y_quadratic)
    except Exception as e:
        a, b, c = None, None, None
        y_quadratic = None
        ss_quadratic = None

    # Вычисления для гиперболической функции
    try:
        popt, _ = curve_fit(hyperbolic_model, x[x != 0], y[x != 0])  # Исключаем x = 0
        c = popt[0]
        y_hyperbolic = hyperbolic_model(x, c)
        ss_hyperbolic = sum_of_squares(y, y_hyperbolic)
    except Exception as e:
        c = None
        y_hyperbolic = None
        ss_hyperbolic = None

    # Вычисления для степенной функции
    try:
        popt, _ = curve_fit(power_model, x, y)
        a, b = popt
        y_power = power_model(x, a, b)
        ss_power = sum_of_squares(y, y_power)
    except Exception as e:
        a, b = None, None
        y_power = None
        ss_power = None

    # Построение графиков
    def plot_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    # График исходных данных
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.scatter(x, y, color='red', label='Исходные данные')
    ax1.set_title(f'Исходные данные ({name})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True)
    ax1.legend()
    plot1 = plot_to_base64(fig1)
    plt.close(fig1)  # Закрываем график, чтобы освободить память

    # График аппроксимации
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.scatter(x, y, color='red', label='Исходные данные')

    # Выбираем подходящую аппроксимацию
    if ss_hyperbolic is not None and ss_hyperbolic < ss_quadratic:
        # Разделяем гиперболу на две ветви
        x_positive = x[x > 0]
        x_negative = x[x < 0]
        y_positive = hyperbolic_model(x_positive, c)
        y_negative = hyperbolic_model(x_negative, c)
        ax2.plot(x_positive, y_positive, label=f'Гипербола (положительная ветвь)', color='blue')
        ax2.plot(x_negative, y_negative, label=f'Гипербола (отрицательная ветвь)', color='blue')
    elif ss_power is not None and ss_power < ss_quadratic:
        ax2.plot(x, y_power, label=f'Степенная: y = {a:.2f} * x^{b:.2f}', color='green')
    elif ss_linear is not None and ss_linear < ss_quadratic:
        ax2.plot(x, y_linear, label=f'Линейная: y = {k:.2f}x + {b:.2f}', color='orange')
    else:
        ax2.plot(x, y_quadratic, label=f'Квадратичная: y = {a:.2f}x^2 + {b:.2f}x + {c:.2f}', color='purple')

    ax2.set_title(f'Аппроксимация ({name})')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(True)
    ax2.legend()
    plot2 = plot_to_base64(fig2)
    plt.close(fig2)  # Закрываем график, чтобы освободить память

    # Возвращаем результаты в шаблон
    return render(request, 'myapp/table.html', {
        'name': name,
        'plot1': plot1,
        'plot2': plot2,
        'linear': {'k': k, 'b': b, 'ss': ss_linear},
        'quadratic': {'a': a, 'b': b, 'c': c, 'ss': ss_quadratic},
        'hyperbolic': {'c': c, 'ss': ss_hyperbolic},  # Всегда передаем гиперболическую
        'power': {'a': a, 'b': b, 'ss': ss_power},  # Всегда передаем степенную
        'next_index': (current_index + 1) % len(tables),  # Индекс следующей таблицы
        'prev_index': (current_index - 1) % len(tables),  # Индекс предыдущей таблицы
        'zip_data': zip(x, y),  # Передаем данные для таблицы
    })