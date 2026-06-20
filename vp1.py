import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

m = 0.5      # масса маятника, кг
L = 1.0      # длина подвеса, м
k = 2.0      # жесткость пружины, Н/м
L1 = 0.3     # расстояние от точки подвеса до места крепления пружины, м
beta = 0.1   # коэффициент затухания (сила сопротивления = 2*m*beta * скорость), с^-1
g = 9.8      # ускорение свободного падения, м/с^2

# Начальные условия
theta1_0 = 0.1    # рад
theta2_0 = 0.0    # рад
omega1_0 = 0.0    # рад/с
omega2_0 = 0.0    # рад/с

# Временной интервал
t_start = 0.0
t_end = 20.0
t_eval = np.linspace(t_start, t_end, 2000)

# Система ОДУ
def coupled_pendulums(t, y):
    th1, om1, th2, om2 = y
    om1_dot = - (g/L) * th1 - (k * L1**2 / (m * L**2)) * (th1 - th2) - 2*beta * om1
    om2_dot = - (g/L) * th2 - (k * L1**2 / (m * L**2)) * (th2 - th1) - 2*beta * om2
    return [om1, om1_dot, om2, om2_dot]

# Численное решение
sol = solve_ivp(coupled_pendulums, (t_start, t_end), [theta1_0, omega1_0, theta2_0, omega2_0],
                t_eval=t_eval, method='RK45')

t = sol.t
theta1 = sol.y[0]
theta2 = sol.y[2]
omega1 = sol.y[1]
omega2 = sol.y[3]

# Без затухания (beta=0) нормальные частоты:
omega_sym = np.sqrt(g / L)
omega_asym = np.sqrt(g / L + 2 * k * L1**2 / (m * L**2))
print(f"Нормальная частота синфазной моды: {omega_sym:.3f} рад/с")
print(f"Нормальная частота противофазной моды: {omega_asym:.3f} рад/с")
print(f"Соответствующие периоды: {2*np.pi/omega_sym:.3f} с и {2*np.pi/omega_asym:.3f} с")

# Графики
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(t, theta1, label=r'$\theta_1$', color='b')
plt.plot(t, theta2, label=r'$\theta_2$', color='r')
plt.xlabel('Время, с')
plt.ylabel('Угол, рад')
plt.title('Углы отклонения')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t, omega1, label=r'$\dot\theta_1$', color='b')
plt.plot(t, omega2, label=r'$\dot\theta_2$', color='r')
plt.xlabel('Время, с')
plt.ylabel('Угловая скорость, рад/с')
plt.title('Угловые скорости')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()