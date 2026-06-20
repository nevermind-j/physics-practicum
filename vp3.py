import numpy as np
import matplotlib.pyplot as plt

# Параметры
N = 5                 # число щелей (1–10)
a = 20e-6             # ширина щели (м)
d = 100e-6            # период (м)
L = 1.0               # расстояние до экрана (м)

lambda0 = 550e-9      # центральная длина волны (м)
delta_lambda = 20e-9  # ширина спектра (м) (0 = монохроматический)

screen_size = 0.02    # размер экрана (м)
points = 4000         # число точек

# Сетка
x = np.linspace(-screen_size/2, screen_size/2, points)
theta = x / L

# Функция интенсивности
def intensity(theta, wavelength):
    beta = np.pi * a * np.sin(theta) / wavelength
    delta = 2*np.pi * d * np.sin(theta) / wavelength

    beta = np.where(beta == 0, 1e-10, beta)
    delta = np.where(delta == 0, 1e-10, delta)

    envelope = (np.sin(beta)/beta)**2
    interference = (np.sin(N*delta/2)/np.sin(delta/2))**2

    return envelope * interference

# Моно/Квазимонохроматический
if delta_lambda == 0:
    I = intensity(theta, lambda0)
else:
    wavelengths = np.linspace(lambda0 - 3*delta_lambda,
                              lambda0 + 3*delta_lambda, 30)
    spectrum = np.exp(-(wavelengths-lambda0)**2 /
                      (2*delta_lambda**2))

    I = np.zeros_like(theta)

    for wl, w in zip(wavelengths, spectrum):
        I += w * intensity(theta, wl)

# Нормировка
I /= np.max(I)

# График интенсивности
plt.figure(figsize=(10,5))
plt.plot(x*1000, I)
plt.xlabel("Координата x (мм)")
plt.ylabel("Нормированная интенсивность")
plt.title(f"Интерференция от {N} щелей")
plt.grid()
plt.show()

# Цветное распределение
image = np.tile(I, (200,1))
plt.figure(figsize=(10,3))
plt.imshow(image, extent=[x[0]*1000, x[-1]*1000, 0, 1],
           cmap='inferno', aspect='auto')
plt.xlabel("x (мм)")
plt.yticks([])
plt.title("Цветное распределение интенсивности")
plt.colorbar(label="Интенсивность")
plt.show()