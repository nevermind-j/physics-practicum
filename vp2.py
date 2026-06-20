import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 1. ПАРАМЕТРЫ
dt = 1e-4                  # 0.1 мс
fs = 1 / dt                # частота дискретизации
print("Частота дискретизации fs =", fs, "Гц")

# 2. ЗАГРУЗКА ДАННЫХ
data = np.load("signal_1.npy")
N = len(data)

print("Число точек:", N)

t = np.arange(N) * dt

# 3. FFT
fft_vals = np.fft.fft(data)
freqs = np.fft.fftfreq(N, dt)

# берем только положительные частоты
positive = freqs > 0
freqs_pos = freqs[positive]
fft_pos = np.abs(fft_vals[positive])

# 4. ПОИСК ПИКОВ (частоты сигнала)
# порог — 30% от максимума спектра
threshold = 0.3 * np.max(fft_pos)

peaks, properties = find_peaks(fft_pos, height=threshold)
detected_freqs = freqs_pos[peaks]

print("\nОбнаруженные частоты (Гц):")
for f in detected_freqs:
    print(f"{f:.2f}")

# 5. ФИЛЬТРАЦИЯ
filtered_fft = fft_vals.copy()

# зануляем частоты ниже порога
mask = np.abs(fft_vals) < threshold
filtered_fft[mask] = 0

filtered_signal = np.fft.ifft(filtered_fft).real

# 6. ГРАФИКИ
plt.figure(figsize=(12, 10))

# исходный сигнал
plt.subplot(3, 1, 1)
plt.plot(t[:3000], data[:3000])
plt.title("Зашумленный сигнал")
plt.xlabel("t (с)")
plt.ylabel("Амплитуда")

# спектр
plt.subplot(3, 1, 2)
plt.plot(freqs_pos, fft_pos)
plt.plot(detected_freqs, fft_pos[peaks], "ro")
plt.title("Амплитудный спектр")
plt.xlabel("Частота (Гц)")
plt.ylabel("|X(f)|")
plt.xlim(0, 3000)

# отфильтрованный сигнал
plt.subplot(3, 1, 3)
plt.plot(t[:3000], filtered_signal[:3000])
plt.title("Отфильтрованный сигнал")
plt.xlabel("t (с)")
plt.ylabel("Амплитуда")

plt.tight_layout()
plt.show()