import numpy as np
import matplotlib.pyplot as plt

n = [64, 128, 214, 532, 1080, 2000, 5000, 8362]
block_sizes = [2, 4, 8, 16, 32]
tiempos = [
    [57, 57, 57, 58, 57],
    [61, 63, 59, 60, 59],
    [70, 63, 63, 62, 62],
    [210, 105, 86, 83.1, 83],
    [1146, 432, 254.7, 242.7, 229],
    [6107, 2004.9, 916.5, 877.2, 782.3],
    [88851, 31319, 12976, 11238, 9677.3],
    [801973, 223272, 81408, 52453, 57563],
]

bar_width = 0.15

# Calcular la posición de las barras en el eje x
x_pos = np.arange(len(n))

# Crear el gráfico de barras
fig, ax = plt.subplots()
for i in range(len(block_sizes)):
    ax.bar(
        x_pos + (i * bar_width),
        [tiempos[j][i] for j in range(len(n))],
        bar_width,
        label=f"{block_sizes[i]}",
    )

# Configurar etiquetas y título del gráfico
ax.set_xlabel("n")
ax.set_ylabel("Tiempos [ms]")
ax.set_title("Tiempo de CUDA con distintos tamaños de bloque (milisegundos)")

ax.set_xticks(x_pos)
ax.set_xticklabels(n)

# Agregar leyenda
ax.legend()
plt.grid(True, which="major", axis="y")
# Mostrar el gráfico
plt.show()
