import matplotlib.pyplot as plt

# Datos
n = [64, 128, 214, 532, 1080, 2000, 5000, 8362]
normal = [4, 22.4, 111.7, 1577.3, 13463.1, 83954.4, 1341106.667, 6085037.5]
vectorizado8 = [0, 2, 12.3, 182.7, 1482.4, 10135.4, 154406.3333, 689809]
vectorizado16 = [0, 1, 8.1, 111.3, 920, 6239.4, 100318, 452683]
openmp = [3, 17.9, 69.8, 859.8, 7179, 44832.8, 667092, 3096147]
vec_openmp = [0.2, 1, 6.9, 107.9, 765.6, 5210.3, 81280.66667, 376512]
cuda = [8, 13, 15, 36.1, 154, 862, 15459, 77725.83]
block_openmp = [5, 17, 55, 486, 3292, 22001, 346436, 2097657]
# dev estandar cuda: 20.3372, 27.6803, 24.6698, 22.9976, 23.9833, 113.986, 754.679, 208.40
# 0
# Convertir a segundos
normal = [t / 1000 for t in normal]
vectorizado8 = [t / 1000 for t in vectorizado8]
vectorizado16 = [t / 1000 for t in vectorizado16]
openmp = [t / 1000 for t in openmp]
vec_openmp = [t / 1000 for t in vec_openmp]
cuda = [t / 1000 for t in cuda]
block_openmp = [t / 1000 for t in block_openmp]
# Crear grafico
plt.plot(n, normal, marker="o", label="Normal")
plt.plot(n, vectorizado8, marker="o", label="Vectorizado8")
plt.plot(n, vectorizado16, marker="o", label="Vectorizado16")
plt.plot(n, openmp, marker="o", label="OpenMP")
plt.plot(n, vec_openmp, marker="o", label="Vec+OpenMP")
plt.plot(n, cuda, marker="o", label="CUDA")
plt.plot(n, block_openmp, marker="o", label="BlockOpenMP")

# Configuraciones del grafico
plt.xlabel("n")
plt.ylabel("Tiempo (segundos)")
plt.title("Tiempo de ejecución")
plt.legend()
plt.grid(True, which="major", axis="y")

plt.savefig("./assets/grafico.png")
plt.show()
