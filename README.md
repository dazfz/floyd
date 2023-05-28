# README

### Descripción

Este es un programa que implementa el algoritmo de Floyd-Warshall para encontrar los caminos más cortos en un grafo con pesos. El programa está escrito en C++ y utiliza distintas técnicas de paralelización: vectorización, OpenMP y (proximamente) CUDA , para mejorar el rendimiento del algoritmo.

### Compilación

Para compilar:

```
g++ floyd.cpp -O3 -march=native -fopenmp 
```

### Ejecución

Para ejecutar:

```
./programa <nombre_archivo>
```
Ejemplo:

```
./a.exe 64.txt
```
El archivo de entrada debe tener un formato específico. El primer número en el archivo debe ser el número de vértices del grafo (V), seguido del número de aristas (E). A continuación, cada línea debe contener tres números separados por espacios: el vértice de origen de la arista, el vértice de destino y el peso de la arista. Los vértices deben ser mayor a 0.  Por ejemplo:

```
8 11
1 2 6
3 2 6
3 5 1
3 6 8
4 2 3
4 8 8
5 3 5
5 8 4
6 7 9
7 5 5
7 8 9
```

### Resultados

Una vez que el programa haya finalizado la ejecución, mostrará los tiempos de ejecución de las diferentes versiones del algoritmo implementadas. 