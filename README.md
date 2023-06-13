# README

## Descripción

Este es un programa que implementa el algoritmo de Floyd-Warshall para encontrar los caminos más cortos en un grafo con pesos. El programa está escrito en C++ y utiliza distintas técnicas de paralelización: vectorización, OpenMP y (proximamente) CUDA.

## Compilación

Para compilar:

```
g++ floyd.cpp -march=native -fopenmp 
```

## Ejecución

Para ejecutar:

```
./programa <nombre_archivo>
```
Ejemplo:

```
./a.out 4.mtx
```
El archivo de entrada debe tener un formato específico. El primer número en el archivo debe ser el número de vértices del grafo (V), seguido del número de aristas (E). A continuación, cada línea debe contener tres números separados por espacios: el vértice de origen de la arista, el vértice de destino y el peso de la arista. Los vértices deben ser mayor a 0.  Por ejemplo:

```
4 5

1 3 -2
2 1 4
4 2 -1
2 3 3
3 4 2
```

## Resultados

Una vez que el programa haya finalizado la ejecución, mostrará los tiempos de ejecución de las diferentes versiones del algoritmo implementadas. Si se agrega el argumento ``` -p``` al ejecutar el programa, se imprimirá las 2 matrices, inicial y resuelta.