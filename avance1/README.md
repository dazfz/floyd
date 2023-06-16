# README

## Descripción

Programa que implementa el algoritmo de Floyd-Warshall para encontrar los caminos más cortos en un grafo con pesos. El programa está escrito en C++ y utiliza distintas técnicas de paralelización: vectorización, OpenMP y (proximamente) CUDA.

### Implementación

El programa tiene 3 archivos:

- `main.cpp`: Lectura del archivo de entrada, inicialización del grafo y ejecuta las funciones, tomando el tiempo de ejecución.
- `floyd.cpp`: Funciones que implementan el algoritmo de Floyd-Warshall con distintas paralelizaciones.
- `print.cpp`: Imprimir la matriz original y la matriz `dist`. Además puede imprimir el camino más corto entre 2 vértices y su costo.

## Compilación

Para compilar:

```
g++ main.cpp -march=native -fopenmp
```

## Ejecución

Para ejecutar:

```
./programa <nombre_archivo>
```

o

```
./programa <nombre_archivo> -p
```

La opción `-p` indica que se desea imprimir las matrices.

Ejemplo:

```
./a.out 4.mtx
```

El archivo de entrada debe tener un formato específico. El primer número en el archivo debe ser el número de vértices del grafo (V), seguido del número de aristas (E). A continuación, cada línea debe contener tres números separados por espacios: el vértice de origen de la arista, el vértice de destino y el peso de la arista. Los vértices deben ser mayores a 0.

Por ejemplo:

```
4 5

1 3 -2
2 1 4
4 2 -1
2 3 3
3 4 2
```

## Resultados

Una vez que el programa haya finalizado la ejecución, mostrará los tiempos de ejecución de las diferentes versiones del algoritmo implementadas. Si se agrega el argumento `-p` al ejecutar el programa, se imprimirá las 2 matrices: inicial y resuelta. Luego pide de entrada 2 vértices para mostrar el camino mas corto y su costo.
