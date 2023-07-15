Instrucciones

Para compilar vectorización y OpenMP (no CUDA) `.cpp`:

    g++ main.cpp -march=native -fopenmp

Para compilar CUDA `.cu`:

    nvcc main.cu

Para ejecutar:

    ./programa <nombre_archivo>

o

    ./programa <nombre_archivo> -p

La opción `-p` indica que se desea imprimir las matrices.

Ejemplo:

    ./a.out 4.mtx

Resultados: Una vez que el programa haya finalizado la ejecución, 
mostrará los tiempos de ejecución de las diferentes versiones del algoritmo implementadas. 
Si se agrega el argumento `-p` al ejecutar el programa, se imprimirá las 2 matrices: inicial y resuelta. 
Luego pide de entrada 2 vértices para mostrar el camino mas corto y su costo.
