# Tensor++: Librería de Tensores en C++

## 1. Descripción general
En esta tarea se deberá diseñar e implementar una librería en **C++** llamada **Tensor++**, inspirada en bibliotecas científicas como *NumPy* y *PyTorch*. El objetivo es crear una clase **Tensor** capaz de manejar **tensores de hasta 3 dimensiones**, realizar operaciones matemáticas avanzadas y ser utilizada en una aplicación realista de procesamiento de datos.

---

## Características Técnicas

* **Gestión de Memoria Dinámica**: Implementación de un "Array dinámico contiguo" (`double*`) para maximizar la eficiencia y el aprovechamiento de la memoria caché.
*  **Garantía de estabilidad** mediante la implementación de:
    * Constructor de copia (Deep Copy).
    * Constructor de movimiento (Move Semantics).
    * Operador de asignación por copia.
    * Operador de asignación por movimiento.
    * Destructor para liberación de recursos.
* **Polimorfismo**: Uso de una clase abstracta `TensorTransform` para desacoplar la lógica del Tensor de las funciones de activación (`ReLU`, `Sigmoid`).
* **Broadcasting**: Soporte para operaciones entre tensores de diferentes tamaños (útil para sumar vectores de sesgo o *bias*).

---

## Estructura del Código

1.  **Interfaz `TensorTransform`**: Define el contrato para cualquier operación elemento a elemento.
2.  **Clase `Tensor`**: Núcleo de la librería que gestiona dimensiones (`shape`), datos y operaciones matemáticas (`matmul`, suma, multiplicación escalar).
3.  **Activaciones**: Implementaciones concretas de `ReLU` y `Sigmoid`.
4.  **Aplicación Real**: Un flujo de red neuronal que procesa un tensor de entrada de $1000 \times 20 \times 20$ hasta obtener una salida clasificada de $1000 \times 10$.

---

## Instrucciones de Compilación y Ejecución

### Requisitos
* Compilador con soporte para **C++11** o superior (GCC, Clang o MSVC).

### Compilación por Consola (Terminal)
Para compilar el proyecto manualmente, ejecute el siguiente comando:
```bash
g++ -std=c++11 main.cpp -o tensor_plus_plus
