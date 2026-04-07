#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include <algorithm>

/**
INTERFAZ DE TRANSFORMACIÓN
 * Usamos una clase abstracta para aplicar el polimorfismo.
 * Esto permite que el Tensor pueda ejecutar cualquier función (como ReLU o Sigmoid)
   sin conocer los detalles de su fórmula matemática.
 **/
class Tensor; // Declaración adelantada para que la interfaz reconozca el tipo Tensor.

class TensorTransform {
public:
    // Método virtual puro: define el "contrato" que ReLU y Sigmoid deben seguir.
    virtual Tensor apply(const Tensor& t) const = 0;

    // Destructor virtual: importante para que al borrar una transformación
    // no queden restos de memoria en las clases hijas.
    virtual ~TensorTransform() = default;
};

/**
 CLASE TENSOR
 * Se eligió un "Array dinámico contiguo" (double*) por ser
   el más eficiente en memoria y permitir un acceso rápido mediante índices.
**/
class Tensor {
private:
    std::vector<size_t> shape_; // Guarda las dimensiones (ej: {1000, 400})
    double* data_;              // Puntero a la memoria dinámica en el Heap
    size_t total_size_;         // Cantidad total de elementos (multiplicación de dimensiones)

public:
    // Constructor Principal
    // Recibe las dimensiones y los valores iniciales.
    Tensor(const std::vector<size_t>& shape, const std::vector<double>& values) {
        // Validación: máximo 3 dimensiones.
        if (shape.size() > 3) {
            throw std::invalid_argument("El tensor no puede tener mas de 3 dimensiones.");
        }

        shape_ = shape;
        total_size_ = 1;
        for (size_t s : shape_) {
            total_size_ *= s; // Calculamos el tamaño total multiplicando las dimensiones.
        }

        // Verificamos que nos pasen la cantidad correcta de datos.
        if (values.size() != total_size_) {
            throw std::runtime_error("La cantidad de valores no coincide con las dimensiones.");
        }

        //  Usamos 'new' para pedir espacio en el Heap.
        data_ = new double[total_size_];
        for (size_t i = 0; i < total_size_; ++i) {
            data_[i] = values[i]; // Copiamos los valores del vector al array dinámico.
        }
    }

    /**
    GESTIÓN DE MEMORIA Y CICLO DE VIDA
    **/
    // 1. Tensor ( const Tensor & other ) ; // Constructor de Copia
    // Sirve para crear un nuevo tensor copiando los datos de uno existente (Deep Copy).
    Tensor(const Tensor& other) {
        shape_ = other.shape_;
        total_size_ = other.total_size_;
        data_ = new double[total_size_]; // Pedimos nuestra propia memoria
        for (size_t i = 0; i < total_size_; ++i) {
            data_[i] = other.data_[i]; // Copiamos valor por valor
        }
    }

    // 2. Tensor ( Tensor && other ) noexcept ; // Constructor de Movimiento
    // Sirve para transferir la propiedad de la memoria de un objeto temporal a uno nuevo sin copiar.
    Tensor(Tensor&& other) noexcept {
        shape_ = std::move(other.shape_); // Movimiento del vector
        total_size_ = other.total_size_;
        data_ = other.data_;
        other.data_ = nullptr;
        other.total_size_ = 0;   // Dejamos al original vacío para que no borre nada
    }

    // 3. Tensor & operator =( const Tensor & other ) ; // Asignación por copia
    // Nos ayuda a copiar los datos de un tensor a otro que ya existe.
    Tensor& operator=(const Tensor& other) {
        if (this != &other) { // Verificamos que no sea el mismo objeto
            delete[] data_;   // Liberamos la memoria que teníamos antes
            shape_ = other.shape_;
            total_size_ = other.total_size_;
            data_ = new double[total_size_]; // Reservamos memoria nueva
            for (size_t i = 0; i < total_size_; ++i) {
                data_[i] = other.data_[i];
            }
        }
        return *this;
    }

    // 4. Tensor & operator =( Tensor && other ) noexcept ; // Asignación por movimiento
    // Su propósito es transferir la memoria de un objeto temporal a otro que ya existe.
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            shape_ = std::move(other.shape_); // Movimiento del vector
            total_size_ = other.total_size_;
            data_ = other.data_;
            other.data_ = nullptr;
            other.total_size_ = 0;
        }
        return *this;
    }

    // 5. ~ Tensor () ; // Destructor
    // Util para asegurar que la memoria dinámica se libere cuando el objeto deje de existir.
    ~Tensor() {
        if (data_ != nullptr) {
            delete[] data_;
        }
    }
    /**
    MÉTODOS ESTÁTICOS DE CREACIÓN
    **/
    static Tensor zeros(const std::vector<size_t>& shape) {
        size_t size = 1;
        for (size_t s : shape) size *= s;
        return Tensor(shape, std::vector<double>(size, 0.0));
    }

    static Tensor ones(const std::vector<size_t>& shape) {
        size_t size = 1;
        for (size_t s : shape) size *= s;
        return Tensor(shape, std::vector<double>(size, 1.0));
    }

    static Tensor random(const std::vector<size_t>& shape, double min, double max) {
        size_t size = 1;
        for (size_t s : shape) size *= s;
        std::vector<double> v(size);
        std::default_random_engine engine(42); // Semilla para resultados consistentes.
        std::uniform_real_distribution<double> dist(min, max);
        for (size_t i = 0; i < size; ++i) v[i] = dist(engine);
        return Tensor(shape, v);
    }

    static Tensor arange(double start, double end) {
        std::vector<double> v;
        for (double i = start; i < end; ++i) v.push_back(i);
        return Tensor({v.size()}, v);
    }
    /**
    SOBRECARGA DE OPERADORES
    **/
    // Suma: Soporta sumar un bias (vector pequeño) a una matriz grande usando el operador módulo.
    Tensor operator+(const Tensor& other) const {
        std::vector<double> res_v(total_size_);
        for (size_t i = 0; i < total_size_; ++i) {
            // El módulo (%) permite que si 'other' es más corto, se repita (Broadcasting).
            res_v[i] = data_[i] + other.data_[i % other.total_size_];
        }
        return Tensor(shape_, res_v);
    }

    // Multiplicación por escalar: Multiplica cada elemento por un número.
    Tensor operator*(double scalar) const {
        std::vector<double> res_v(total_size_);
        for (size_t i = 0; i < total_size_; ++i) {
            res_v[i] = data_[i] * scalar;
        }
        return Tensor(shape_, res_v);
    }

    /**
     MÉTODO VIEW
     * Permite cambiar la forma (ej: de 3D a 2D) sin copiar datos.
     * Simplemente creamos un nuevo objeto que reinterpreta el 'shape_'.
     **/
    Tensor view(const std::vector<size_t>& new_shape) const {
        size_t new_size = 1;
        for (size_t s : new_shape) new_size *= s;

        if (new_size != total_size_) {
            throw std::runtime_error("View: El nuevo shape no coincide con el tamano original.");
        }

        Tensor result = *this; // Copia los datos (gracias al constructor de copia).
        result.shape_ = new_shape;
        return result;
    }

    /**
    *CONCATENACIÓN
    **/
    static Tensor concat(const std::vector<Tensor>& tensors, size_t axis) {
        if (axis != 0) throw std::invalid_argument("Solo eje 0 soportado.");

        size_t new_dim0 = 0;
        for (const auto& t : tensors) new_dim0 += t.shape_[0];

        std::vector<size_t> new_shape = tensors[0].shape_;
        new_shape[0] = new_dim0;

        std::vector<double> vals;
        for (const auto& t : tensors) {
            for (size_t i = 0; i < t.total_size_; ++i) {
                vals.push_back(t.data_[i]);
            }
        }
        return Tensor(new_shape, vals);
    }
    /**
     FUNCIONES AMIGAS (MATMUL)
     * La multiplicación matricial necesita acceder a los datos
       privados de dos tensores distintos (A y B) para calcular Filas x Columnas.
    **/
    friend Tensor matmul(const Tensor& a, const Tensor& b) {
        size_t rows_a = a.shape_[0];
        size_t cols_a = a.shape_[1];
        size_t rows_b = b.shape_[0];
        size_t cols_b = b.shape_[1];

        if (cols_a != rows_b) throw std::runtime_error("Matmul: Columnas de A != Filas de B.");

        std::vector<double> res(rows_a * cols_b, 0.0);
        // Algoritmo triple for para multiplicación de matrices.
        for (size_t i = 0; i < rows_a; ++i) {
            for (size_t j = 0; j < cols_b; ++j) {
                for (size_t k = 0; k < cols_a; ++k) {
                    res[i * cols_b + j] += a.data_[i * cols_a + k] * b.data_[k * cols_b + j];
                }
            }
        }
        return Tensor({rows_a, cols_b}, res);
    }

    // Método apply
    Tensor apply(const TensorTransform& transform) const {
        return transform.apply(*this);
    }

    // Getters necesarios para ReLU y Sigmoid.
    size_t total_size() const { return total_size_; }
    double* data() const { return data_; }
    std::vector<size_t> shape() const { return shape_; }
};

/**
 IMPLEMENTACIÓN DE TRANSFORMACIONES
 * Heredan de la interfaz y aplican la fórmula a cada elemento.
**/
class ReLU : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override {
        std::vector<double> v(t.total_size());
        for (size_t i = 0; i < t.total_size(); ++i) {
            // ReLU: Si es negativo se vuelve 0, si es positivo se queda igual.
            v[i] = (t.data()[i] > 0) ? t.data()[i] : 0.0;
        }
        return Tensor(t.shape(), v);
    }
};

class Sigmoid : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override {
        std::vector<double> v(t.total_size());
        for (size_t i = 0; i < t.total_size(); ++i) {
            // Sigmoid: Aplasta los valores entre 0 y 1.
            v[i] = 1.0 / (1.0 + std::exp(-t.data()[i]));
        }
        return Tensor(t.shape(), v);
    }
};

/**
 RED NEURONAL
 */
int main() {

    try {
        // Paso 1: Crear tensor 1000x20x20
        Tensor paso1 = Tensor::random({1000, 20, 20}, 0.0, 1.0);

        // Paso 2: Cambiar forma a 1000x400 (view)
        Tensor paso2 = paso1.view({1000, 400});

        // Paso 3: Multiplicación por matriz 400x100
        Tensor W1 = Tensor::random({400, 100}, -0.5, 0.5);
        Tensor paso3 = matmul(paso2, W1);

        // Paso 4: Sumar bias 1x100
        Tensor b1 = Tensor::ones({1, 100}) * 0.1;
        Tensor paso4 = paso3 + b1;

        // Paso 5: Función ReLU
        ReLU relu;
        Tensor paso5 = paso4.apply(relu);

        // Paso 6: Multiplicar por 100x10
        Tensor W2 = Tensor::random({100, 10}, -0.5, 0.5);
        Tensor paso6 = matmul(paso5, W2);

        // Paso 7: Sumar bias 1x10
        Tensor b2 = Tensor::ones({1, 10}) * 0.1;
        Tensor paso7 = paso6 + b2;

        // Paso 8: Función Sigmoid
        Sigmoid sigmoid;
        Tensor paso8 = paso7.apply(sigmoid);

        std::cout << "Red neuronal ejecutada exitosamente." << std::endl;
        std::cout << "Dimension final del resultado: " << paso8.shape()[0] << "x" << paso8.shape()[1] << std::endl;

    } catch (const std::exception& e) {
        // Captura de cualquier error de dimensiones o memoria.
        std::cerr << "Error detectado: " << e.what() << std::endl;
    }

    return 0;
}