# Versión mínima de CMake
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# Nombre del proyecto (ahora solo CXX, no CUDA)
project(MPolylla_OMP LANGUAGES CXX)

# Establecer el estándar de C++ (C++11 es suficiente)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Buscar el paquete OpenMP
find_package(OpenMP REQUIRED)

# Incluir el directorio 'src' para los headers
include_directories(src)

# Crear el ejecutable
add_executable(
    MPolylla_OMP
    main.cpp
)

# Enlazar las bibliotecas necesarias
target_link_libraries(MPolylla_OMP
    PRIVATE
    OpenMP::OpenMP_CXX # La biblioteca OpenMP
)

# Mensaje de estado
if(OpenMP_FOUND)
    message(STATUS "OpenMP encontrado. Compilando con soporte para OpenMP.")
else()
    message(WARNING "OpenMP no se encontró. La compilación será monohilo.")
endif()