#include <stdio.h>
#include <string>
#include <iostream> // <--- Añadido para std::cout
#include "src/triangulation.h"  // Tu nueva clase Triangulation
#include "src/mpolylla_omp.h"   // La clase Polylla convertida (nombre de archivo original)

int main(int argc, char* argv[])
{
    std::cout << "[DEBUG] main: Inicio de main." << std::endl;

    if (argc < 2){
        printf("Error: Archivo de entrada no especificado.\n");
        printf("Uso: ./MPolylla_OMP <archivo_entrada.off> [archivo_salida_base]\n");
        printf("Ej: ./MPolylla_OMP data/mesh.off resultados/mesh_out\n");
        printf("     Generará: resultados/mesh_out.off y resultados/mesh_out.json\n");
        return 0;
    }

    std::string input_file_name(argv[1]);
    std::string output_file_name("output"); // Nombre base, sin extensión
    
    if (argc > 2)
        output_file_name = std::string(argv[2]);

    std::cout << "[DEBUG] main: Archivo de entrada: " << input_file_name << std::endl;
    std::cout << "[DEBUG] main: Nombre base de salida: " << output_file_name << std::endl;

    
    // ---------------------------------------------------
    // INICIO DE SECCIÓN MODIFICADA
    // ---------------------------------------------------

    std::string off_file = input_file_name;
    std::string output = output_file_name;
    std::string output_off_file = output + ".off";
    std::string output_json_file = output + ".json";

    printf("Ejecutando MPolylla (OpenMP)...\n");
    std::cout << "[DEBUG] main: Llamando constructor Polylla(" << off_file << ")..." << std::endl;

    // 1. Crear y ejecutar Polylla (el constructor hace todo el trabajo)
    Polylla mesh(off_file);
    
    std::cout << "[DEBUG] main: Constructor Polylla finalizado." << std::endl;
    printf("Ejecución de MPolylla finalizada.\n");

    // 2. Escribir el resultado a un nuevo archivo .off
    printf("Escribiendo archivo de salida .off...\n");
    std::cout << "[DEBUG] main: Llamando print_OFF(" << output_off_file << ")..." << std::endl;
    mesh.print_OFF(output_off_file);
    std::cout << "[DEBUG] main: print_OFF finalizado." << std::endl;
    printf("Archivo de salida escrito en: %s\n", output_off_file.c_str());

    // 3. Escribir las estadísticas a un archivo .json
    printf("Escribiendo archivo de stats .json...\n");
    std::cout << "[DEBUG] main: Llamando print_stats(" << output_json_file << ")..." << std::endl;
    mesh.print_stats(output_json_file);
    std::cout << "[DEBUG] main: print_stats finalizado." << std::endl;
    printf("Archivo de stats escrito en: %s\n", output_json_file.c_str());

    // 4. Liberar memoria (El destructor de 'mesh' se llama automáticamente al salir de main)
    std::cout << "[DEBUG] main: Fin de main. El destructor de Polylla se llamará ahora." << std::endl;
    
    // ---------------------------------------------------
    // FIN DE SECCIÓN MODIFICADA
    // ---------------------------------------------------

    return 0;
}