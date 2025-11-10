//#pragma once

#include "kernel_omp.h" 
#include <omp.h>
#include <chrono>
#include <iostream>
#include <cstring>      // Para memcpy y memset
#include <vector>       // Para std::vector
#include <fstream>      // Para std::ofstream
#include <iomanip>      // Para std::setprecision
#include <string>       // Para std::string
#include <atomic>       // Para std::atomic

// NOTA: Las funciones de scan/compaction ahora están en kernel_omp.h

// Temporizador de CPU usando std::chrono
struct CPUTimer {
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point stop;

    CPUTimer() {}
    ~CPUTimer() {}

    void Start() {
        start = std::chrono::high_resolution_clock::now();
    }

    void Stop() {
        stop = std::chrono::high_resolution_clock::now();
    }

    float Elapsed() {
        return std::chrono::duration<float, std::milli>(stop - start).count();
    }
};


/*
Esta clase contiene la estructura principal del algoritmo MPolylla
(Nombres de variables alineados con GPolylla original)
*/
class Polylla{ // <--- Renombrada de MPolylla a Polylla
    private:
        // Punteros a los datos de la triangulación (como en GPolylla)
        Triangulation *mesh_input = nullptr;
        Triangulation *mesh_output = nullptr;

        // Búferes de trabajo internos de MPolylla
        // (Equivalentes a max_edges_d, frontier_edges_d, seed_edges_ad, seed_edges_bd en GPolylla)
        bit_vector_d *h_max_edges;
        bit_vector_d *h_frontier_edges;
        int *h_seed_edges;      // bit-vector (1 / 0) -> GPolylla::seed_edges_ad (tipo cambiado a int)
        int *h_scan;            // scan result -> GPolylla::seed_edges_bd
        int *h_seed_edges_comp; // compacted result -> GPolylla::seed_edges_d (host copy)

        // Miembros de GPolylla necesarios para print_OFF
        int m_polygons = 0; // GPolylla::seed_len
        std::vector<int> output_seeds; // GPolylla::output_seeds

        // número de items
        int n_vertices;
        int n_halfedges;
        int n_faces;
        
        // temporizador
        CPUTimer timer;

        // Tiempos (para print_stats, igual que en GPolylla)
        double t_label_max_edges_d = 0;
        double t_label_frontier_edges_d = 0;
        double t_label_seed_edges_d = 0;
        double t_label_extra_frontier_edge_d = 0;
        double t_label_seed_scan_d = 0;
        double t_label_seed_compaction_d = 0;
        double t_traversal_1_d = 0;
        double t_traversal_2_d = 0;
        double t_overwrite_seed_d = 0;

        // Función principal de ejecución (privada, llamada por el constructor)
        void construct_Polylla(){
            std::cout << "[DEBUG] mpolylla_omp.h: run() [construct_Polylla] llamado." << std::endl;
            
            // --- Fases 1, 2, 3 (Label, Frontier, Seed) ---
            std::cout << "[DEBUG] mpolylla_omp.h: run: Iniciando Fases 1-3: Label, Frontier, Seed..." << std::endl;
            timer.Start();
            label_edges_max_d(*mesh_input, h_max_edges, n_faces);
            timer.Stop();
            t_label_max_edges_d = timer.Elapsed();
            printf ("[TIME] Label max edges phase: %f\n", (float) t_label_max_edges_d);

            timer.Start();
            label_phase(*mesh_input, h_max_edges, h_frontier_edges, n_halfedges);
            timer.Stop();
            t_label_frontier_edges_d = timer.Elapsed();
            printf ("[TIME] Label frontier edges phase: %f\n", (float) t_label_frontier_edges_d);

            timer.Start();
            seed_phase_d(*mesh_input, h_max_edges, h_seed_edges, n_halfedges);
            timer.Stop();
            t_label_seed_edges_d = timer.Elapsed();
            printf ("[TIME] Label seed edges phase: %f\n", (float) t_label_seed_edges_d);

            // --- Fase 4 (Repair) ---
            std::cout << "[DEBUG] mpolylla_omp.h: run: Iniciando Fase 4: Repair (label_extra_frontier_edge_d)..." << std::endl;
            timer.Start();
            label_extra_frontier_edge_d(*mesh_input, h_frontier_edges, h_seed_edges, n_vertices);
            timer.Stop();
            t_label_extra_frontier_edge_d = timer.Elapsed();
            printf ("[TIME] Repair phase (GPolylla 4): %f\n", (float) t_label_extra_frontier_edge_d);

            // --- Fase 5 (Travel) ---
            std::cout << "[DEBUG] mpolylla_omp.h: run: Iniciando Fase 5: Travel..." << std::endl;
            timer.Start();
            travel_phase_d(*mesh_input, *mesh_output, h_max_edges, h_frontier_edges, n_halfedges);
            timer.Stop();
            t_traversal_1_d = timer.Elapsed();
            printf ("[TIME] Travel phase (GPolylla 5): %f\n", (float) t_traversal_1_d);

            // --- Fase 6 (Post-Travel Repair) ---
            std::cout << "[DEBUG] mpolylla_omp.h: run: Iniciando Fase 6: Post-Travel Repair..." << std::endl;
            timer.Start();
            search_frontier_edge_d(*mesh_input, h_frontier_edges, h_seed_edges, n_halfedges);
            timer.Stop();
            t_traversal_2_d = timer.Elapsed();
            printf ("[TIME] Repair search frontier edge (GPolylla 6): %f\n", (float) t_traversal_2_d);

            timer.Start();
            overwrite_seed_d(*mesh_output, h_seed_edges, n_halfedges);
            timer.Stop();
            t_overwrite_seed_d = timer.Elapsed();
            printf ("[TIME] Repair overwrite seed (GPolylla 7): %f\n", (float) t_overwrite_seed_d);


            // --- Fase 7 (Scan & Compaction) ---
            std::cout << "[DEBUG] mpolylla_omp.h: run: Iniciando Fase 7: Scan y Compaction (Serial)..." << std::endl;
            
            // 7.2. COMPACTION (GPolylla: Label seed edges 3)
            timer.Start();
            compaction_serial_omp(h_seed_edges_comp, &m_polygons, h_seed_edges, n_halfedges);
            timer.Stop();
            t_label_seed_compaction_d = timer.Elapsed();
            std::cout << "[DEBUG] mpolylla_omp.h: run: 7.2. " << m_polygons << " polígonos (seeds) contados." << std::endl;
            printf ("[TIME] Compaction phase (GPolylla 9): %f\n", (float) t_label_seed_compaction_d);
            
            // print frotier_edges for debug
            std::cout << "[DEBUG] h_seed_edges_comp after compaction:" << std::endl;
            for(int i = 0; i < m_polygons; i++){
                std::cout << h_seed_edges_comp[i] << " ";
            }
            std::cout << std::endl;
            


            // --- Fase 8: Copia Final ---
            
            // Copiar vértices (no cambiaron)
            mesh_output->Vertices = mesh_input->Vertices;

            // Copia final a output_seeds (GPolylla: Loop final)
            std::cout << "[DEBUG] mpolylla_omp.h: run: Copiando resultado final a output_seeds..." << std::endl;
            output_seeds.clear();
            output_seeds.resize(m_polygons);
            memcpy(output_seeds.data(), h_seed_edges_comp, sizeof(int) * m_polygons);

            std::cout << "[DEBUG] mpolylla_omp.h: run: Fin." << std::endl;
        }

    public:
        // --- CONSTRUCTORES (NUEVOS) ---

        // Constructor por defecto
        Polylla() : mesh_input(nullptr), mesh_output(nullptr) {
            std::cout << "[DEBUG] Polylla::Constructor() (default) llamado." << std::endl;
        }

        // Constructor desde OFF file (el que pide main.cpp)
        Polylla(std::string off_file) {
            std::cout << "[DEBUG] Polylla::Constructor(OFF_file=" << off_file << ") llamado." << std::endl;
            
            // 1. Crear mesh_input
            this->mesh_input = new Triangulation(off_file);
            std::cout << "[DEBUG] Polylla::Constructor: mesh_input creado." << std::endl;
            
            // 2. Crear mesh_output usando constructor de copia
            this->mesh_output = new Triangulation(*mesh_input);
            std::cout << "[DEBUG] Polylla::Constructor: mesh_output (copia) creado." << std::endl;
            
            // Inicializar variables de GPolylla
            n_vertices = mesh_input->n_vertices;
            n_halfedges = mesh_input->n_halfedges;
            n_faces = mesh_input->n_faces;
            m_polygons = 0;

            std::cout << "[DEBUG] Polylla::Constructor: n_vertices=" << n_vertices << ", n_halfedges=" << n_halfedges << ", n_faces=" << n_faces << std::endl;

            // 3. Alocar memoria de trabajo
            std::cout << "[DEBUG] Polylla::Constructor: Alocando buffers de trabajo..." << std::endl;
            h_max_edges = new bit_vector_d[n_halfedges];
            h_frontier_edges = new bit_vector_d[n_halfedges];
            h_seed_edges = new int[n_halfedges]; // GPolylla::seed_edges_ad (bit-vector)
            h_scan = new int[n_halfedges];           // GPolylla::seed_edges_bd (scan result)
            h_seed_edges_comp = new int[n_halfedges]; // GPolylla::seed_edges_d (compacted)

            // 4. Inicializar memoria a 0
            memset(h_max_edges, 0, sizeof(bit_vector_d)*n_halfedges);
            memset(h_frontier_edges, 0, sizeof(bit_vector_d)*n_halfedges);
            memset(h_seed_edges, 0, sizeof(int)*n_halfedges); 
            memset(h_scan, 0, sizeof(int)*n_halfedges);
            memset(h_seed_edges_comp, 0, sizeof(int)*n_halfedges);
            std::cout << "[DEBUG] Polylla::Constructor: Buffers alocados e inicializados." << std::endl;

            // 5. Ejecutar el algoritmo
            construct_Polylla();
        }

        // Destructor (NUEVO)
        ~Polylla(){
            std::cout << "[DEBUG] Polylla::Destructor ~Polylla() llamado." << std::endl;
            // Liberar triangulaciones
            if (mesh_input) delete mesh_input;
            if (mesh_output) delete mesh_output;

            // Liberar buffers de trabajo
            delete[] h_max_edges;
            delete[] h_frontier_edges;
            delete[] h_seed_edges; 
            delete[] h_scan;
            delete[] h_seed_edges_comp;
            
            std::cout << "[DEBUG] Polylla::Destructor ~Polylla: Fin." << std::endl;
        }


    // --- FUNCIONES PÚBLICAS ---

    // Función print_OFF (original de GPolylla)
    void print_OFF(std::string filename){
        std::ofstream out(filename);

        // GPolylla usa mesh_input para vértices y m_polygons
        std::cout << "Printing OFF file " <<  mesh_input->vertices() << " " << m_polygons << std::endl;

        out<<"OFF"<<std::endl;
        //num_vertices num_polygons 0
        out<<std::setprecision(15)<<mesh_input->vertices()<<" "<<m_polygons<<" 0"<<std::endl;
        
        //print nodes (desde mesh_input)
        for(std::size_t v = 0; v < mesh_input->vertices(); v++)
            out<<mesh_input->get_PointX(v)<<" "<<mesh_input->get_PointY(v)<<" 0"<<std::endl; 
        
        //print polygons (usa output_seeds y mesh_output)
        int size_poly;
        int e_curr;
        for(auto &e_init : output_seeds){
            size_poly = 1;
            e_curr = mesh_output->next(e_init);
            while(e_init != e_curr){
                size_poly++;
                e_curr = mesh_output->next(e_curr);
            }
            out<<size_poly<<" ";            
            
            out<<mesh_output->origin(e_init)<<" ";
            e_curr = mesh_output->next(e_init);
            while(e_init != e_curr){
                out<<mesh_output->origin(e_curr)<<" ";
                e_curr = mesh_output->next(e_curr);
            }
            out<<std::endl; 
        }
        out.close();
    }

    // Función print_stats (NUEVA, adaptada de GPolylla)
    void print_stats(std::string filename){
        std::ofstream out(filename);
        std::cout<<"Printing JSON file as "<<filename<<std::endl;
        
        double t_total = t_label_max_edges_d + t_label_frontier_edges_d + t_label_seed_edges_d + 
                         t_label_extra_frontier_edge_d + t_traversal_1_d + t_traversal_2_d + 
                         t_overwrite_seed_d + t_label_seed_scan_d + t_label_seed_compaction_d;

        out<<"{"<<std::endl;
        out<<"\"parallel\": "<< 1 <<","<<std::endl; // 1 para OMP (0 es secuencial)
        out<<"\"n_polygons\": "<< m_polygons <<","<<std::endl;
        out<<"\"n_half_edges\": "<< n_halfedges <<","<<std::endl;
        out<<"\"n_faces\": "<< n_faces <<","<<std::endl;
        out<<"\"n_vertices\": "<< n_vertices <<","<<std::endl;
        
        // Tiempos (adaptados a las fases de OMP)
        out<<"\"time_triangulation_generation\": "<< (mesh_input ? mesh_input->get_triangulation_generation_time() : 0) <<","<<std::endl;

        out<<"\"d_time_to_label_max_edges\": "<< t_label_max_edges_d <<","<<std::endl;
        out<<"\"d_time_to_label_frontier_edges\": "<< t_label_frontier_edges_d <<","<<std::endl;
        out<<"\"d_time_to_label_seed_edges\": "<< t_label_seed_edges_d <<","<<std::endl;
        out<<"\"d_time_to_label_extra_frontier_edge\": "<< t_label_extra_frontier_edge_d <<","<<std::endl;
        out<<"\"d_time_to_traversal\": "<< t_traversal_1_d <<","<<std::endl;
        out<<"\"d_time_to_traversal_search_frontier_edge\": "<< t_traversal_2_d <<","<<std::endl;
        out<<"\"d_time_to_overwrite_seed\": "<< t_overwrite_seed_d <<","<<std::endl;
        out<<"\"d_time_to_label_scan_edges\": "<< t_label_seed_scan_d <<","<<std::endl;
        out<<"\"d_time_to_label_compaction_edges\": "<< t_label_seed_compaction_d <<","<<std::endl;
        out<<"\"d_time_to_generate_polygonal_mesh\": "<< t_total <<std::endl;
        
        out<<"}"<<std::endl;
        out.close();
    }
};