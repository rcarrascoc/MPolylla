#pragma once

#include "triangulation.h" // <--- Incluye la nueva CLASE Triangulation
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric> 
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <omp.h>
#include <algorithm> 

//#define DEBUG

// Here starts the "parallel cpu" code!!!!!!!

#include <cstdint>
typedef uint8_t bit_vector_d;

inline int Equality(double a, double b, double epsilon) {
  return fabs(a - b) < epsilon;
}
 
inline int GreaterEqualthan(double a, double b, double epsilon){
    return Equality(a,b,epsilon) || a > b;
}

// Calcula el borde más largo de un triángulo (cara)
// 'e' es *cualquier* half-edge de esa cara.
inline int compute_max_edge_d(Triangulation &tr, int e){
    double epsion = 0.0000000001f;
    int e_next = tr.next(e);
    int e_prev = tr.prev(e);
    int v0 = tr.origin(e);
    int v1 = tr.origin(e_next);
    int v2 = tr.origin(e_prev);

    double dx0 = tr.Vertices[v0].x - tr.Vertices[v1].x;
    double dy0 = tr.Vertices[v0].y - tr.Vertices[v1].y;
    double l0 = dx0 * dx0 + dy0 * dy0;

    double dx1 = tr.Vertices[v1].x - tr.Vertices[v2].x;
    double dy1 = tr.Vertices[v1].y - tr.Vertices[v2].y;
    double l1 = dx1 * dx1 + dy1 * dy1;

    double dx2 = tr.Vertices[v2].x - tr.Vertices[v0].x;
    double dy2 = tr.Vertices[v2].y - tr.Vertices[v0].y;
    double l2 = dx2 * dx2 + dy2 * dy2;

   if( (GreaterEqualthan(l0,l1,epsion) && GreaterEqualthan(l1,l2,epsion)) || ( GreaterEqualthan(l0,l2,epsion) && GreaterEqualthan(l2,l1,epsion)))
   {
           return e;
   }
   else if((GreaterEqualthan(l1,l0,epsion) && GreaterEqualthan(l0,l2,epsion)) || ( GreaterEqualthan(l1,l2,epsion) && GreaterEqualthan(l2,l0,epsion)))
   {
           return e_next;
   }
   else
   {
           return e_prev;
   }
}

inline bool is_frontier_edge_d(Triangulation &tr, bit_vector_d *max_edges, const int e) {
    int twin = tr.twin(e);
    bool is_border_edge = tr.is_border_face(e) || tr.is_border_face(twin);
    bool is_not_max_edge = !(max_edges[e] || max_edges[twin]);
    if(is_border_edge || is_not_max_edge)
        return 1;
    else
        return 0;
}

void label_phase(Triangulation &tr, bit_vector_d *max_edges, bit_vector_d *frontier_edges, int n){
    #ifdef DEBUG 
    std::cout << "[DEBUG] kernel_omp.h: label_phase(n = " << n << ") llamado." << std::endl;
    #endif
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int t = omp_get_thread_num();
                int chunk_size = n / num_threads;
                int remainder = n % num_threads;
                int start = t * chunk_size + std::min(t, remainder);
                int end = (t + 1) * chunk_size + std::min(t + 1, remainder);

                #pragma omp task firstprivate(start, end)
                {
                    for (int off = start; off < end; off++){
                        frontier_edges[off] = 0;
                        if(is_frontier_edge_d(tr, max_edges, off))
                            frontier_edges[off] = 1;
                    }
                }
        #pragma omp taskwait
    }
}

void label_edges_max_d(Triangulation &tr, bit_vector_d *output, int n_faces) {
    #ifdef DEBUG 
    std::cout << "[DEBUG] kernel_omp.h: label_edges_max_d(n_faces = " << n_faces << ") llamado." << std::endl;
    #endif
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int t = omp_get_thread_num();
                int chunk_size = n_faces / num_threads;
                int remainder = n_faces % num_threads;
                int start = t * chunk_size + std::min(t, remainder);
                int end = (t + 1) * chunk_size + std::min(t + 1, remainder);

                #pragma omp task firstprivate(start, end)
                {
                    for(int off = start; off < end; off++) {
                        // tr.incident_halfedge(off) es 3*off
                        int edge_max_index = compute_max_edge_d(tr, tr.incident_halfedge(off));
                        output[edge_max_index] = 1;
                    }
                }
        #pragma omp taskwait
    }
}

inline bool is_seed_edge_d(Triangulation &tr, bit_vector_d *max_edges, int e){
    int twin = tr.twin(e);

    bool is_terminal_edge = (tr.is_interior_face(twin) &&  (max_edges[e] && max_edges[twin]) );
    bool is_terminal_border_edge = (tr.is_border_face(twin) && max_edges[e]);

    if( (is_terminal_edge && e < twin ) || is_terminal_border_edge){
        return true;
    }

    return false;
}

// Modificado para usar 'int' como en GPolylla (en lugar de float)
void seed_phase_d(Triangulation &tr, bit_vector_d *max_edges, bit_vector_d *seed_edges, int n_halfedges){
    #ifdef DEBUG 
    std::cout << "[DEBUG] kernel_omp.h: seed_phase_d(n = " << n_halfedges << ") llamado." << std::endl;
    #endif
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int t = omp_get_thread_num();
                int chunk_size = n_halfedges / num_threads;
                int remainder = n_halfedges % num_threads;
                int start = t * chunk_size + std::min(t, remainder);
                int end = (t + 1) * chunk_size + std::min(t + 1, remainder);

                #pragma omp task firstprivate(start, end)
                {
                    for (int off = start; off < end; off++){
                        if(tr.is_interior_face(off) && is_seed_edge_d(tr, max_edges, off))
                            seed_edges[off] = 1; // 1 en lugar de 1.0f
                    }
                }
        #pragma omp taskwait
    }
}


// -------------------------------------------------------------------
// INICIO: Funciones de Scan/Compaction (Serial, como en GPolylla)
// -------------------------------------------------------------------

/**
 * @brief (GPolylla: Fase 8 - Scan)
 * Calcula la suma de prefijos (scan exclusivo) de forma serial.
 *
 * @param out Puntero al array de salida (h_scan)
 * @param in Puntero al array de entrada (h_seed_edges, tipo int)
 * @param n Número de elementos
 */
template <typename T, typename V>
void compaction_serial_omp(T *output, int *h_num, const V *auxiliary, int size) {
    #ifdef DEBUG 
    std::cout << "[DEBUG] kernel_omp.h: compaction_serial_omp(size = " << size << ") llamado." << std::endl;
    #endif

    int j = 0;
    for (int i = 0; i < size; ++i) {
        if (auxiliary[i] == 1) {
            output[j++] = static_cast<T>(i); // store index i
        }
    }
    *h_num = j;
}


/**
 * @brief (GPolylla: Fase 9 - Compaction)
 * Compactación serial (un solo hilo) de un bit-vector.
 * Basado en la función 'compaction_parallel' (corregida) que proporcionaste.
 * Guarda los *índices* (i) en el array 'output'.
 *
 * @param output El array de salida (h_seed_edges_comp)
 * @param h_num Puntero al contador total (m_polygons)
 * @param auxiliary El bit-vector de entrada (h_seed_edges, tipo int)
 * @param size El número total de halfedges (n_halfedges)
 */
template <typename T, typename V>
void compaction_parallel(T *output, int *h_num, const V *auxiliary, int size){
    int total_elements = 0;
    
    // Arreglos compartidos para la suma de prefijos
    int max_threads = omp_get_max_threads();
    std::vector<int> local_counts(max_threads, 0);
    std::vector<int> offsets(max_threads, 0);
    
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int t = omp_get_thread_num();
        
        int chunk_size = size / num_threads;
        int remainder = size % num_threads;
        int start = t * chunk_size + std::min(t, remainder);
        int end = (t + 1) * chunk_size + std::min(t + 1, remainder);

        // FASE 1: Conteo Local
        int my_count = 0;
        for (int i = start; i < end; i++) {
            if (auxiliary[i] == 1) my_count++;
        }
        local_counts[t] = my_count;

        #pragma omp barrier

        // FASE 2: Suma de prefijos (Un solo hilo)
        #pragma omp single
        {
            int current_offset = 0;
            for (int i = 0; i < num_threads; i++) {
                offsets[i] = current_offset;
                current_offset += local_counts[i];
            }
            total_elements = current_offset;
        }

        // FASE 3: Escritura paralela (Almacenamos el índice 'i')
        int write_index = offsets[t];
        for (int i = start; i < end; i++) {
            if (auxiliary[i] == 1) {
                output[write_index++] = static_cast<T>(i);
            }
        }
    }
    
    *h_num = total_elements;
}



inline int search_next_frontier_edge_d(Triangulation &tr, bit_vector_d *frontier_edges, const int e) {
    int nxt = e;
    while(!frontier_edges[nxt])
    {
        nxt = tr.CW_edge_to_vertex(nxt);
    }  
    return nxt;
}

inline int search_prev_frontier_edge_d(Triangulation &tr, bit_vector_d *frontier_edges, const int e) {
    int prv = e;
    while(!frontier_edges[prv])
    {
        prv = tr.CCW_edge_to_vertex(prv);
    }  
    return prv;
}

void travel_phase_d(Triangulation &tr, Triangulation &output_tr, bit_vector_d *max_edges, bit_vector_d *frontier_edges, int n_halfedges){
    #ifdef DEBUG 
    std::cout << "[DEBUG] kernel_omp.h: travel_phase_d(n = " << n_halfedges << ") llamado." << std::endl;
    #endif
    
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int t = omp_get_thread_num();
        int chunk_size = n_halfedges / num_threads;
        int remainder = n_halfedges % num_threads;
        int start = t * chunk_size + std::min(t, remainder);
        int end = (t + 1) * chunk_size + std::min(t + 1, remainder);

        // Fused: copy and assign next/prev in a single pass to eliminate redundant memory traffic and barriers
        #pragma omp task firstprivate(start, end)
        {
            for (int off = start; off < end; off++){
                halfEdge he = tr.HalfEdges[off];
                he.next = search_next_frontier_edge_d(tr, frontier_edges, he.next);
                he.prev = search_prev_frontier_edge_d(tr, frontier_edges, he.prev);
                output_tr.HalfEdges[off] = he;
            }
        }
        #pragma omp taskwait
    }
}


inline int calculate_middle_edge_d(Triangulation &tr, bit_vector_d *frontier_edges, const int v){
    int frontieredge_with_bet = search_next_frontier_edge_d(tr, frontier_edges, tr.edge_of_vertex(v));
    int internal_edges = tr.degree(v) - 1; //internal-edges incident to v
    int adv = (internal_edges%2 == 0) ? internal_edges/2 - 1 : internal_edges/2 ;
    int nxt = tr.CW_edge_to_vertex(frontieredge_with_bet);
    //back to traversing the edges of v_bet until select the middle-edge
    while (adv != 0){
        nxt = tr.CW_edge_to_vertex(nxt);
        adv--;
    }
    return nxt;
}

//Return the number of frontier edges of a vertex
inline int count_frontier_edges_d(Triangulation &tr, bit_vector_d *frontier_edges, int v){
    int e_start = tr.edge_of_vertex(v);
    if (e_start == -1) return 0; // Vértice aislado

    int e = e_start;
    int count = 0;
    do{
        if(frontier_edges[e] == 1)
            count++;
        e = tr.CW_edge_to_vertex(e);
    } while(e != e_start);
    
    return count;
}

// new repair phase (single-pass 1-ring traversal with degree and frontier detection fused)
void label_extra_frontier_edge_d(Triangulation &tr, bit_vector_d *frontier_edges, bit_vector_d *seed_edges, int n_vertices){
    #ifdef DEBUG 
    std::cout << "[DEBUG] kernel_omp.h: label_extra_frontier_edge_d(n_vertices = " << n_vertices << ") llamado." << std::endl;
    #endif
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int t = omp_get_thread_num();
        int chunk_size = n_vertices / num_threads;
        int remainder = n_vertices % num_threads;
        int start = t * chunk_size + std::min(t, remainder);
        int end = (t + 1) * chunk_size + std::min(t + 1, remainder);

        #pragma omp task firstprivate(start, end)
        {
            for (int v = start; v < end; v++){
                int e_start = tr.edge_of_vertex(v);
                if (e_start == -1) continue;

                int e = e_start;
                int count = 0;
                int degree = 0;
                int frontier_edge = -1;

                do {
                    degree++;
                    if (frontier_edges[e] == 1) {
                        count++;
                        frontier_edge = e;
                    }
                    e = tr.CW_edge_to_vertex(e);
                } while (e != e_start);

                if (count == 1) {
                    int internal_edges = degree - 1;
                    int adv = (internal_edges % 2 == 0) ? internal_edges / 2 - 1 : internal_edges / 2;
                    int nxt = tr.CW_edge_to_vertex(frontier_edge);
                    while (adv > 0) {
                        nxt = tr.CW_edge_to_vertex(nxt);
                        adv--;
                    }

                    int t1 = nxt;
                    int t2 = tr.twin(nxt);

                    frontier_edges[t1] = 1;
                    frontier_edges[t2] = 1;

                    seed_edges[t1] = 1;
                    seed_edges[t2] = 1;
                }
            }
        }
        #pragma omp taskwait
    }  
}

// Modificado para usar int*
void search_frontier_edge_d(Triangulation &tr, bit_vector_d *frontier_edges,  bit_vector_d *seed_edges, int n_halfedges) {
    #ifdef DEBUG 
    std::cout << "[DEBUG] kernel_omp.h: search_frontier_edge_d(n = " << n_halfedges << ") llamado." << std::endl;
    #endif
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int t = omp_get_thread_num();
                int chunk_size = n_halfedges / num_threads;
                int remainder = n_halfedges % num_threads;
                int start = t * chunk_size + std::min(t, remainder);
                int end = (t + 1) * chunk_size + std::min(t + 1, remainder);

                #pragma omp task firstprivate(start, end)
                {
                    for (int off = start; off < end; off++){
                        if(seed_edges[off] == 1){
                            int nxt = off;
                            while(!frontier_edges[nxt])
                            {
                                nxt = tr.CW_edge_to_vertex(nxt);
                            }  
                            if(nxt != off)
                                seed_edges[off] = 0;
                            
                            seed_edges[nxt] = 1;
                        }
                    }
                }
        #pragma omp taskwait
    }
}

// Modificado para usar int*
void overwrite_seed_d(Triangulation &tr, bit_vector_d *seed_edges, int n_halfedges){
    #ifdef DEBUG 
    std::cout << "[DEBUG] kernel_omp.h: overwrite_seed_d(n = " << n_halfedges << ") llamado." << std::endl;
    #endif
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int t = omp_get_thread_num();
                int chunk_size = n_halfedges / num_threads;
                int remainder = n_halfedges % num_threads;
                int start = t * chunk_size + std::min(t, remainder);
                int end = (t + 1) * chunk_size + std::min(t + 1, remainder);

                #pragma omp task firstprivate(start, end)
                {
                    for (int i = start; i < end; i++){        
                        if(seed_edges[i] == 1){
                            int e_init = i;
                            int min_ind = e_init;
                            int e_curr = tr.next(e_init);
                            while(e_init != e_curr){
                                min_ind = std::min(min_ind, e_curr);
                                e_curr = tr.next(e_curr);
                            }
                            
                            if(min_ind != i){
                                seed_edges[i] = 0;
                            }
                            
                            seed_edges[min_ind] = 1;
                        }
                    }
                }
        #pragma omp taskwait
    }  
}