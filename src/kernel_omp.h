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

// Here starts the "gpu" code!!!!!!!

typedef int bit_vector_d;

// NOTA: Las funciones 'inline' 'twin_d', 'next_d', 'prev_d', 'origin_d', etc.,
// han sido eliminadas. Ahora usaremos los métodos de la clase Triangulation,
// como tr.twin(e), tr.next(e), etc.

// Las funciones 'kernel' ahora toman una referencia a la clase Triangulation
// para poder llamar a sus métodos.

// --- Implementación de funciones auxiliares que SÍ estaban en kernel.cu ---

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
    // Llama al método distance() de la clase Triangulation
    double l0 = tr.distance(e); 
    double l1 = tr.distance(tr.next(e)); 
    double l2 = tr.distance(tr.prev(e)); 

   if( (GreaterEqualthan(l0,l1,epsion) && GreaterEqualthan(l1,l2,epsion)) || ( GreaterEqualthan(l0,l2,epsion) && GreaterEqualthan(l2,l1,epsion)))
   {
           return e;
   }
   else if((GreaterEqualthan(l1,l0,epsion) && GreaterEqualthan(l0,l2,epsion)) || ( GreaterEqualthan(l1,l2,epsion) && GreaterEqualthan(l2,l0,epsion)))
   {
           return tr.next(e);
   }
   else
   {
           return tr.prev(e);
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
    #pragma omp parallel for
    for (int off = 0; off < n; off++){
        frontier_edges[off] = 0;
        if(is_frontier_edge_d(tr, max_edges, off))
            frontier_edges[off] = 1;
    }
}

void label_edges_max_d(Triangulation &tr, bit_vector_d *output, int n_faces) {
    #ifdef DEBUG 
    std::cout << "[DEBUG] kernel_omp.h: label_edges_max_d(n_faces = " << n_faces << ") llamado." << std::endl;
    #endif
    #pragma omp parallel for
    for(int off = 0; off < n_faces; off++) {
        // tr.incident_halfedge(off) es 3*off
        int edge_max_index = compute_max_edge_d(tr, tr.incident_halfedge(off));
        
         
        output[edge_max_index] = 1;
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
void seed_phase_d(Triangulation &tr, bit_vector_d *max_edges, int *seed_edges, int n_halfedges){
    #ifdef DEBUG 
    std::cout << "[DEBUG] kernel_omp.h: seed_phase_d(n = " << n_halfedges << ") llamado." << std::endl;
    #endif
    #pragma omp parallel for
    for (int off = 0; off < n_halfedges; off++){
        if(tr.is_interior_face(off) && is_seed_edge_d(tr, max_edges, off))
            seed_edges[off] = 1; // 1 en lugar de 1.0f
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
void compaction_parallel(T *output, int *h_num, V *auxiliary, T *input, int size){
    int i, j=0;
    #pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        if (auxiliary[i] == 1)
        {
            output[j] = input[i];
            j++;
        }
    }
    *h_num = j;
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
    
    // Copiamos los halfedges de 'tr' (entrada) a 'output_tr' (salida)
    #pragma omp parallel for
    for (int off = 0; off < n_halfedges; off++){
        output_tr.HalfEdges[off] = tr.HalfEdges[off];
    }
    
    #pragma omp parallel for
    for (int off = 0; off < n_halfedges; off++){
        // Modificamos los punteros next/prev en la *copia de salida*
        output_tr.HalfEdges[off].next = search_next_frontier_edge_d(tr, frontier_edges, tr.next(off));
        output_tr.HalfEdges[off].prev = search_prev_frontier_edge_d(tr, frontier_edges, tr.prev(off));
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
    
    // CORRECCIÓN DE BUG: El bucle debe comparar con la arista inicial, no
    // llamar a edge_of_vertex(v) en cada iteración.
    
    int e_start = tr.edge_of_vertex(v);
    if (e_start == -1) return 0; // Vértice aislado

    int e = e_start;
    int count = 0;
    do{
        if(frontier_edges[e] == 1)
            count++;
        e = tr.CW_edge_to_vertex(e);
    } while(e != e_start); // <--- Bucle corregido
    
    return count;
}

// new repair phase (modificado para usar int*)
void label_extra_frontier_edge_d(Triangulation &tr, bit_vector_d *frontier_edges, int *seed_edges, int n_vertices){
    #ifdef DEBUG 
    std::cout << "[DEBUG] kernel_omp.h: label_extra_frontier_edge_d(n_vertices = " << n_vertices << ") llamado." << std::endl;
    #endif
    #pragma omp parallel for
    for (int v = 0; v < n_vertices; v++){

        if(count_frontier_edges_d(tr, frontier_edges, v) == 1){
            int middle_edge = calculate_middle_edge_d(tr, frontier_edges, v);

            int t1 = middle_edge;
            int t2 = tr.twin(middle_edge);

            //edges of middle-edge are labeled as frontier-edge
             
            frontier_edges[t1] = 1;
             
            frontier_edges[t2] = 1;

             
            seed_edges[t1] = 1;
             
            seed_edges[t2] = 1;
        }
    }  
}

// Modificado para usar int*
void search_frontier_edge_d(Triangulation &tr, bit_vector_d *frontier_edges,  int *seed_edges, int n_halfedges) {
    #ifdef DEBUG 
    std::cout << "[DEBUG] kernel_omp.h: search_frontier_edge_d(n = " << n_halfedges << ") llamado." << std::endl;
    #endif
    #pragma omp parallel for
    for (int off = 0; off < n_halfedges; off++){
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

// Modificado para usar int*
void overwrite_seed_d(Triangulation &tr, int *seed_edges, int n_halfedges){
    #ifdef DEBUG 
    std::cout << "[DEBUG] kernel_omp.h: overwrite_seed_d(n = " << n_halfedges << ") llamado." << std::endl;
    #endif
    #pragma omp parallel for
    for (int i = 0; i < n_halfedges; i++){        
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