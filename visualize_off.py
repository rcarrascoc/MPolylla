#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.collections as collections

def read_off(filename):
    """
    Parses a 2D/3D OFF file.
    Returns:
        vertices: numpy array of shape (N, 3)
        faces: list of lists containing vertex indices for each polygon
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    clean_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        clean_lines.append(line)
        
    if not clean_lines:
        raise ValueError(f"El archivo {filename} está vacío o no es un OFF válido.")
        
    header = clean_lines[0]
    idx = 1
    if header.startswith('OFF') or header.startswith('off'):
        tokens = header.split()
        if len(tokens) > 1:
            n_verts = int(tokens[1])
            n_faces = int(tokens[2])
        else:
            if idx < len(clean_lines):
                tokens = clean_lines[idx].split()
                n_verts = int(tokens[0])
                n_faces = int(tokens[1])
                idx += 1
            else:
                raise ValueError("No se pudieron leer las cantidades de vértices/caras.")
    else:
        tokens = header.split()
        n_verts = int(tokens[0])
        n_faces = int(tokens[1])
    
    vertices = []
    for _ in range(n_verts):
        if idx >= len(clean_lines):
            raise ValueError(f"Se esperaban {n_verts} vértices, pero el archivo terminó antes.")
        tokens = clean_lines[idx].split()
        vertices.append([float(tokens[0]), float(tokens[1]), float(tokens[2])])
        idx += 1
        
    faces = []
    for _ in range(n_faces):
        if idx >= len(clean_lines):
            raise ValueError(f"Se esperaban {n_faces} caras, pero el archivo terminó antes.")
        tokens = clean_lines[idx].split()
        n_face_verts = int(tokens[0])
        face_indices = [int(x) for x in tokens[1:1+n_face_verts]]
        faces.append(face_indices)
        idx += 1
        
    return np.array(vertices), faces

def plot_mesh(vertices, faces, title="Mesh Visualization", save_path=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Dibujar polígonos
    patches = []
    # Generar colores distintos para los polígonos usando una paleta armónica
    cmap = plt.cm.get_cmap('rainbow', len(faces))
    
    # Para evitar colores idénticos juntos, podemos desordenar los índices de colores de forma pseudoaleatoria
    np.random.seed(42)
    color_indices = np.random.permutation(len(faces))
    
    for i, face in enumerate(faces):
        poly_coords = vertices[face][:, :2]  # Usar solo X e Y
        polygon = MplPolygon(poly_coords, closed=True, 
                             facecolor=cmap(color_indices[i]), 
                             edgecolor='black', 
                             linewidth=1.2, 
                             alpha=0.6)
        ax.add_patch(polygon)
        
    # Dibujar puntos de los vértices
    ax.scatter(vertices[:, 0], vertices[:, 1], color='darkblue', s=12, zorder=3, label='Vértices')
    
    # Auto-ajustar límites
    x_min, y_min = np.min(vertices[:, :2], axis=0)
    x_max, y_max = np.max(vertices[:, :2], axis=0)
    padding = max((x_max - x_min) * 0.05, (y_max - y_min) * 0.05, 0.1)
    
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_aspect('equal')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Agregar leyenda sencilla
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', markersize=8, label='Vértices'),
        Line2D([0], [0], color='black', lw=1.5, label='Bordes Polígonos')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Visualización guardada con éxito en: {save_path}")
        
    try:
        # Intentar mostrar si hay entorno gráfico disponible
        plt.show()
    except Exception as e:
        print("No se pudo mostrar la ventana interactiva (entorno sin GUI).")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 visualize_off.py <archivo.off> [archivo_salida.png]")
        print("Ejemplo: python3 visualize_off.py output_100.off")
        sys.exit(1)
        
    input_file = sys.argv[1]
    
    if len(sys.argv) > 2:
        output_image = sys.argv[2]
    else:
        basename = os.path.splitext(os.path.basename(input_file))[0]
        output_image = f"{basename}.png"
        
    print(f"Cargando malla desde {input_file}...")
    try:
        vertices, faces = read_off(input_file)
        print(f"Malla leída con éxito: {len(vertices)} vértices y {len(faces)} polígonos/caras.")
        plot_mesh(vertices, faces, title=f"Malla: {os.path.basename(input_file)}", save_path=output_image)
    except Exception as e:
        print(f"Error al visualizar la malla: {e}", file=sys.stderr)
        sys.exit(1)
