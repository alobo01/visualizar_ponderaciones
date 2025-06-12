import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network as PyvisNetwork
import re # Added import
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import tempfile # Added import
import os # Added import

# --- Definiciones Globales y Constantes ---
DATA_FILE = 'ponderaciones_andalucia.csv' # Asegúrate que este archivo está en el mismo directorio

# Definición de Relaciones 1º -> 2º Bachillerato (Movida a Escopo Global)
RELACIONES_1_A_2 = {
    'Matemáticas_I': ['Matemáticas_II'],
    'Mates_Aplicadas_CCSS_I': ['Matemáticas_Aplicadas_CC.SS.'],
    'Física_y_Química': ['Física', 'Química'],
    'Biología_y_Geología': ['Biología', 'Geología_y_Ciencias_Ambientales'],
    'Dibujo_Técnico_I': ['Dibujo_Técnico_II', 'Dibujo_Técnico_aplicado_a_las_artes_plásticas_y_al_diseño_II'],
    'Latín_I': ['Latín_II'],
    'Griego_I': ['Griego_II'],
    'Economía': ['Empresa_y_Diseño_de_modelos_de_negocio'],
    'Hª_Mundo_Contemporáneo': ['Historia_de_la_Filosofía', 'Historia_del_Arte', 'Geografía']
}

# --- Funciones de generate_flow_graph.py (adaptadas o importadas) ---

def cargar_y_limpiar_csv(filepath):
    try:
        # Try with utf-8 first
        df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo '{filepath}'. Asegúrate de que el archivo 'ponderaciones_andalucia.csv' está en el mismo directorio que la aplicación.")
        return None, ""
    except UnicodeDecodeError:
        st.warning("Error de decodificación UTF-8. Probando con 'latin1' como alternativa.")
        try:
            df = pd.read_csv(filepath, encoding='latin1', on_bad_lines='skip')
        except Exception as e:
            st.error(f"Error al cargar con Latin1: {e}")
            return None, ""

    if df.empty:
        st.error("El archivo CSV está vacío o no se pudo cargar correctamente.")
        return None, ""

    legend_text_display = ""
    if not df.empty:
        first_col_name = df.columns[0]
        if isinstance(first_col_name, str) and "Ramas de Conocimiento:" in first_col_name:
            legend_content_full = first_col_name
            legend_actual_content = legend_content_full.split("Grados")[0].strip()
            
            title_part, items_part = legend_actual_content.split(":", 1)
            title = title_part.strip() + ":"
            
            # Normalize various newline representations to '\\n', then split.
            # Handles \\r\\n (as a literal string), \\r\\n (CRLF chars), \\n (LF char), \\r (CR char).
            normalized_items_text = items_part.strip().replace("\\\\r\\\\n", "\\n").replace("\\r\\n", "\\n").replace("\\n", "<br>").replace("\\r", "<br>")
            items_list_raw = [item.strip() for item in normalized_items_text.split('\\n') if item.strip()]

            formatted_items = []
            for item_line in items_list_raw: # item_line is already stripped and non-empty from the list comprehension
                if ":" in item_line:
                    abbrev, desc = item_line.split(":", 1)
                    formatted_items.append(f"- **{abbrev.strip()}**: {desc.strip()}")
                else: # item_line is guaranteed to be a non-empty string here
                    formatted_items.append(f"- {item_line}")
            
            if formatted_items:
                legend_text_display = f"**{title}**<br>" + "<br>".join(formatted_items)
            else: # Fallback if parsing was difficult or items_part was empty/malformed
                processed_items_part_fb = items_part.strip()
                # Replace all known newline variations with <br>
                processed_items_part_fb = processed_items_part_fb.replace("\\r\\n", "<br>")
                processed_items_part_fb = processed_items_part_fb.replace("\r\n", "<br>")
                processed_items_part_fb = processed_items_part_fb.replace("\n", "<br>")
                processed_items_part_fb = processed_items_part_fb.replace("\r", "<br>")
                
                # Clean up multiple <br> tags that might result from original multiple newlines
                while "<br><br>" in processed_items_part_fb:
                    processed_items_part_fb = processed_items_part_fb.replace("<br><br>", "<br>")
                
                # Remove leading/trailing <br> if any, after all replacements
                if processed_items_part_fb.startswith("<br>"):
                    processed_items_part_fb = processed_items_part_fb[4:]
                if processed_items_part_fb.endswith("<br>"):
                    processed_items_part_fb = processed_items_part_fb[:-4]
                
                legend_text_display = f"**Leyenda Ramas:**<br>{processed_items_part_fb}"


    df.rename(columns={df.columns[0]: 'Grado'}, inplace=True)
    
    if 'Rama de conocimiento' in df.columns:
        df = df.dropna(subset=['Rama de conocimiento'])
    else:
        st.error("La columna 'Rama de conocimiento' no se encuentra en el CSV. Verifica el formato del archivo.")
        return None, legend_text_display
        
    df = df[~df['Grado'].str.contains("Pondera|No pondera|This work", case=False, na=False)]
    
    nuevas_columnas = {col: col.strip().replace(' ', '_').replace('-', '_') for col in df.columns}
    df.rename(columns=nuevas_columnas, inplace=True)

    # Handle "IyA+C" by duplicating rows
    rama_col_name = 'Rama_de_conocimiento' # After potential rename by nuevas_columnas
    if rama_col_name in df.columns:
        new_rows = []
        for _, row in df.iterrows():
            if row[rama_col_name] == 'IyA+C':
                row_iya = row.copy()
                row_iya[rama_col_name] = 'Ingeniería y Arquitectura'
                new_rows.append(row_iya)
                
                row_c = row.copy()
                row_c[rama_col_name] = 'Ciencias'
                new_rows.append(row_c)
            else:
                new_rows.append(row)
        df = pd.DataFrame(new_rows).reset_index(drop=True)
    
    columnas_asignaturas = [col for col in df.columns if col not in ['Grado', rama_col_name]]

    for col in columnas_asignaturas:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.fillna(0.0, inplace=True)
    return df, legend_text_display

def generar_diagrama_networkx_pyvis(df_data, rama_filter_display_name, mostrar_ponderacion_015=False, mostrar_ponderacion_01=False, alto_px=800, ancho_px=1000, selected_node_id=None):
    """
    Genera un diagrama interactivo usando NetworkX para la lógica y Pyvis para la visualización.
    Permite filtrar por un nodo seleccionado.
    """
    
    # Detectar si el nodo seleccionado tiene prefijo y extraer su tipo y nombre base
    node_type = None
    node_base_name = None
    
    if selected_node_id:
        if selected_node_id.startswith("1bach_"):
            node_type = "1bach"
            node_base_name = selected_node_id[6:]  # Quitar "1bach_"
        elif selected_node_id.startswith("2bach_"):
            node_type = "2bach"
            node_base_name = selected_node_id[6:]  # Quitar "2bach_"
        elif selected_node_id.startswith("grado_"):
            node_type = "grado"
            node_base_name = selected_node_id[6:]  # Quitar "grado_"
        else:
            # Si no tiene prefijo, determinar el tipo por su contenido
            if selected_node_id in RELACIONES_1_A_2:
                node_type = "1bach"
                node_base_name = selected_node_id
            elif selected_node_id in df_data.columns and selected_node_id not in ['Grado', 'Rama_de_conocimiento']:
                node_type = "2bach"
                node_base_name = selected_node_id
            elif selected_node_id in df_data['Grado'].unique():
                node_type = "grado"
                node_base_name = selected_node_id
    
    # Filtrar df_data según el tipo de nodo seleccionado
    if node_type == "1bach" and node_base_name:
        # Si el nodo seleccionado es una asignatura de 1º Bach
        if node_base_name in RELACIONES_1_A_2:
            # Mantener solo las asignaturas de 2º Bach relacionadas y los grados a los que estas conectan
            sucesores_directos = RELACIONES_1_A_2[node_base_name]
            df_data = df_data[df_data.apply(lambda row: any(row[s] > 0 for s in sucesores_directos if s in row), axis=1)]
            # Y también filtrar las columnas de asignaturas de 2º Bach a solo las sucesoras
            cols_a_mantener = ['Grado', 'Rama_de_conocimiento'] + [s for s in sucesores_directos if s in df_data.columns]
            df_data = df_data[cols_a_mantener]
    elif node_type == "2bach" and node_base_name:
        # Si el nodo seleccionado es una asignatura de 2º Bach
        if node_base_name in df_data.columns and node_base_name not in ['Grado', 'Rama_de_conocimiento']:
            df_data = df_data[df_data[node_base_name] > 0] # Mantener solo grados donde esta asignatura pondera
            # Mantener solo esta asignatura de 2º Bach y los grados
            cols_a_mantener = ['Grado', 'Rama_de_conocimiento', node_base_name]
            df_data = df_data[cols_a_mantener]
    elif node_type == "grado" and node_base_name:
        # Si el nodo seleccionado es un Grado
        if node_base_name in df_data['Grado'].unique():
            df_data = df_data[df_data['Grado'] == node_base_name] # Mantener solo este grado
    
    if df_data.empty and selected_node_id:
        st.warning(f"No se encontraron datos relevantes para el nodo seleccionado: {selected_node_id}")
        # For now, let it proceed, Pyvis will just show an empty graph or a very small one.


    # Inicializar Pyvis Network
    # Usamos un nombre de archivo temporal para el gráfico Pyvis
    # pyvis_file = f"temp_graph_{rama_filter_display_name.replace(' ','_')}.html"
    # nt = PyvisNetwork(height=f"{alto_px}px", width="100%", notebook=True, directed=True, cdn_resources='remote') # Usar 100% para ancho responsivo
    
    # Para evitar problemas con `notebook=True` en Streamlit y asegurar que se pueda guardar:
    # Guardamos el HTML y lo leemos para st.components.v1.html
    # Es crucial que el path sea accesible por el servidor de Streamlit.
    # Usar un nombre de archivo único si se generan muchos gráficos dinámicamente para evitar colisiones,
    # o limpiar después. Para este caso, un nombre fijo por rama podría ser suficiente si se actualiza.
    
    # Simplificación: Usar NetworkX para construir y luego convertir a DOT para st.graphviz_chart
    # ya que la interactividad de clic directa con Pyvis en Streamlit es compleja de implementar
    # para el filtrado dinámico sin componentes personalizados.
    # Si la interactividad de arrastrar/zoom es lo principal, Pyvis es bueno, pero el clic para filtrar es el desafío.

    # Re-enfocando en NetworkX + Graphviz para Streamlit por simplicidad de "clic" (usando st.experimental_rerun o callbacks)
    # O, para Pyvis, el "clic" sería más bien una guía visual y el usuario usaría selectores externos.

    # --- Construcción del Grafo con NetworkX (lógica similar a la anterior de Graphviz) ---
    G = nx.DiGraph()

    # Columnas de ponderación (asignaturas de 2º Bach)
    columnas_ponderacion = [col for col in df_data.columns if col not in ['Grado', 'Rama_de_conocimiento']]
    
    asignaturas_2_activas = [col for col in columnas_ponderacion if df_data[col].sum() > 0 or col == selected_node_id]
    if not asignaturas_2_activas and selected_node_id and selected_node_id in columnas_ponderacion:
         asignaturas_2_activas = [selected_node_id] # Asegurar que el nodo seleccionado se incluya si es de 2º Bach
    elif not asignaturas_2_activas:
        asignaturas_2_activas = columnas_ponderacion # Fallback si no hay sum > 0

    asignaturas_1_bach_filtradas = []
    if not selected_node_id or (selected_node_id and selected_node_id in RELACIONES_1_A_2): # Mostrar 1º Bach si no hay filtro o el filtro es de 1º
        for precursor, sucesores in RELACIONES_1_A_2.items(): # Usar la variable global
            if any(sucesor in asignaturas_2_activas for sucesor in sucesores):
                asignaturas_1_bach_filtradas.append(precursor)
    elif selected_node_id and any(selected_node_id in v for v in RELACIONES_1_A_2.values()): # Si el seleccionado es de 2º, mostrar sus precursores
        for k,v in RELACIONES_1_A_2.items(): # Usar la variable global
            if selected_node_id in v:
                asignaturas_1_bach_filtradas.append(k)
    # Añadir nodos con atributos optimizados para layout jerárquico (Sugiyama framework)
    # Capa 1: 1º Bachillerato (nivel 0)
    for i, nodo_1 in enumerate(sorted(asignaturas_1_bach_filtradas)):
        node_id = f"1bach_{nodo_1}"  # Prefijo para nodos de 1º Bach
        G.add_node(
            node_id, 
            level=0,  # Explicit level for hierarchical layout
            layer=1, 
            color='#E6E6FA', 
            title=nodo_1.replace('_', ' '), 
            shape='box', 
            type='1_bach',
            x=None,  # Let hierarchical layout determine position
            y=i * 100,  # Vertical spacing hint
            fixed=False,
            physics=False
        )

    # Capa 2: 2º Bachillerato (nivel 1)
    if asignaturas_2_activas:
        cmap = plt.cm.get_cmap('tab20', len(asignaturas_2_activas))
        color_map_2_bach = {asig: mcolors.to_hex(cmap(i)) for i, asig in enumerate(asignaturas_2_activas)}
    
    for i, nodo_2 in enumerate(sorted(asignaturas_2_activas)):
        node_id = f"2bach_{nodo_2}"  # Prefijo para nodos de 2º Bach
        # Colorear en rojo si es la asignatura seleccionada
        # Ajustamos la comprobación para manejar el nodo seleccionado con o sin prefijo
        is_selected_asignatura = selected_node_id and (nodo_2 == selected_node_id or f"2bach_{nodo_2}" == selected_node_id)
        
        if is_selected_asignatura:
            color = '#FF0000'  # Rojo para asignatura seleccionada
        else:
            color = color_map_2_bach.get(nodo_2, '#D3D3D3')
        G.add_node(
            node_id, 
            level=1,  # Explicit level for hierarchical layout
            layer=2, 
            color=color + 'BF', 
            title=nodo_2.replace('_', ' '), 
            shape='box', 
            type='2_bach',
            x=None,  # Let hierarchical layout determine position
            y=i * 80,  # Vertical spacing hint
            fixed=False,
            physics=False
        )

    # Capa 3: Grados Universitarios (nivel 2)
    grados_en_df = sorted(df_data['Grado'].unique())
    for i, grado_uni in enumerate(grados_en_df):
        node_id = f"grado_{grado_uni}"  # Prefijo para nodos de grado
        G.add_node(
            node_id, 
            level=2,  # Explicit level for hierarchical layout
            layer=3, 
            color='#FFDAB9', 
            title=grado_uni, 
            shape='box', 
            type='grado',
            x=None,  # Let hierarchical layout determine position
            y=i * 60,  # Vertical spacing hint, more compact for many nodes
            fixed=False,
            physics=False
        )    # Conexiones 1º Bach -> 2º Bach (optimizadas para layout jerárquico)
    for precursor, sucesores in RELACIONES_1_A_2.items(): # Usar la variable global
        node_1_id = f"1bach_{precursor}"
        if node_1_id in G: # Si el nodo de 1º Bach está en el grafo
            for sucesor in sucesores:
                node_2_id = f"2bach_{sucesor}"
                if node_2_id in G: # Si el nodo de 2º Bach está en el grafo
                    G.add_edge(
                        node_1_id, 
                        node_2_id, 
                        color='#6A5ACD', 
                        weight=2,
                        width=2,
                        arrows={'to': {'enabled': True, 'scaleFactor': 0.8}},
                        smooth={'type': 'straightCross', 'forceDirection': 'horizontal'},
                        physics=False
                    )    # Conexiones 2º Bach -> Grados (optimizadas para minimizar cruces)
    min_ponderacion_mostrar = 0.19 # Por defecto, solo muestra ponderación 0.2
    if mostrar_ponderacion_01:
        min_ponderacion_mostrar = 0.01 # Muestra todas las ponderaciones mayores que 0
    elif mostrar_ponderacion_015:  # Mantenemos para compatibilidad, pero efectivamente se ignora
        min_ponderacion_mostrar = 0.19 # Sigue mostrando solo 0.2

    # Agrupar conexiones por asignatura para mejor organización
    for asignatura_2 in sorted(asignaturas_2_activas):
        node_2_id = f"2bach_{asignatura_2}"
        if node_2_id not in G: continue # Si la asignatura no está en el grafo (p.ej. por filtrado de nodo)
        color_base_edge = color_map_2_bach.get(asignatura_2, '#808080')
        
        # Recopilar grados que van a conectar para ordenarlos
        grados_a_conectar = []
        for _, row in df_data.iterrows():
            grado = row['Grado']
            grado_id = f"grado_{grado}"
            if grado_id not in G: continue # Si el grado no está en el grafo
            
            ponderacion = row.get(asignatura_2, 0.0)
            if ponderacion >= min_ponderacion_mostrar:
                grados_a_conectar.append((grado, grado_id, ponderacion))
          # Ordenar grados por ponderación (mayor primero) para minimizar cruces
        grados_a_conectar.sort(key=lambda x: x[2], reverse=True)
        
        # Crear conexiones ordenadas
        for grado, grado_id, ponderacion in grados_a_conectar:
            edge_color = color_base_edge
            
            # Determinar estilo de línea y grosor basado en ponderación
            dashes = False # Default to solid

            if ponderacion >= 0.2:
                edge_width = 2.5
                # dashes remains False
            elif ponderacion >= 0.1 and ponderacion < 0.2:  # Covers 0.1 to 0.19
                edge_width = 1.5 # Consistent width for all dashed lines in this range
                dashes = [5, 5]
            else: # For ponderaciones < 0.1, if they are shown (e.g. 0.09)
                edge_width = 1.0 # Thinner solid line for these
                dashes = False # Not dashed as per specific request
            
            edge_title = f"{ponderacion:.2f}"
            
            edge_properties = {
                'color': edge_color, 
                'weight': edge_width, 
                'width': edge_width,
                'title': edge_title, 
                'arrows': {'to': {'enabled': True, 'scaleFactor': 0.8}},
                'smooth': {'type': 'straightCross', 'forceDirection': 'horizontal'},
                'physics': False
            }
            
            # Agregar propiedades de línea discontinua si es necesario
            if dashes:
                edge_properties['dashes'] = dashes
            
            G.add_edge(node_2_id, grado_id, **edge_properties)

    # --- Visualización con Pyvis ---
    if not G.nodes():
        return None # Devuelve None si no hay nodos para evitar errores en Pyvis

    nt = PyvisNetwork(height=f"{alto_px}px", width="100%", notebook=False, directed=True, cdn_resources='remote')
    nt.from_nx(G)

    # Hierarchical layout configuration following Sugiyama framework principles
    options_json = """
    {
      "physics": {
        "enabled": false
      },
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",
          "sortMethod": "directed",
          "shakeTowards": "roots",
          "levelSeparation": 1000,
          "nodeSpacing": 120,
          "treeSpacing": 200,
          "blockShifting": true,
          "edgeMinimization": true,
          "parentCentralization": true,
          "improvedLayout": true
        }
      },
      "interaction": {
        "dragNodes": false,
        "dragView": true,
        "zoomView": true,
        "selectConnectedEdges": true,
        "tooltipDelay": 200,
        "hideEdgesOnDrag": false,
        "hideNodesOnDrag": false
      },
      "nodes": {
        "font": {
          "size": 11,
          "face": "arial",
          "strokeWidth": 2,
          "strokeColor": "#ffffff"
        },
        "borderWidth": 2,
        "borderWidthSelected": 3,
        "chosen": {
          "node": {
            "borderColor": "#2B7CE9",
            "borderWidth": 3
          }
        },
        "shape": "box",
        "margin": 10,
        "widthConstraint": {
          "minimum": 80,
          "maximum": 200
        }
      },
      "edges": {
        "font": {
          "size": 9,
          "align": "top",
          "strokeWidth": 1,
          "strokeColor": "#ffffff"
        },
        "arrows": {
          "to": {
            "enabled": true, 
            "scaleFactor": 0.8,
            "type": "arrow"
          }
        },
        "smooth": {
          "enabled": true,
          "type": "cubicBezier",
          "forceDirection": "horizontal",
          "roundness": 0.7
        },
        "color": {
          "inherit": false,
          "opacity": 0.8
        },
        "width": 2,
        "chosen": {
          "edge": {
            "color": "#2B7CE9",
            "width": 3
          }
        }
      }
    }
    """
    nt.set_options(options_json)
    
    # Guardar en un archivo HTML temporal y luego leerlo
    # Esto es necesario porque st.components.v1.html no toma directamente el objeto nt.
    # Asegúrate de que el directorio temporal es escribible por la app Streamlit.
    # Podrías usar el módulo `tempfile` de Python para crear un archivo temporal de forma segura.
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        nt.save_graph(tmp_file.name)
        html_path = tmp_file.name
    
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    import os
    os.unlink(html_path) # Eliminar el archivo temporal después de leerlo

    return html_content


# --- Configuración de la página de Streamlit ---
st.set_page_config(page_title="Visor Ponderaciones Selectividad Andalucía", layout="wide", initial_sidebar_state="expanded")

st.title("📊 Visor Interactivo de Ponderaciones para Selectividad (Andalucía)")
st.markdown("""
Bienvenido al visor de ponderaciones. Explora cómo las asignaturas de bachillerato 
ponderan para los grados universitarios en Andalucía.
""")

# --- Carga de datos ---
# DATA_FILE ya está definido globalmente
# df_ponderaciones_original = cargar_y_limpiar_csv(DATA_FILE) # Old call
df_ponderaciones_original, leyenda_ramas = cargar_y_limpiar_csv(DATA_FILE) # New call

if df_ponderaciones_original is not None:
    st.sidebar.image("logo.png", use_container_width=True)
    st.sidebar.markdown("<h5 style='text-align: center;'>Rosa María Santos Vilches</h5>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>IES Politécnico Sevilla</p>", unsafe_allow_html=True)

    if leyenda_ramas:
        # Print leyenda_ramas with <br> replaced by newlines, but only lines NOT starting with a dash
        
        st.sidebar.markdown("---")
        st.sidebar.markdown(leyenda_ramas.replace("\n","\n-**").replace("\n","<br>").replace(":","**:").replace("****","**"), unsafe_allow_html=True)
        st.sidebar.markdown("---")

    st.sidebar.header("🛠️ Opciones de Visualización")
    
    # Nombres de ramas directamente del CSV (ya son descriptivos)
    ramas_conocimiento_disponibles = sorted(df_ponderaciones_original['Rama_de_conocimiento'].unique())

    modo_visualizacion = st.sidebar.radio(
        "Selecciona el modo de visualización:",
        ('Gráfico Interactivo de Flujo', 'Tabla de Ponderaciones', 'Calculadora de Nota de Acceso'),
        key='modo_viz'
    )

    if modo_visualizacion == 'Tabla de Ponderaciones':
        st.subheader("📜 Tabla de Ponderaciones")
        
        # Radio button to switch between table views
        vista_tabla = st.radio(
            "Ver tabla por:",
            ("Grados (vista tradicional)", "Asignaturas (qué grados las ponderan)"),
            key="vista_tabla_tipo"
        )

        todas_asignaturas_ponderables = [col for col in df_ponderaciones_original.columns if col not in ['Grado', 'Rama_de_conocimiento']]
        map_asignaturas_display_tabla = {asig: asig.replace('_', ' ').replace('.', ' ') for asig in todas_asignaturas_ponderables}

        if vista_tabla == "Grados (vista tradicional)":
            st.markdown("Puedes ordenar y buscar en la tabla. Las columnas de asignaturas muestran su ponderación (0.1, 0.15 o 0.2).")
            
            # Filtro por Rama (ya existente)
            rama_seleccionada_tabla = st.selectbox(
                "Filtrar por Rama de Conocimiento (opcional):",
                options=['Todas'] + ramas_conocimiento_disponibles,
                index=0,
                key="tabla_rama_filter_grados" # Changed key to avoid conflict
            )

            df_filtrado_tabla = df_ponderaciones_original.copy()
            if rama_seleccionada_tabla != 'Todas':
                df_filtrado_tabla = df_filtrado_tabla[df_filtrado_tabla['Rama_de_conocimiento'] == rama_seleccionada_tabla]

            # Filtro multiselect para Grados
            grados_disponibles_tabla = sorted(df_filtrado_tabla['Grado'].unique())
            grados_seleccionados_tabla = st.multiselect(
                "Filtrar por Grados Específicos (opcional):",
                options=grados_disponibles_tabla,
                default=[],
                key="tabla_grados_filter"
            )
            if grados_seleccionados_tabla:
                df_filtrado_tabla = df_filtrado_tabla[df_filtrado_tabla['Grado'].isin(grados_seleccionados_tabla)]
            
            asignaturas_seleccionadas_tabla_display = st.multiselect(
                "Seleccionar Asignaturas de 2º Bachillerato a mostrar (columnas, opcional):",
                options=sorted(list(map_asignaturas_display_tabla.values())), 
                default=[], 
                key="tabla_asignaturas_filter_grados" # Changed key
            )
            
            asignaturas_seleccionadas_tabla_cols = [col_original for col_original, col_display in map_asignaturas_display_tabla.items() if col_display in asignaturas_seleccionadas_tabla_display]

            columnas_a_mostrar = ['Grado', 'Rama_de_conocimiento']
            if asignaturas_seleccionadas_tabla_cols:
                columnas_a_mantener.extend(asignaturas_seleccionadas_tabla_cols) # Corrected from columnas_a_mantener
            elif not asignaturas_seleccionadas_tabla_display: 
                columnas_a_mantener.extend([col for col in df_filtrado_tabla.columns if col not in ['Grado', 'Rama_de_conocimiento']]) # Corrected from columnas_a_mantener
            # Ensure this line is properly indented and on its own line
            columnas_finales_tabla = [col for col in columnas_a_mostrar if col in df_filtrado_tabla.columns]
            if 'Grado' not in columnas_finales_tabla and 'Grado' in df_filtrado_tabla.columns: 
                columnas_finales_tabla.insert(0, 'Grado')
            
            from collections import OrderedDict
            columnas_finales_tabla = list(OrderedDict.fromkeys(columnas_finales_tabla))

            if not df_filtrado_tabla.empty and 'Grado' in columnas_finales_tabla:
                st.dataframe(df_filtrado_tabla[columnas_finales_tabla].set_index('Grado'), height=600)
            elif not df_filtrado_tabla.empty:
                 st.dataframe(df_filtrado_tabla[columnas_finales_tabla], height=600) # Fallback if Grado is not indexable
            else:
                st.info("No hay datos para mostrar con los filtros seleccionados.")
        
        elif vista_tabla == "Asignaturas (qué grados ponderan)":
            st.markdown("Selecciona una o varias asignaturas de 2º Bachillerato para ver qué grados las ponderan con 0.2 (y opcionalmente 0.15 y 0.1).")

            asignaturas_para_analisis_display = st.multiselect(
                "Seleccionar Asignaturas de 2º Bachillerato:",
                options=sorted(list(map_asignaturas_display_tabla.values())),
                default=[],
                key="tabla_asignaturas_analisis_filter"
            )
            asignaturas_para_analisis_cols = [col_original for col_original, col_display in map_asignaturas_display_tabla.items() if col_display in asignaturas_para_analisis_display]

            col_015, col_01 = st.columns(2)
            with col_015:
                incluir_015_tabla_asignatura = st.checkbox("Incluir ponderaciones de 0.15", value=False, key="tabla_incluir_015_asignatura")
            with col_01:
                incluir_01_tabla_asignatura = st.checkbox("Incluir ponderaciones de 0.1", value=False, key="tabla_incluir_01_asignatura")

            if asignaturas_para_analisis_cols:
                df_melted = df_ponderaciones_original.melt(
                    id_vars=['Grado', 'Rama_de_conocimiento'],
                    value_vars=asignaturas_para_analisis_cols,
                    var_name='Asignatura_Original', # Store original column name
                    value_name='Ponderacion'
                )
                
                # Map to display names for the 'Asignatura' column after melting
                df_melted['Asignatura'] = df_melted['Asignatura_Original'].map(map_asignaturas_display_tabla)

                ponderaciones_a_buscar = [0.2]
                if incluir_015_tabla_asignatura:
                    ponderaciones_a_buscar.append(0.15)
                if incluir_01_tabla_asignatura:
                    ponderaciones_a_buscar.append(0.1)
                
                df_resultado_asignaturas = df_melted[df_melted['Ponderacion'].isin(ponderaciones_a_buscar)]
                if not df_resultado_asignaturas.empty:
                    df_resultado_asignaturas = df_resultado_asignaturas.sort_values(by=['Asignatura', 'Ponderacion', 'Grado'], ascending=[True, False, True])
                    st.dataframe(
                        df_resultado_asignaturas[['Asignatura', 'Grado', 'Rama_de_conocimiento', 'Ponderacion']],
                        height=600,
                        use_container_width=True
                    )
                else:
                    st.info("No se encontraron grados con las ponderaciones especificadas para las asignaturas seleccionadas.")
            else:
                st.info("Por favor, selecciona al menos una asignatura para analizar.")
    
    elif modo_visualizacion == 'Gráfico Interactivo de Flujo':
        st.subheader("🌊 Gráfico de Flujo Académico Interactivo")
        st.markdown("""
        Selecciona una **Rama de Conocimiento**. El gráfico mostrará las conexiones con ponderación 0.2.
        Puedes activar la opción "Incluir ponderación 0.1" para mostrar todas las ponderaciones.
        """)
        
        col_g1, col_g2, col_g3 = st.columns([2,1,1])

        with col_g1:
            rama_seleccionada_grafo = st.selectbox(
                "Selecciona una Rama de Conocimiento:",
                options=ramas_conocimiento_disponibles, # Ya definidas globalmente
                index=0,
                key="grafo_rama_filter",
                help="El gráfico mostrará los grados pertenecientes a esta rama."
            )
        with col_g2:
            # Mantenemos el checkbox pero no lo usamos activamente
            mostrar_015 = st.checkbox("Incluir ponderación 0.15", value=False, key='grafo_show_015', disabled=True, help="Esta opción ya no se utiliza. Use 'Incluir ponderación 0.1' para mostrar todas las ponderaciones.")
        with col_g3:
            mostrar_01 = st.checkbox("Incluir ponderación 0.1", value=False, key='grafo_show_01', help="Muestra todas las ponderaciones mayores que 0")

        # Filtro para seleccionar una asignatura específica para enfocar el gráfico (opcional)
        # Las opciones para este selector dependerán de la rama seleccionada
        
        df_para_opciones_nodos = df_ponderaciones_original
        if rama_seleccionada_grafo:
             df_para_opciones_nodos = df_ponderaciones_original[df_ponderaciones_original['Rama_de_conocimiento'] == rama_seleccionada_grafo]

        nodos_2_bach_posibles = [col for col in df_para_opciones_nodos.columns if col not in ['Grado', 'Rama_de_conocimiento']]
        nodos_grado_posibles = list(df_para_opciones_nodos['Grado'].unique())
        
        todas_asignaturas_display = {f"{n.replace('_',' ')}": n for n in sorted(nodos_2_bach_posibles)}
        todos_grados_display = {f"{n}": n for n in sorted(nodos_grado_posibles)}
        
        # Permitir al usuario seleccionar una asignatura para filtrar/enfocar
        st.markdown("🔴 Filtrar por <span style='color:red;'>asignatura</span> (opcional, borrar para quitar filtro):", unsafe_allow_html=True)
        asignatura_enfocada_display = st.selectbox(
            "", # Label is now part of st.markdown above
            options=[''] + list(todas_asignaturas_display.keys()),
            index=0,
            key='grafo_asignatura_enfocada',
            help="Selecciona una asignatura de 2º Bachillerato para ver solo sus conexiones directas en rojo."
        )
        asignatura_enfocada_id = todas_asignaturas_display.get(asignatura_enfocada_display)
        
        # Filtro adicional por grado
        st.markdown("🔴 Filtrar por <span style='color:red;'>grado</span> (opcional, borrar para quitar filtro):", unsafe_allow_html=True)
        grado_enfocado_display = st.selectbox(
            "", # Label moved to st.markdown
            options=[''] + list(todos_grados_display.keys()),
            index=0,
            key='grafo_grado_enfocado',
            help="Selecciona un grado universitario para ver solo sus conexiones directas."
        )
        grado_enfocado_id = todos_grados_display.get(grado_enfocado_display)
        
        # Determinar el nodo enfocado (prioridad: asignatura > grado)
        # Modificamos para añadir prefijos apropiados dependiendo del tipo de nodo
        nodo_enfocado_id = None
        if asignatura_enfocada_id:
            # Determinar si es de 1º o 2º Bach
            if asignatura_enfocada_id in RELACIONES_1_A_2:
                nodo_enfocado_id = f"1bach_{asignatura_enfocada_id}"
            else:
                nodo_enfocado_id = f"2bach_{asignatura_enfocada_id}"
        elif grado_enfocado_id:
            nodo_enfocado_id = f"grado_{grado_enfocado_id}"


        if rama_seleccionada_grafo:
            df_filtrado_rama_grafo = df_ponderaciones_original[
                df_ponderaciones_original['Rama_de_conocimiento'] == rama_seleccionada_grafo
            ].copy()

            if df_filtrado_rama_grafo.empty:
                st.warning(f"No se encontraron grados para la rama: '{rama_seleccionada_grafo}'.")
            else:
                # Aplicar filtro de grados específicos si se seleccionaron
                grados_disponibles_grafo = sorted(df_filtrado_rama_grafo['Grado'].unique())
                grados_seleccionados_grafo = st.multiselect(
                    "Filtrar por Grados Específicos en el gráfico (opcional):",                    options=grados_disponibles_grafo,
                    default=[],
                    key="grafo_grados_filter"
                )
                if grados_seleccionados_grafo:
                    df_filtrado_rama_grafo = df_filtrado_rama_grafo[df_filtrado_rama_grafo['Grado'].isin(grados_seleccionados_grafo)]

                if df_filtrado_rama_grafo.empty and grados_seleccionados_grafo:
                    st.warning("Ninguno de los grados específicos seleccionados se encuentra en la rama elegida o no hay datos tras el filtro.")
                elif not df_filtrado_rama_grafo.empty:
                    with st.spinner(f"Generando gráfico interactivo para {rama_seleccionada_grafo}..."):
                        # Pasar el nodo_enfocado_id a la función de generación
                        html_content = generar_diagrama_networkx_pyvis(
                            df_filtrado_rama_grafo, 
                            rama_seleccionada_grafo, 
                            mostrar_ponderacion_015=mostrar_015,
                            mostrar_ponderacion_01=mostrar_01,
                            alto_px=700, # Altura fija
                            selected_node_id=nodo_enfocado_id 
                        )
                        if html_content:
                            st.components.v1.html(html_content, height=720) # Ajustar altura + un poco de padding
                        else:
                            st.info("No hay datos para mostrar en el gráfico con los filtros actuales.")
        else:
            st.info("Por favor, selecciona una Rama de Conocimiento para ver el gráfico.")

    elif modo_visualizacion == 'Calculadora de Nota de Acceso':
        st.subheader("🧮 Calculadora de Nota de Acceso a Grados")
        
        asignaturas_ponderables_cols = [
            col for col in df_ponderaciones_original.columns 
            if col not in ['Grado', 'Rama_de_conocimiento']
        ]
        map_asignaturas_display = {asig: asig.replace('_', ' ').replace('.', ' ') for asig in asignaturas_ponderables_cols}
        
        grado_seleccionado_calc = st.selectbox(
            "Selecciona el Grado Universitario al que quieres acceder:",
            options=[''] + sorted(df_ponderaciones_original['Grado'].unique()),
            index=0,
            format_func=lambda x: 'Selecciona un grado...' if x == '' else x,
            key='grado_seleccionado_calculadora_main_reactive' 
        )

        if grado_seleccionado_calc:
            ponderaciones_grado = df_ponderaciones_original[df_ponderaciones_original['Grado'] == grado_seleccionado_calc].iloc[0]
            asignaturas_que_ponderan_para_grado = {
                asig: ponderaciones_grado[asig] 
                for asig in asignaturas_ponderables_cols 
                if ponderaciones_grado.get(asig, 0) > 0
            }

            col1, col2 = st.columns(2)
            with col1:
                nota_bachillerato = st.number_input("Nota media de Bachillerato (sobre 10):", min_value=0.0, max_value=10.0, value=7.5, step=0.01, format="%.2f", key="calc_nota_bach_reactive")
            with col2:
                nota_fase_general = st.number_input("Nota de la Fase General (EvAU/PEvAU, sobre 10):", min_value=0.0, max_value=10.0, value=7.0, step=0.01, format="%.2f", key="calc_nota_fase_gen_reactive")
            
            st.markdown("---")
            st.markdown("#### Selección de Asignaturas Específicas")
            
            notas_especificas_ingresadas = {} 

            if not asignaturas_que_ponderan_para_grado:
                st.warning("Este grado no tiene asignaturas específicas con ponderación > 0 en nuestros datos.")
            else:
                st.markdown("Selecciona **hasta 2 asignaturas** de la fase específica y sus notas (sobre 10). Solo se considerarán si la nota es >= 5.0.")

                opciones_fase_especifica_map = {
                    f"{map_asignaturas_display[asig]} (Pondera: {ponderacion:.1f})": asig
                    for asig, ponderacion in sorted(asignaturas_que_ponderan_para_grado.items(), key=lambda item: item[0])
                }
                
                asig1_original_name = None
                asig1_display_options = [''] + list(opciones_fase_especifica_map.keys())
                asig1_display_name = st.selectbox(
                    "Asignatura Específica 1 (opcional):", 
                    options=asig1_display_options, 
                    format_func=lambda x: 'Ninguna' if x == '' else x, 
                    key='calc_asig1_sel_reactive'
                )
                
                if asig1_display_name:
                    asig1_original_name = opciones_fase_especifica_map[asig1_display_name]
                    notas_especificas_ingresadas[asig1_original_name] = st.number_input(
                        f"Nota en {map_asignaturas_display[asig1_original_name]}:", 
                        min_value=0.0, max_value=10.0, value=0.0, step=0.1, format="%.1f", 
                        key=f"calc_nota_asig1_{asig1_original_name}_reactive" 
                    )

                asig2_original_name = None
                opciones_para_asig2 = list(opciones_fase_especifica_map.keys())
                if asig1_display_name: 
                    opciones_para_asig2 = [opt for opt in opciones_para_asig2 if opt != asig1_display_name]
                
                asig2_display_options = [''] + opciones_para_asig2
                asig2_display_name = st.selectbox(
                    "Asignatura Específica 2 (opcional):", 
                    options=asig2_display_options, 
                    format_func=lambda x: 'Ninguna' if x == '' else x, 
                    key='calc_asig2_sel_reactive'
                )
                
                if asig2_display_name:
                    asig2_original_name = opciones_fase_especifica_map[asig2_display_name]
                    # Ensure asig2 is different from asig1 if asig1 was selected and has a name
                    if asig1_original_name != asig2_original_name:
                         notas_especificas_ingresadas[asig2_original_name] = st.number_input(
                            f"Nota en {map_asignaturas_display[asig2_original_name]}:", 
                            min_value=0.0, max_value=10.0, value=0.0, step=0.1, format="%.1f", 
                            key=f"calc_nota_asig2_{asig2_original_name}_reactive"
                        )
            
            nota_acceso_base = 0.6 * nota_bachillerato + 0.4 * nota_fase_general
            
            contribuciones_potenciales = []
            for asig_original, nota_ingresada in notas_especificas_ingresadas.items():
                if nota_ingresada >= 5.0:
                    ponderacion_materia = ponderaciones_grado.get(asig_original, 0) 
                    if ponderacion_materia > 0:
                        contribucion = ponderacion_materia * nota_ingresada
                        contribuciones_potenciales.append({
                            "name": asig_original,
                            "grade": nota_ingresada,
                            "ponderacion": ponderacion_materia,
                            "contribution": contribucion
                        })
            
            contribuciones_finales_seleccionadas = sorted(contribuciones_potenciales, key=lambda x: x["contribution"], reverse=True)[:2]
            
            suma_ponderaciones_especificas = sum(c["contribution"] for c in contribuciones_finales_seleccionadas)
            
            nota_final_acceso = nota_acceso_base + suma_ponderaciones_especificas
            
            st.markdown("---")
            st.subheader(f"📈 Tu Nota de Admisión Estimada para {grado_seleccionado_calc}:")
            st.metric(label="Nota Final (sobre 14)", value=f"{nota_final_acceso:.3f}")

            desglose_md = f"""
            **Desglose:**
            - Componente Bachillerato (60%): `{nota_bachillerato * 0.6:.3f}`
            - Componente Fase General (40%): `{nota_fase_general * 0.4:.3f}`
            - **Subtotal Nota Base (sobre 10): `{nota_acceso_base:.3f}`**
            - Suma de ponderaciones de asignaturas específicas (sobre 4): `{suma_ponderaciones_especificas:.3f}`
            """
            st.markdown(desglose_md)

            if contribuciones_finales_seleccionadas:
                st.markdown("**Detalle de asignaturas específicas consideradas (nota >= 5.0, se eligen las dos que más aporten):**")
                for detalle in contribuciones_finales_seleccionadas:
                    st.markdown(f"- {map_asignaturas_display[detalle['name']]}: Nota `{detalle['grade']:.1f}`, Ponderación `{detalle['ponderacion']:.1f}`, Aporta: `{detalle['contribution']:.3f}`")
            else:
                st.info("No se han añadido asignaturas específicas válidas (nota >= 5.0 y ponderación > 0) a la nota de admisión, o las notas son < 5.0.")
            
            st.caption("Recuerda: solo las asignaturas específicas con nota >= 5.0 contribuyen a la fase específica. Se eligen las dos que más aporten.")

else: # if df_ponderaciones_original is None
    st.error("Error Crítico: No se pudieron cargar los datos de ponderaciones. Verifica que el archivo 'ponderaciones_andalucia.csv' existe y está en el formato correcto.")

st.sidebar.markdown("---")
st.sidebar.info("Aplicación desarrollada para la visualización de ponderaciones de selectividad en Andalucía. Los datos de ponderaciones son oficiales pero te recomendamos verificarlos en las fuentes oficiales de la universidad a la que quieras acceder.")
st.sidebar.markdown("---")
st.sidebar.markdown("Autora: Rosa María Santos Vilches")
st.sidebar.markdown("IES Politécnico Sevilla")

