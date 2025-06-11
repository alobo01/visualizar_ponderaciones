import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network as PyvisNetwork
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math

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
        df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo '{filepath}'. Asegúrate de que el archivo 'ponderaciones_andalucia.csv' está en el mismo directorio que la aplicación.")
        return None
    except UnicodeDecodeError:
        st.warning("Error de decodificación UTF-8. Probando con 'latin1' como alternativa.")
        try:
            df = pd.read_csv(filepath, encoding='latin1', on_bad_lines='skip')
        except Exception as e:
            st.error(f"Error al cargar con Latin1: {e}")
            return None

    if df.empty:
        st.error("El archivo CSV está vacío o no se pudo cargar correctamente.")
        return None

    df.rename(columns={df.columns[0]: 'Grado'}, inplace=True)
    
    # Comprobar si 'Rama de conocimiento' existe antes de dropear NA
    if 'Rama de conocimiento' in df.columns:
        df = df.dropna(subset=['Rama de conocimiento'])
    else:
        st.error("La columna 'Rama de conocimiento' no se encuentra en el CSV. Verifica el formato del archivo.")
        return None
        
    df = df[~df['Grado'].str.contains("Pondera|No pondera|This work", case=False, na=False)]
    
    nuevas_columnas = {col: col.strip().replace(' ', '_').replace('-', '_') for col in df.columns}
    df.rename(columns=nuevas_columnas, inplace=True)
    
    # Asumiendo que las ponderaciones empiezan desde la tercera columna ('Grado', 'Rama_de_conocimiento', luego asignaturas)
    columnas_asignaturas = [col for col in df.columns if col not in ['Grado', 'Rama_de_conocimiento']]

    for col in columnas_asignaturas:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.fillna(0.0, inplace=True)
    return df

def generar_diagrama_networkx_pyvis(df_data, rama_filter_display_name, mostrar_ponderacion_015=False, alto_px=800, ancho_px=1000, selected_node_id=None):
    """
    Genera un diagrama interactivo usando NetworkX para la lógica y Pyvis para la visualización.
    Permite filtrar por un nodo seleccionado.
    """
    
    # 1. Definir Relaciones 1º -> 2º Bach (Ahora se usa la global RELACIONES_1_A_2)
    # relaciones_1_a_2 = { ... } # Esta definición local se elimina

    # Filtrar df_data si se ha seleccionado un nodo
    # Esta es una simplificación. Una implementación completa podría requerir identificar si el nodo es
    # de 1º Bach, 2º Bach o Grado y filtrar las conexiones relevantes.
    if selected_node_id:
        # Si el nodo seleccionado es una asignatura de 1º Bach
        if selected_node_id in RELACIONES_1_A_2: # Usar la variable global
            # Mantener solo las asignaturas de 2º Bach relacionadas y los grados a los que estas conectan
            sucesores_directos = RELACIONES_1_A_2[selected_node_id] # Usar la variable global
            df_data = df_data[df_data.apply(lambda row: any(row[s] > 0 for s in sucesores_directos if s in row), axis=1)]
            # Y también filtrar las columnas de asignaturas de 2º Bach a solo las sucesoras
            cols_a_mantener = ['Grado', 'Rama_de_conocimiento'] + [s for s in sucesores_directos if s in df_data.columns]
            df_data = df_data[cols_a_mantener]

        # Si el nodo seleccionado es una asignatura de 2º Bach
        elif selected_node_id in df_data.columns and selected_node_id not in ['Grado', 'Rama_de_conocimiento']:
            df_data = df_data[df_data[selected_node_id] > 0] # Mantener solo grados donde esta asignatura pondera
            # Mantener solo esta asignatura de 2º Bach y los grados
            cols_a_mantener = ['Grado', 'Rama_de_conocimiento', selected_node_id]
            df_data = df_data[cols_a_mantener]
        
        # Si el nodo seleccionado es un Grado
        elif selected_node_id in df_data['Grado'].unique():
            df_data = df_data[df_data['Grado'] == selected_node_id] # Mantener solo este grado
        
        if df_data.empty:
            st.warning(f"No se encontraron datos relevantes para el nodo seleccionado: {selected_node_id}")
            # Podríamos devolver un grafo vacío o un mensaje
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


    # Añadir nodos con atributos para Pyvis (capa, color, tamaño, título)
    # Capa 1: 1º Bachillerato
    for nodo_1 in asignaturas_1_bach_filtradas:
        G.add_node(nodo_1, layer=1, color='#E6E6FA', title=nodo_1.replace('_', ' '), shape='box', type='1_bach')

    # Capa 2: 2º Bachillerato
    if asignaturas_2_activas:
        cmap = plt.cm.get_cmap('tab20', len(asignaturas_2_activas))
        color_map_2_bach = {asig: mcolors.to_hex(cmap(i)) for i, asig in enumerate(asignaturas_2_activas)}
    else:
        color_map_2_bach = {}

    for nodo_2 in asignaturas_2_activas:
        color = color_map_2_bach.get(nodo_2, '#D3D3D3')
        G.add_node(nodo_2, layer=2, color=color + 'BF', title=nodo_2.replace('_', ' '), shape='box', type='2_bach')

    # Capa 3: Grados Universitarios
    grados_en_df = df_data['Grado'].unique()
    for grado_uni in grados_en_df:
        G.add_node(grado_uni, layer=3, color='#FFDAB9', title=grado_uni, shape='box', type='grado')

    # Conexiones 1º Bach -> 2º Bach
    for precursor, sucesores in RELACIONES_1_A_2.items(): # Usar la variable global
        if precursor in G: # Si el nodo de 1º Bach está en el grafo
            for sucesor in sucesores:
                if sucesor in G: # Si el nodo de 2º Bach está en el grafo
                    G.add_edge(precursor, sucesor, color='#6A5ACD', weight=2) # weight para pyvis

    # Conexiones 2º Bach -> Grados
    min_ponderacion_mostrar = 0.19 # Ponderación 0.2
    if mostrar_ponderacion_015:
        min_ponderacion_mostrar = 0.14 # Ponderación 0.15

    for asignatura_2 in asignaturas_2_activas:
        if asignatura_2 not in G: continue # Si la asignatura no está en el grafo (p.ej. por filtrado de nodo)
        color_base_edge = color_map_2_bach.get(asignatura_2, '#808080')
        for _, row in df_data.iterrows():
            grado = row['Grado']
            if grado not in G: continue # Si el grado no está en el grafo
            
            ponderacion = row.get(asignatura_2, 0.0)
            
            if ponderacion >= min_ponderacion_mostrar:
                edge_color = color_base_edge
                edge_width = 2.5 if ponderacion >= 0.19 else 1.5
                edge_title = f"{ponderacion:.2f}"
                G.add_edge(asignatura_2, grado, color=edge_color, weight=edge_width, title=edge_title, label=f"{ponderacion:.2f}")
            elif ponderacion > 0 and st.session_state.get('show_zero_ponderations', False): # Para mostrar 0.0 o 0.1 si el checkbox general está activo
                 # Esta lógica necesitaría ajustarse si 'show_zero_ponderations' debe aplicar aquí
                 pass # Por ahora, nos ceñimos a 0.2 y 0.15 opcional


    # --- Visualización con Pyvis ---
    # Crear un objeto Pyvis Network
    # El height y width se pueden ajustar según necesidad.
    # `cdn_resources='remote'` es bueno para asegurar que funcione en diferentes entornos.
    # `select_menu=True` y `filter_menu=True` añaden controles de Pyvis, pero pueden ser redundantes con los de Streamlit.
    
    # Generar un nombre de archivo único para evitar conflictos si la función se llama rápidamente o en paralelo
    # import uuid
    # html_file_name = f"pyvis_graph_{uuid.uuid4().hex}.html"
    
    # Para Streamlit, es mejor generar el HTML y pasarlo como string si es posible,
    # o guardarlo en un directorio temporal accesible por Streamlit.
    # st.components.v1.html necesita una ruta de archivo o una cadena HTML.

    if not G.nodes():
        # st.warning("No hay datos para mostrar en el gráfico con los filtros actuales.")
        return None # Devuelve None si no hay nodos para evitar errores en Pyvis

    nt = PyvisNetwork(height=f"{alto_px}px", width="100%", notebook=False, directed=True, cdn_resources='remote')
    nt.from_nx(G)

    # Configuración de la física para que los nodos sean arrastrables
    # nt.show_buttons(filter_=['physics']) # Muestra botones para controlar la física
    # Valid JSON options string
    options_json = """
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -3000,
          "centralGravity": 0.1,
          "springLength": 150,
          "springConstant": 0.05,
          "damping": 0.09,
          "avoidOverlap": 0.1
        },
        "minVelocity": 0.75,
        "solver": "barnesHut"
      },
      "layout": {
        "hierarchical": {
          "enabled": false
        }
      },
      "interaction":{
        "dragNodes": true,
        "dragView": true,
        "zoomView": true,
        "tooltipDelay": 200,
        "hideEdgesOnDrag": false,
        "hideNodesOnDrag": false
      },
      "nodes": {
          "font": {
              "size": 10
           }
      },
      "edges": {
          "font": {
              "size": 8,
              "align": "top"
          },
          "arrows": {
              "to": {"enabled": true, "scaleFactor": 0.7}
          },
          "smooth": {
              "type": "cubicBezier",
              "forceDirection": "horizontal",
              "roundness": 0.7
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
df_ponderaciones_original = cargar_y_limpiar_csv(DATA_FILE)

if df_ponderaciones_original is not None:
    st.sidebar.header("🛠️ Opciones de Visualización")
    
    # Nombres de ramas directamente del CSV (ya son descriptivos)
    ramas_conocimiento_disponibles = sorted(df_ponderaciones_original['Rama_de_conocimiento'].unique())

    modo_visualizacion = st.sidebar.radio(
        "Selecciona el modo de visualización:",
        ('Gráfico Interactivo de Flujo', 'Tabla de Ponderaciones', 'Calculadora de Nota de Acceso'),
        key='modo_viz'
    )

    if modo_visualizacion == 'Tabla de Ponderaciones':
        st.subheader("📜 Tabla Completa de Ponderaciones")
        st.markdown("Puedes ordenar y buscar en la tabla. Las columnas de asignaturas muestran su ponderación (0.1, 0.15 o 0.2).")
        
        # Filtro por Rama (ya existente)
        rama_seleccionada_tabla = st.selectbox(
            "Filtrar por Rama de Conocimiento (opcional):",
            options=['Todas'] + ramas_conocimiento_disponibles,
            index=0,
            key="tabla_rama_filter"
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

        # Filtro multiselect para Asignaturas de Bachillerato (columnas de ponderación)
        # Tomar todas las asignaturas del df original para que la lista sea completa
        todas_asignaturas_ponderables = [col for col in df_ponderaciones_original.columns if col not in ['Grado', 'Rama_de_conocimiento']]
        map_asignaturas_display_tabla = {asig: asig.replace('_', ' ').replace('.', ' ') for asig in todas_asignaturas_ponderables}
        
        asignaturas_seleccionadas_tabla_display = st.multiselect(
            "Seleccionar Asignaturas de 2º Bachillerato a mostrar (columnas, opcional):",
            options=sorted(list(map_asignaturas_display_tabla.values())), # Mostrar nombres "limpios"
            default=[], # Por defecto ninguna, o podrías poner todas
            key="tabla_asignaturas_filter"
        )
        
        # Convertir nombres de display seleccionados de nuevo a nombres de columna originales
        asignaturas_seleccionadas_tabla_cols = [col_original for col_original, col_display in map_asignaturas_display_tabla.items() if col_display in asignaturas_seleccionadas_tabla_display]

        columnas_a_mostrar = ['Grado', 'Rama_de_conocimiento']
        if asignaturas_seleccionadas_tabla_cols:
            columnas_a_mostrar.extend(asignaturas_seleccionadas_tabla_cols)
        elif not asignaturas_seleccionadas_tabla_display: # Si no se selecciona ninguna, mostrar todas las disponibles en el df_filtrado_tabla
            columnas_a_mostrar.extend([col for col in df_filtrado_tabla.columns if col not in ['Grado', 'Rama_de_conocimiento']])


        # Asegurarse de que las columnas seleccionadas existen en el df_filtrado_tabla antes de intentar mostrarlas
        columnas_finales_tabla = [col for col in columnas_a_mostrar if col in df_filtrado_tabla.columns]
        if 'Grado' not in columnas_finales_tabla: # Asegurar que Grado siempre esté si existe
            columnas_finales_tabla.insert(0, 'Grado')
        
        # Eliminar duplicados manteniendo el orden
        from collections import OrderedDict
        columnas_finales_tabla = list(OrderedDict.fromkeys(columnas_finales_tabla))


        if not df_filtrado_tabla.empty:
            st.dataframe(df_filtrado_tabla[columnas_finales_tabla].set_index('Grado'), height=600)
        else:
            st.info("No hay datos para mostrar con los filtros seleccionados.")


    elif modo_visualizacion == 'Gráfico Interactivo de Flujo':
        st.subheader("🌊 Gráfico de Flujo Académico Interactivo")
        st.markdown("""
        Selecciona una **Rama de Conocimiento**. El gráfico mostrará las conexiones con ponderación 0.2.
        Puedes optar por incluir también las de 0.15. Haz clic y arrastra los nodos para reorganizar.
        """)
        
        col_g1, col_g2 = st.columns([3,1])

        with col_g1:
            rama_seleccionada_grafo = st.selectbox(
                "Selecciona una Rama de Conocimiento:",
                options=ramas_conocimiento_disponibles, # Ya definidas globalmente
                index=0,
                key="grafo_rama_filter",
                help="El gráfico mostrará los grados pertenecientes a esta rama."
            )
        with col_g2:
            mostrar_015 = st.checkbox("Incluir ponderación 0.15", value=False, key='grafo_show_015')

        # Filtro para seleccionar un nodo específico para enfocar el gráfico (opcional)
        # Las opciones para este selector dependerán de la rama seleccionada
        
        df_para_opciones_nodos = df_ponderaciones_original
        if rama_seleccionada_grafo:
             df_para_opciones_nodos = df_ponderaciones_original[df_ponderaciones_original['Rama_de_conocimiento'] == rama_seleccionada_grafo]

        nodos_1_bach_posibles = list(RELACIONES_1_A_2.keys()) # Usar la variable global
        nodos_2_bach_posibles = [col for col in df_para_opciones_nodos.columns if col not in ['Grado', 'Rama_de_conocimiento']]
        nodos_grado_posibles = list(df_para_opciones_nodos['Grado'].unique())
        
        todos_nodos_posibles_display = {
            **{f"1º Bach: {n.replace('_',' ')}": n for n in sorted(nodos_1_bach_posibles)},
            **{f"2º Bach: {n.replace('_',' ')}": n for n in sorted(nodos_2_bach_posibles)},
            **{f"Grado: {n}": n for n in sorted(nodos_grado_posibles)}
        }
        
        # Permitir al usuario seleccionar un nodo para filtrar/enfocar
        # Esto es un intento de "simular" el clic en el nodo. El usuario selecciona del dropdown.
        nodo_enfocado_display = st.selectbox(
            "Enfocar en un nodo específico (opcional, borrar para quitar filtro):",
            options=[''] + list(todos_nodos_posibles_display.keys()),
            index=0,
            key='grafo_nodo_enfocado',
            help="Selecciona una asignatura o grado para ver solo sus conexiones directas."
        )
        nodo_enfocado_id = todos_nodos_posibles_display.get(nodo_enfocado_display)


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
                    "Filtrar por Grados Específicos en el gráfico (opcional):",
                    options=grados_disponibles_grafo,
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
        
        with st.form(key='calculadora_form'):
            col1, col2 = st.columns(2)
            with col1:
                nota_bachillerato = st.number_input("Nota media de Bachillerato (sobre 10):", min_value=0.0, max_value=10.0, value=7.5, step=0.01, format="%.2f")
            with col2:
                nota_fase_general = st.number_input("Nota de la Fase General (EvAU/PEvAU, sobre 10):", min_value=0.0, max_value=10.0, value=7.0, step=0.01, format="%.2f")
            
            st.markdown("---")
            st.markdown("#### Selección de Grado y Asignaturas Específicas")

            grado_seleccionado_calc = st.selectbox(
                "Selecciona el Grado Universitario al que quieres acceder:",
                options=[''] + sorted(df_ponderaciones_original['Grado'].unique()),
                index=0,
                format_func=lambda x: 'Selecciona un grado...' if x == '' else x
            )

            if grado_seleccionado_calc:
                ponderaciones_grado = df_ponderaciones_original[df_ponderaciones_original['Grado'] == grado_seleccionado_calc].iloc[0]
                
                asignaturas_que_ponderan_para_grado = {
                    asig: ponderaciones_grado[asig] 
                    for asig in asignaturas_ponderables_cols 
                    if ponderaciones_grado.get(asig, 0) > 0 # Solo las que ponderan > 0
                }
                
                if not asignaturas_que_ponderan_para_grado:
                    st.warning("Este grado no tiene asignaturas específicas con ponderación > 0 en nuestros datos.")
                else:
                    st.markdown("Selecciona **hasta 2 asignaturas** de la fase específica y sus notas (sobre 10). Solo se considerarán si la nota es >= 5.0.")

                    opciones_fase_especifica = {
                        f"{map_asignaturas_display[asig]} (Pondera: {ponderacion:.1f})": asig # Display: internal_name
                        for asig, ponderacion in sorted(asignaturas_que_ponderan_para_grado.items(), key=lambda item: item[0]) # Ordenar alfabéticamente
                    }
                    
                    notas_asignaturas_elegidas = {}
                    
                    # Selectores para las dos asignaturas
                    asig1_display = st.selectbox("Asignatura Específica 1 (opcional):", options=[''] + list(opciones_fase_especifica.keys()), format_func=lambda x: 'Ninguna' if x == '' else x, key='asig1_sel')
                    if asig1_display:
                        asig1_original = opciones_fase_especifica[asig1_display]
                        notas_asignaturas_elegidas[asig1_original] = st.number_input(f"Nota en {map_asignaturas_display[asig1_original]}:", min_value=0.0, max_value=10.0, value=5.0, step=0.1, format="%.1f", key=f"nota_{asig1_original}")

                    # Filtrar opciones para la segunda asignatura para no repetir la primera
                    opciones_asig2 = {k:v for k,v in opciones_fase_especifica.items() if k != asig1_display} if asig1_display else opciones_fase_especifica
                    
                    asig2_display = st.selectbox("Asignatura Específica 2 (opcional):", options=[''] + list(opciones_asig2.keys()), format_func=lambda x: 'Ninguna' if x == '' else x, key='asig2_sel')
                    if asig2_display:
                        asig2_original = opciones_fase_especifica[asig2_display] # Usar opciones_fase_especifica para el mapeo original
                        notas_asignaturas_elegidas[asig2_original] = st.number_input(f"Nota en {map_asignaturas_display[asig2_original]}:", min_value=0.0, max_value=10.0, value=5.0, step=0.1, format="%.1f", key=f"nota_{asig2_original}")
            
            submitted = st.form_submit_button("Calcular Nota de Acceso")

            if submitted and grado_seleccionado_calc:
                nota_acceso_base = 0.6 * nota_bachillerato + 0.4 * nota_fase_general
                suma_ponderaciones_especificas = 0.0
                contribuciones_especificas = []

                for asig_original, nota_materia in notas_asignaturas_elegidas.items():
                    ponderacion_materia = ponderaciones_grado.get(asig_original, 0)
                    if ponderacion_materia > 0 and nota_materia >= 5.0:
                        contribucion = ponderacion_materia * nota_materia
                        suma_ponderaciones_especificas += contribucion
                        contribuciones_especificas.append(f"- {map_asignaturas_display[asig_original]}: Nota `{nota_materia:.1f}`, Ponderación `{ponderacion_materia:.1f}`, Aporta: `{contribucion:.3f}`")
                    elif ponderacion_materia > 0 and nota_materia < 5.0:
                         contribuciones_especificas.append(f"- {map_asignaturas_display[asig_original]}: Nota `{nota_materia:.1f}` (No pondera por ser < 5.0)")
                
                nota_final_acceso = nota_acceso_base + suma_ponderaciones_especificas
                
                st.markdown("---")
                st.subheader(f"📈 Tu Nota de Acceso Estimada para {grado_seleccionado_calc}:")
                st.metric(label="Nota Final (sobre 14)", value=f"{nota_final_acceso:.3f}")

                st.markdown(f"""
                **Desglose:**
                - Componente Bachillerato (60%): `{(0.6 * nota_bachillerato):.3f}`
                - Componente Fase General (40%): `{(0.4 * nota_fase_general):.3f}`
                - **Subtotal Nota Base (sobre 10): `{nota_acceso_base:.3f}`**
                - Suma de ponderaciones de asignaturas específicas (sobre 4): `{suma_ponderaciones_especificas:.3f}`
                """)
                if contribuciones_especificas:
                    st.markdown("**Detalle de asignaturas específicas consideradas:**")
                    for detalle in contribuciones_especificas:
                        st.markdown(detalle)
                elif notas_asignaturas_elegidas:
                     st.info("Ninguna de las asignaturas específicas seleccionadas y aprobadas pondera para este grado o no alcanzaron el 5.0.")
                else:
                    st.info("No se seleccionaron asignaturas específicas.")
            elif submitted and not grado_seleccionado_calc:
                st.error("Por favor, selecciona un Grado Universitario para realizar el cálculo.")


else:
    st.error("Error Crítico: No se pudieron cargar los datos de ponderaciones. Verifica que el archivo 'ponderaciones_andalucia.csv' existe y está en el formato correcto.")

st.sidebar.markdown("---")
st.sidebar.info("Aplicación desarrollada para la visualización de ponderaciones de selectividad en Andalucía. Los datos de ponderaciones son cruciales y deben ser verificados con fuentes oficiales.")
st.sidebar.markdown("---")
st.sidebar.markdown("Autora: Rosa María Santos Vilches")
st.sidebar.markdown("IES Politécnico Sevilla")

