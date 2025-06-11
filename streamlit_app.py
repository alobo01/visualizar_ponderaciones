import streamlit as st
import pandas as pd
from graphviz import Digraph
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math

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

def generar_diagrama_streamlit(df_data, rama_filter_display_name):
    dot = Digraph(comment=f'Flujo Académico para {rama_filter_display_name}')
    dot.attr('graph', rankdir='LR', splines='curved', overlap='false', bgcolor='transparent', compound='true', concentrate='true')
    dot.attr('graph', label=f'Ruta Académica para: {rama_filter_display_name}', labelloc='t', fontsize='20', fontname="Arial")
    dot.attr('node', fontname="Arial", fontsize="10")
    dot.attr('edge', fontname="Arial", fontsize="8")


    relaciones_1_a_2 = {
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
    asignaturas_1_bach = list(relaciones_1_a_2.keys())
    
    # Columnas de ponderación (asignaturas de 2º Bach)
    columnas_ponderacion = [col for col in df_data.columns if col not in ['Grado', 'Rama_de_conocimiento']]
    
    # Asignaturas de 2º Bach que realmente tienen alguna ponderación > 0 en el df_filtrado
    # o todas si ninguna tiene ponderación (para asegurar que se muestran)
    asignaturas_2_activas = [col for col in columnas_ponderacion if df_data[col].sum() > 0]
    if not asignaturas_2_activas: # Si ninguna tiene sum > 0 en el subset, mostrar todas las posibles de 2º
        asignaturas_2_activas = columnas_ponderacion
    
    # Filtrar relaciones_1_a_2 para que solo los sucesores en asignaturas_2_activas se consideren
    # y también para que solo los precursores que tienen algún sucesor activo se muestren.
    asignaturas_1_bach_filtradas = []
    for precursor, sucesores in relaciones_1_a_2.items():
        if any(sucesor in asignaturas_2_activas for sucesor in sucesores):
            asignaturas_1_bach_filtradas.append(precursor)
    
    # Colores para asignaturas de 2º Bach
    if asignaturas_2_activas:
        cmap = plt.cm.get_cmap('tab20', len(asignaturas_2_activas))
        color_asignatura = {asig: mcolors.to_hex(cmap(i)) for i, asig in enumerate(asignaturas_2_activas)}
    else: # Caso sin asignaturas activas
        color_asignatura = {}


    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='1º Bachillerato', style='filled', color='#F0F8FF') # AliceBlue
        c.attr('node', shape='box', style='filled,rounded', color='#E6E6FA', fontname="Arial", fontsize="10") # Lavender
        for nodo in asignaturas_1_bach_filtradas:
            c.node(nodo, nodo.replace('_', '\n').replace('.', '\n'), tooltip=nodo.replace('_', ' '))

    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='2º Bachillerato (Asignaturas que ponderan)', style='filled', color='#FAF0E6') # Linen
        c.attr('node', shape='box', style='filled,rounded', fontname="Arial", fontsize="10")
        for nodo in asignaturas_2_activas:
            color = color_asignatura.get(nodo, '#D3D3D3') # LightGray por defecto
            c.node(nodo, nodo.replace('_', '\n').replace('.', '\n'), color=color + 'BF', tooltip=nodo.replace('_', ' ')) # BF para ~75% opacidad

    grados_unicos = df_data['Grado'].unique()
    with dot.subgraph(name='cluster_3') as c:
        c.attr(label='Grados Universitarios', style='filled', color='#FFF0F5') # LavenderBlush
        c.attr('node', shape='box', style='filled,rounded', color='#FFDAB9', fontname="Arial", fontsize="9") # PeachPuff
        for grado in grados_unicos:
            label_grado = grado.replace(' + ', '+\n').replace(' y ', ' y\n').replace(' de ', ' de\n').replace('/', '/\n')
            if len(grado) > 45: # Acortar nombres muy largos para el nodo
                 # Trata de cortar en un espacio si es posible cerca del límite
                corte_idx = label_grado.rfind(' ', 0, 40)
                if corte_idx != -1:
                    label_grado = label_grado[:corte_idx] + "\n" + label_grado[corte_idx+1:]
                else: # Si no hay espacio, corta bruscamente
                    label_grado = label_grado[:40] + "..."

            dot.node(grado, label_grado, tooltip=grado) # Tooltip siempre con nombre completo

    for precursor, sucesores in relaciones_1_a_2.items():
        if precursor in asignaturas_1_bach_filtradas:
            for sucesor in sucesores:
                if sucesor in asignaturas_2_activas:
                    dot.edge(precursor, sucesor, penwidth='1.5', color='#6A5ACD') # SlateBlue

    for asignatura_2 in asignaturas_2_activas:
        color_base = color_asignatura.get(asignatura_2, '#808080') # Gris si no tiene color específico
        for _, row in df_data.iterrows():
            grado = row['Grado']
            ponderacion = row.get(asignatura_2, 0.0)
            
            label_edge = f"{ponderacion:.1f}"
            tooltip_edge = f"{asignatura_2.replace('_', ' ')} → {grado}\nPonderación: {ponderacion:.1f}"

            if ponderacion >= 0.19: # Ponderación 0.2
                penwidth = '2.0'
                style = 'solid'
                edge_color = color_base
            elif ponderacion >= 0.09: # Ponderación 0.1
                penwidth = '1.2'
                style = 'dashed'
                edge_color = mcolors.to_hex(mcolors.rgb_to_hsv(mcolors.to_rgb(color_base)) * [1, 0.7, 0.9]) # Un poco más claro/desaturado
            elif ponderacion > 0.0: # Ponderación > 0 pero < 0.1 (raro, pero por si acaso)
                penwidth = '0.7'
                style = 'dotted'
                edge_color = mcolors.to_hex(mcolors.rgb_to_hsv(mcolors.to_rgb(color_base)) * [1, 0.5, 0.8]) # Aún más claro
            else: # Ponderación 0.0
                penwidth = '0.5'
                style = 'dotted'
                edge_color = '#D3D3D3' # LightGray para 0.0
                label_edge = "" # No mostrar etiqueta para 0.0 para no saturar

            if ponderacion > 0.0 or (ponderacion == 0.0 and st.session_state.get('show_zero_ponderations', True)): # Mostrar 0.0 si está activado
                 dot.edge(asignatura_2, grado, label=label_edge, color=edge_color, penwidth=penwidth, style=style, tooltip=tooltip_edge)
            
    return dot

# --- Configuración de la página de Streamlit ---
st.set_page_config(page_title="Visor Ponderaciones Selectividad Andalucía", layout="wide", initial_sidebar_state="expanded")

st.title("📊 Visor Interactivo de Ponderaciones para Selectividad (Andalucía)")
st.markdown("""
Bienvenido al visor de ponderaciones. Explora cómo las asignaturas de bachillerato
ponderan para los grados universitarios en Andalucía.
""")

# --- Carga de datos ---
DATA_FILE = 'ponderaciones_andalucia.csv' # Asegúrate que este archivo está en el mismo directorio
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
        
        rama_seleccionada_tabla = st.selectbox(
            "Filtrar por Rama de Conocimiento (opcional):",
            options=['Todas'] + ramas_conocimiento_disponibles,
            index=0,
            key="tabla_rama_filter"
        )
        
        df_mostrado = df_ponderaciones_original.copy()
        if rama_seleccionada_tabla != 'Todas':
            df_mostrado = df_mostrado[df_mostrado['Rama_de_conocimiento'] == rama_seleccionada_tabla]

        st.dataframe(df_mostrado.set_index('Grado'), height=700) # Grado como índice para mejor visualización

    elif modo_visualizacion == 'Gráfico Interactivo de Flujo':
        st.subheader("🌊 Gráfico de Flujo Académico")
        st.markdown("""
        Selecciona una **Rama de Conocimiento** para generar el gráfico.
        Las líneas indican posibles trayectorias; el número sobre ellas es la ponderación.
        """)
        
        rama_seleccionada_grafo = st.selectbox(
            "Selecciona una Rama de Conocimiento:",
            options=ramas_conocimiento_disponibles,
            index=0,
            help="El gráfico mostrará los grados pertenecientes a esta rama."
        )
        
        st.checkbox("Mostrar conexiones con ponderación 0.0 (líneas punteadas grises)", value=True, key='show_zero_ponderations',
                        help="Desmarcar para un gráfico menos denso, mostrando solo ponderaciones > 0.")


        if rama_seleccionada_grafo:
            df_filtrado_rama = df_ponderaciones_original[
                df_ponderaciones_original['Rama_de_conocimiento'] == rama_seleccionada_grafo
            ].copy()

            if df_filtrado_rama.empty:
                st.warning(f"No se encontraron grados para la rama: '{rama_seleccionada_grafo}'.")
            else:
                with st.spinner(f"Generando gráfico para {rama_seleccionada_grafo}... Esto puede tardar un momento."):
                    diagrama = generar_diagrama_streamlit(df_filtrado_rama, rama_seleccionada_grafo)
                    st.graphviz_chart(diagrama, use_container_width=True)
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

