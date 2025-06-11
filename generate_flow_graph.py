import pandas as pd
from graphviz import Digraph
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math

# --- FUNCIÓN PARA CARGAR Y LIMPIAR EL CSV (con UTF-8) ---
def cargar_y_limpiar_csv(filepath):
    try:
        # Usar codificación UTF-8 como se solicitó
        df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{filepath}'.")
        return None
    except UnicodeDecodeError:
        print("Error de decodificación. Probando con 'latin1' como alternativa.")
        df = pd.read_csv(filepath, encoding='latin1', on_bad_lines='skip')


    df.rename(columns={df.columns[0]: 'Grado'}, inplace=True)
    df = df.dropna(subset=['Rama de conocimiento'])
    df = df[~df['Grado'].str.contains("Pondera|No pondera|This work", case=False, na=False)]
    
    nuevas_columnas = {col: col.strip().replace(' ', '_').replace('-', '_') for col in df.columns}
    df.rename(columns=nuevas_columnas, inplace=True)
    
    for col in df.columns[2:]:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.fillna(0.0, inplace=True)
    
    return df

# --- FUNCIÓN PARA CREAR DIAGRAMAS FILTRADOS POR RAMA ---
def crear_diagrama_filtrado(df, rama_filter):
    """
    Crea un diagrama de flujo de 3 capas (1º Bach -> 2º Bach -> Grados)
    filtrado por una rama de conocimiento.
    """
    df_filtrado = df[df['Rama_de_conocimiento'].str.startswith(rama_filter, na=False)].copy()
    if df_filtrado.empty:
        print(f"No se encontraron datos para la rama: '{rama_filter}'")
        return
        
    # Llamar a la función de dibujo con los datos filtrados
    _dibujar_diagrama(df_filtrado, f'ruta_academica_{rama_filter.lower()}')


# --- FUNCIÓN PARA CREAR EL DIAGRAMA GLOBAL ---
def crear_diagrama_global(df):
    """
    Crea un diagrama de flujo global con todas las asignaturas y grados.
    ADVERTENCIA: El resultado será un archivo grande y complejo.
    """
    print("\nADVERTENCIA: Generando el diagrama global. Esto puede tardar y el archivo resultante será muy grande y denso.")
    # Para hacerlo manejable, solo mostraremos las conexiones más fuertes (ponderación 0.2)
    # y limitaremos a un máximo de grados por asignatura para no saturar.
    _dibujar_diagrama(df, 'ruta_academica_global', global_mode=True, max_grados_por_asignatura=10)


# --- FUNCIÓN INTERNA DE DIBUJO (lógica compartida) ---
def _dibujar_diagrama(df_data, output_filename, global_mode=False, max_grados_por_asignatura=None):
    
    # 1. Definir Relaciones 1º -> 2º Bach
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

    # 2. Asignar colores únicos a las asignaturas de 2º Bach
    asignaturas_2_utiles = [col for col in df_data.columns[2:-1] if df_data[col].sum() > 0]
    cmap = plt.cm.get_cmap('tab20', len(asignaturas_2_utiles))
    color_asignatura = {asig: mcolors.to_hex(cmap(i)) for i, asig in enumerate(asignaturas_2_utiles)}
    
    # 3. Inicializar el Gráfico
    dot = Digraph(comment='Flujo Académico')
    dot.attr('graph', rankdir='LR', splines='curved', overlap='false', bgcolor='transparent')
    if not global_mode:
        dot.attr('graph', label=f'Ruta Académica para: {df_data["Rama_de_conocimiento"].iloc[0].split("+")[0]}', labelloc='t', fontsize='30')
    else:
        dot.attr('graph', label='Ruta Académica Global', labelloc='t', fontsize='40', size="40,60", dpi="300")

    # 4. Construir Capas
    # Capa 1: 1º Bachillerato
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='1º Bachillerato', style='filled', color='#F5F5F5')
        c.attr('node', shape='box', style='filled,rounded', color='#E8E8E8')
        nodos_1 = list(relaciones_1_a_2.keys())
        for nodo in nodos_1: c.node(nodo, nodo.replace('_', ' '))

    # Capa 2: 2º Bachillerato
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='2º Bachillerato (Asignaturas que ponderan)', style='filled', color='#E0E0E0')
        c.attr('node', shape='box', style='filled,rounded')
        for nodo in asignaturas_2_utiles:
            c.node(nodo, nodo.replace('_', ' '), color=color_asignatura[nodo] + '80') # Color con transparencia

    # Capa 4: Grados Universitarios (Agrupados por Rama)
    with dot.subgraph(name='cluster_4') as c:
        c.attr(label='Grados Universitarios', style='filled', color='#D0D0D0')
        c.attr('node', shape='box', style='filled,rounded')
        for rama in sorted(df_data['Rama_de_conocimiento'].unique()):
            with c.subgraph(name=f'cluster_rama_{rama.replace(" ", "")}') as sub_c:
                sub_c.attr(label=rama, style='filled', color='#C8C8C8')
                grados_en_rama = df_data[df_data['Rama_de_conocimiento'] == rama]['Grado']
                for grado in grados_en_rama:
                    label_grado = grado.replace(' + ', '+\n').replace(' y ', ' y\n').replace(' de ', ' de\n')
                    sub_c.node(grado, label_grado, color='#FFDAB9') # Color melocotón para los grados

    # 5. Crear Conexiones
    # 1º Bach -> 2º Bach
    for precursor, sucesores in relaciones_1_a_2.items():
        for sucesor in sucesores:
            if sucesor in asignaturas_2_utiles: dot.edge(precursor, sucesor)

    # 2º Bach -> Grados
    for asignatura in asignaturas_2_utiles:
        # Ponderación mínima para dibujar una línea
        min_pond = 0.2 if global_mode else 0.1
        
        grados_conectados = df_data[df_data[asignatura] >= min_pond]
        
        # Limitar en modo global
        if global_mode and max_grados_por_asignatura and len(grados_conectados) > max_grados_por_asignatura:
            grados_conectados = grados_conectados.nlargest(max_grados_por_asignatura, asignatura)

        for _, row in grados_conectados.iterrows():
            grado = row['Grado']
            ponderacion = row[asignatura]
            penwidth = '2.5' if ponderacion == 0.2 else '1.0'
            dot.edge(asignatura, grado, color=color_asignatura[asignatura], penwidth=penwidth)

    # 6. Renderizar
    print(f"Generando diagrama... se guardará como '{output_filename}.png'")
    dot.render(output_filename, format='png', view=True)
    print("¡Diagrama generado con éxito!")

# --- SCRIPT PRINCIPAL ---
if __name__ == "__main__":
    df_ponderaciones = cargar_y_limpiar_csv('ponderaciones_andalucia.csv')
    
    if df_ponderaciones is not None:
        
        # --- OPCIÓN 1: Generar un gráfico para una RAMA ESPECÍFICA ---
        # Descomenta la línea que te interese
        
        # crear_diagrama_filtrado(df_ponderaciones, rama_filter='IyA')
        # crear_diagrama_filtrado(df_ponderaciones, rama_filter='SD')
        # crear_diagrama_filtrado(df_ponderaciones, rama_filter='SyJ')
        # crear_diagrama_filtrado(df_ponderaciones, rama_filter='C')
        # crear_diagrama_filtrado(df_ponderaciones, rama_filter='AyH')

        # --- OPCIÓN 2: Generar el gráfico GLOBAL ---
        # Descomenta la siguiente línea para crear el gráfico con todas las conexiones.
        # ¡CUIDADO! Puede ser muy lento y generar un archivo muy grande.
        
        crear_diagrama_global(df_ponderaciones)