import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- FUNCIÓN PARA CARGAR Y LIMPIAR EL CSV ---
def cargar_y_limpiar_csv(filepath):
    """
    Carga el archivo CSV, lo limpia y prepara para el análisis.
    """
    # Leer el archivo, saltando las filas finales que no son datos de grados.
    try:
        # Usamos 'latin1' encoding que suele ser más robusto con CSVs de origen español.
        df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{filepath}'.")
        print("Asegúrate de que el archivo CSV esté en el mismo directorio que el script.")
        return None

    # Limpiar el nombre de la primera columna que contiene texto extra
    df.rename(columns={df.columns[0]: 'Grado'}, inplace=True)

    # Eliminar filas que no son grados (resúmenes, notas al pie, etc.)
    # Identificamos estas filas porque suelen tener valores nulos en 'Rama de conocimiento'
    df = df.dropna(subset=['Rama de conocimiento'])
    # También eliminamos filas de resumen que puedan haberse colado
    df = df[~df['Grado'].str.contains("Pondera|No pondera|This work", case=False, na=False)]

    # Limpiar y estandarizar los nombres de las columnas (asignaturas)
    # Quitando espacios, acentos y caracteres especiales para facilitar el manejo.
    nuevas_columnas = {col: col.strip().replace(' ', '_').replace('-', '_') for col in df.columns}
    df.rename(columns=nuevas_columnas, inplace=True)
    
    # Iterar por las columnas de asignaturas para convertir los valores
    for col in df.columns[2:]: # Empezamos desde la tercera columna (después de Grado y Rama)
        # Reemplazar la coma decimal por un punto
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.', regex=False)
        # Convertir a número, los errores (celdas vacías) se convertirán en NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Rellenar los valores NaN (celdas vacías) con 0.0
    df.fillna(0.0, inplace=True)
    
    # Mapeo de códigos de Rama a nombres completos
    ramas_map = {
        'SyJ': 'Ciencias Sociales y Jurídicas',
        'SD': 'Ciencias de la Salud',
        'C': 'Ciencias',
        'IyA': 'Ingeniería y Arquitectura',
        'AyH': 'Artes y Humanidades',
        'AyH+C': 'Artes y Humanidades + Ciencias',
        'IyA+C': 'Ingeniería y Arquitectura + Ciencias',
        'AyH+SyJ': 'Artes y Humanidades + CC. Sociales',
        'C+IyA': 'Ciencias + Ingeniería y Arquitectura',
        'C+SD': 'Ciencias + CC. de la Salud',
        'SD+SyJ': 'CC. de la Salud + CC. Sociales',
        'IyA+SyJ': 'Ingeniería y Arquitectura + CC. Sociales',
        'C + SyJ': 'Ciencias + CC. Sociales' # Corregido por si hay espacios
    }
    # Para simplificar el análisis principal, agruparemos las ramas dobles en sus componentes
    # o las trataremos por separado si es necesario. Por ahora, nos centramos en las 5 principales.
    df['Rama_Principal'] = df['Rama_de_conocimiento'].apply(lambda x: ramas_map.get(x.split('+')[0].strip(), 'Otro'))
    
    return df

# --- FUNCIÓN PARA ANALIZAR Y VISUALIZAR ---
def analizar_y_visualizar_por_rama(df):
    """
    Analiza la utilidad de las asignaturas por rama y crea un gráfico para cada una.
    """
    if df is None:
        return

    # Convertir a formato largo para facilitar la agrupación
    id_vars = ['Grado', 'Rama_Principal']
    value_vars = [col for col in df.columns if col not in id_vars and col not in ['Rama_de_conocimiento']]
    df_long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='Asignatura', value_name='Ponderacion')

    # Filtramos solo las ponderaciones que consideramos "útiles" (0.2) y "ligeramente útiles" (0.15)
    df_util = df_long[df_long['Ponderacion'] >= 0.15].copy()

    # Contamos cuántos grados en cada rama consideran útil cada asignatura
    utilidad_por_rama = df_util.groupby(['Rama_Principal', 'Asignatura']).size().reset_index(name='Num_Grados_Utiles')

    # Definimos las 5 ramas principales
    ramas_principales = [
        'Ciencias', 'Ciencias de la Salud', 'Ingeniería y Arquitectura',
        'Ciencias Sociales y Jurídicas', 'Artes y Humanidades'
    ]

    for rama in ramas_principales:
        print(f"\n--- Procesando Rama: {rama} ---")
        
        # Filtrar datos para la rama actual
        resultado_rama = utilidad_por_rama[utilidad_por_rama['Rama_Principal'] == rama]
        
        if resultado_rama.empty:
            print(f"No se encontraron asignaturas con ponderación >= 0.15 para la rama {rama}.")
            continue

        # Ordenar y tomar el Top 10
        top_10_asignaturas = resultado_rama.sort_values(by='Num_Grados_Utiles', ascending=False).head(10)
        
        # Crear la visualización
        plt.figure(figsize=(12, 8))
        sns.barplot(
            x='Num_Grados_Utiles',
            y='Asignatura',
            data=top_10_asignaturas,
            palette='viridis'
        )
        
        plt.title(f'Top 10 Asignaturas más Útiles (Pond. >= 0.15) para\n{rama}', fontsize=16)
        plt.xlabel('Número de Grados para los que Pondera', fontsize=12)
        plt.ylabel('Asignatura', fontsize=12)
        plt.tight_layout()
        
        # Añadir el número en cada barra para mayor claridad
        for index, value in enumerate(top_10_asignaturas['Num_Grados_Utiles']):
            plt.text(value, index, f' {value}', va='center')
            
        # Guardar el gráfico
        nombre_archivo = f"top_asignaturas_{rama.replace(' ', '_').lower()}.png"
        plt.savefig(nombre_archivo)
        print(f"Gráfico guardado como: {nombre_archivo}")
        plt.show()

# --- SCRIPT PRINCIPAL ---
if __name__ == "__main__":
    archivo_csv = 'ponderaciones_andalucia.csv'
    df_ponderaciones = cargar_y_limpiar_csv(archivo_csv)
    
    if df_ponderaciones is not None:
        print("\n--- Datos cargados y limpios. Iniciando análisis por rama. ---")
        analizar_y_visualizar_por_rama(df_ponderaciones)