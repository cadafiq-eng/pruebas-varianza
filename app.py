"""
Aplicación Streamlit para Pruebas de Hipótesis de Varianzas
"""

import streamlit as st
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="Pruebas de Varianza",
    page_icon="📊",
    layout="wide"
)

# Título principal
st.title("📊 Pruebas de Hipótesis para Varianzas")
st.markdown("---")

# Clase de pruebas (del código original)
class PruebasVarianza:
    """Clase para realizar pruebas de hipótesis sobre varianzas."""
    
    @staticmethod
    def prueba_normalidad(datos: np.ndarray, alpha: float = 0.05) -> dict:
        """Prueba de Shapiro-Wilk para normalidad."""
        stat, p_valor = stats.shapiro(datos)
        
        resultado = {
            'test': 'Shapiro-Wilk',
            'estadistico': stat,
            'p_valor': p_valor,
            'alpha': alpha,
            'es_normal': p_valor > alpha,
            'conclusion': f"Los datos {'SÍ' if p_valor > alpha else 'NO'} siguen una distribución normal (α={alpha})"
        }
        
        return resultado
    
    @staticmethod
    def intervalo_confianza_varianza(datos: np.ndarray = None, 
                                     confianza: float = 0.95,
                                     n: int = None,
                                     varianza_muestral: float = None) -> dict:
        """Intervalo de confianza para la varianza usando Chi-cuadrada."""
        if datos is not None:
            n = len(datos)
            varianza_muestral = np.var(datos, ddof=1)
            desv_std = np.std(datos, ddof=1)
        else:
            desv_std = np.sqrt(varianza_muestral)
        
        gl = n - 1
        alpha = 1 - confianza
        chi2_inf = stats.chi2.ppf(alpha/2, gl)
        chi2_sup = stats.chi2.ppf(1 - alpha/2, gl)
        
        ic_varianza_inf = (gl * varianza_muestral) / chi2_sup
        ic_varianza_sup = (gl * varianza_muestral) / chi2_inf
        
        ic_desv_inf = np.sqrt(ic_varianza_inf)
        ic_desv_sup = np.sqrt(ic_varianza_sup)
        
        resultado = {
            'n': n,
            'grados_libertad': gl,
            'varianza_muestral': varianza_muestral,
            'desviacion_estandar': desv_std,
            'confianza': confianza,
            'ic_varianza': (ic_varianza_inf, ic_varianza_sup),
            'ic_desviacion': (ic_desv_inf, ic_desv_sup),
            'chi2_critico_inf': chi2_inf,
            'chi2_critico_sup': chi2_sup
        }
        
        return resultado
    
    @staticmethod
    def prueba_hipotesis_varianza(datos: np.ndarray = None, 
                                  varianza_h0: float = None,
                                  hipotesis: str = 'bilateral', 
                                  alpha: float = 0.05,
                                  n: int = None,
                                  varianza_muestral: float = None) -> dict:
        """Prueba de hipótesis para una varianza usando Chi-cuadrada."""
        if datos is not None:
            n = len(datos)
            varianza_muestral = np.var(datos, ddof=1)
        
        gl = n - 1
        
        chi2_calc = (gl * varianza_muestral) / varianza_h0
        
        if hipotesis == 'bilateral':
            p_valor = 2 * min(stats.chi2.cdf(chi2_calc, gl), 
                            1 - stats.chi2.cdf(chi2_calc, gl))
            chi2_critico_inf = stats.chi2.ppf(alpha/2, gl)
            chi2_critico_sup = stats.chi2.ppf(1 - alpha/2, gl)
            rechazar = chi2_calc < chi2_critico_inf or chi2_calc > chi2_critico_sup
            h_alternativa = f"σ² ≠ {varianza_h0}"
        elif hipotesis == 'menor':
            p_valor = stats.chi2.cdf(chi2_calc, gl)
            chi2_critico = stats.chi2.ppf(alpha, gl)
            rechazar = chi2_calc < chi2_critico
            h_alternativa = f"σ² < {varianza_h0}"
        else:
            p_valor = 1 - stats.chi2.cdf(chi2_calc, gl)
            chi2_critico = stats.chi2.ppf(1 - alpha, gl)
            rechazar = chi2_calc > chi2_critico
            h_alternativa = f"σ² > {varianza_h0}"
        
        resultado = {
            'n': n,
            'grados_libertad': gl,
            'varianza_muestral': varianza_muestral,
            'varianza_h0': varianza_h0,
            'hipotesis': hipotesis,
            'chi2_calculado': chi2_calc,
            'p_valor': p_valor,
            'alpha': alpha,
            'rechazar_h0': rechazar,
            'h0': f"σ² = {varianza_h0}",
            'h1': h_alternativa,
            'conclusion': f"{'Rechazamos' if rechazar else 'No rechazamos'} H0 al nivel α={alpha}"
        }
        
        return resultado
    
    @staticmethod
    def prueba_dos_varianzas(datos1: np.ndarray = None, 
                            datos2: np.ndarray = None,
                            hipotesis: str = 'bilateral', 
                            alpha: float = 0.05,
                            n1: int = None,
                            n2: int = None,
                            var1: float = None,
                            var2: float = None) -> dict:
        """Prueba F para comparar dos varianzas."""
        if datos1 is not None and datos2 is not None:
            n1 = len(datos1)
            n2 = len(datos2)
            var1 = np.var(datos1, ddof=1)
            var2 = np.var(datos2, ddof=1)
        
        gl1 = n1 - 1
        gl2 = n2 - 1
        
        F_calc = var1 / var2
        var1_orig, var2_orig = var1, var2
        gl1_orig, gl2_orig = gl1, gl2
        
        if hipotesis == 'bilateral':
            if F_calc < 1:
                F_calc = 1 / F_calc
                gl1, gl2 = gl2, gl1
                var1, var2 = var2, var1
            p_valor = 2 * (1 - stats.f.cdf(F_calc, gl1, gl2))
            F_critico_sup = stats.f.ppf(1 - alpha/2, gl1, gl2)
            F_critico_inf = stats.f.ppf(alpha/2, gl1, gl2)
            rechazar = F_calc > F_critico_sup
            h_alternativa = "σ₁² ≠ σ₂²"
        elif hipotesis == 'mayor':
            p_valor = 1 - stats.f.cdf(F_calc, gl1, gl2)
            F_critico = stats.f.ppf(1 - alpha, gl1, gl2)
            rechazar = F_calc > F_critico
            h_alternativa = "σ₁² > σ₂²"
        else:
            p_valor = stats.f.cdf(F_calc, gl1, gl2)
            F_critico = stats.f.ppf(alpha, gl1, gl2)
            rechazar = F_calc < F_critico
            h_alternativa = "σ₁² < σ₂²"
        
        resultado = {
            'n1': n1,
            'n2': n2,
            'gl1': gl1_orig,
            'gl2': gl2_orig,
            'varianza1': var1_orig,
            'varianza2': var2_orig,
            'hipotesis': hipotesis,
            'F_calculado': F_calc,
            'p_valor': p_valor,
            'alpha': alpha,
            'rechazar_h0': rechazar,
            'h0': "σ₁² = σ₂²",
            'h1': h_alternativa,
            'conclusion': f"{'Rechazamos' if rechazar else 'No rechazamos'} H0 al nivel α={alpha}"
        }
        
        return resultado


# Sidebar para selección de prueba
st.sidebar.title("⚙️ Configuración")
tipo_prueba = st.sidebar.selectbox(
    "Selecciona el tipo de prueba:",
    ["Prueba de Normalidad", 
     "Intervalo de Confianza (1 Varianza)",
     "Prueba de Hipótesis (1 Varianza)",
     "Prueba F (2 Varianzas)"]
)

# Instanciar clase
pruebas = PruebasVarianza()

# ============================================================================
# PRUEBA DE NORMALIDAD
# ============================================================================
if tipo_prueba == "Prueba de Normalidad":
    st.header("🔍 Prueba de Normalidad (Shapiro-Wilk)")
    
    st.markdown("""
    Esta prueba evalúa si los datos siguen una distribución normal.
    - **H₀**: Los datos siguen una distribución normal
    - **H₁**: Los datos NO siguen una distribución normal
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        datos_input = st.text_area(
            "Ingresa los datos (separados por comas o espacios):",
            "12.5, 13.2, 11.8, 12.9, 13.5, 12.1, 13.8, 12.4, 13.1, 12.7"
        )
    
    with col2:
        alpha = st.slider("Nivel de significancia (α):", 0.01, 0.10, 0.05, 0.01)
    
    if st.button("Realizar Prueba", key="norm"):
        try:
            datos = np.array([float(x.strip()) for x in datos_input.replace(',', ' ').split()])
            
            if len(datos) < 3:
                st.error("Se necesitan al menos 3 datos")
            else:
                resultado = pruebas.prueba_normalidad(datos, alpha)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Tamaño de muestra", len(datos))
                col2.metric("Estadístico W", f"{resultado['estadistico']:.6f}")
                col3.metric("p-valor", f"{resultado['p_valor']:.6f}")
                
                if resultado['es_normal']:
                    st.success(f"✅ {resultado['conclusion']}")
                else:
                    st.warning(f"⚠️ {resultado['conclusion']}")
                
                # Gráficos
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Histograma
                ax1.hist(datos, bins=min(10, len(datos)//2), edgecolor='black', alpha=0.7)
                ax1.set_title('Histograma de Datos')
                ax1.set_xlabel('Valores')
                ax1.set_ylabel('Frecuencia')
                ax1.grid(True, alpha=0.3)
                
                # Q-Q Plot
                stats.probplot(datos, dist="norm", plot=ax2)
                ax2.set_title('Q-Q Plot')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error: {e}")

# ============================================================================
# INTERVALO DE CONFIANZA
# ============================================================================
elif tipo_prueba == "Intervalo de Confianza (1 Varianza)":
    st.header("📏 Intervalo de Confianza para Varianza")
    
    st.markdown("Calcula el intervalo de confianza para la varianza poblacional usando la distribución Chi-cuadrada.")
    
    # Selector de método de entrada
    metodo = st.radio(
        "¿Cómo deseas ingresar la información?",
        ["📊 Ingresar datos crudos", "📐 Ingresar estadísticos calculados (n y s²)"],
        horizontal=True
    )
    
    if metodo == "📊 Ingresar datos crudos":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            datos_input = st.text_area(
                "Ingresa los datos (separados por comas o espacios):",
                "12.5, 13.2, 11.8, 12.9, 13.5, 12.1, 13.8, 12.4, 13.1, 12.7"
            )
        
        with col2:
            confianza = st.slider("Nivel de confianza:", 0.80, 0.99, 0.95, 0.01)
        
        if st.button("Calcular IC", key="ic"):
            try:
                datos = np.array([float(x.strip()) for x in datos_input.replace(',', ' ').split()])
                
                if len(datos) < 2:
                    st.error("Se necesitan al menos 2 datos")
                else:
                    resultado = pruebas.intervalo_confianza_varianza(datos=datos, confianza=confianza)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Tamaño muestra", resultado['n'])
                    col2.metric("Varianza muestral", f"{resultado['varianza_muestral']:.4f}")
                    col3.metric("Desv. estándar", f"{resultado['desviacion_estandar']:.4f}")
                    
                    st.subheader(f"📊 Intervalo de Confianza al {confianza*100:.0f}%")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"""
                        **Para la Varianza:**
                        - Límite inferior: {resultado['ic_varianza'][0]:.4f}
                        - Límite superior: {resultado['ic_varianza'][1]:.4f}
                        """)
                    
                    with col2:
                        st.info(f"""
                        **Para la Desviación Estándar:**
                        - Límite inferior: {resultado['ic_desviacion'][0]:.4f}
                        - Límite superior: {resultado['ic_desviacion'][1]:.4f}
                        """)
                    
                    # Gráfico
                    fig, ax = plt.subplots(figsize=(10, 4))
                    x = np.linspace(0, stats.chi2.ppf(0.999, resultado['grados_libertad']), 1000)
                    y = stats.chi2.pdf(x, resultado['grados_libertad'])
                    
                    ax.plot(x, y, 'b-', linewidth=2, label=f'χ²({resultado["grados_libertad"]})')
                    ax.axvline(resultado['chi2_critico_inf'], color='red', linestyle='--', label='Límites IC')
                    ax.axvline(resultado['chi2_critico_sup'], color='red', linestyle='--')
                    ax.fill_between(x, y, where=(x >= resultado['chi2_critico_inf']) & (x <= resultado['chi2_critico_sup']), 
                                   alpha=0.3, color='green', label=f'IC {confianza*100:.0f}%')
                    ax.set_xlabel('Valor χ²')
                    ax.set_ylabel('Densidad')
                    ax.set_title('Distribución Chi-cuadrada')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Error: {e}")
    
    else:  # Ingresar estadísticos calculados
        st.info("💡 Ingresa el tamaño de muestra (n) y la varianza muestral (s²) que ya calculaste")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_input = st.number_input("Tamaño de muestra (n):", min_value=2, value=10, step=1)
        
        with col2:
            var_input = st.number_input("Varianza muestral (s²):", min_value=0.0001, value=0.5, step=0.01, format="%.4f")
        
        with col3:
            confianza = st.slider("Nivel de confianza:", 0.80, 0.99, 0.95, 0.01, key="conf_calc")
        
        if st.button("Calcular IC", key="ic_calc"):
            try:
                resultado = pruebas.intervalo_confianza_varianza(
                    n=n_input,
                    varianza_muestral=var_input,
                    confianza=confianza
                )
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Tamaño muestra", resultado['n'])
                col2.metric("Varianza muestral", f"{resultado['varianza_muestral']:.4f}")
                col3.metric("Desv. estándar", f"{resultado['desviacion_estandar']:.4f}")
                
                st.subheader
