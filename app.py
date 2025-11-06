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
    def intervalo_confianza_varianza(datos: np.ndarray, confianza: float = 0.95) -> dict:
        """Intervalo de confianza para la varianza usando Chi-cuadrada."""
        n = len(datos)
        varianza_muestral = np.var(datos, ddof=1)
        desv_std = np.std(datos, ddof=1)
        
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
    def prueba_hipotesis_varianza(datos: np.ndarray, varianza_h0: float,
                                  hipotesis: str = 'bilateral', alpha: float = 0.05) -> dict:
        """Prueba de hipótesis para una varianza usando Chi-cuadrada."""
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
    def prueba_dos_varianzas(datos1: np.ndarray, datos2: np.ndarray,
                            hipotesis: str = 'bilateral', alpha: float = 0.05) -> dict:
        """Prueba F para comparar dos varianzas."""
        n1 = len(datos1)
        n2 = len(datos2)
        var1 = np.var(datos1, ddof=1)
        var2 = np.var(datos2, ddof=1)
        
        gl1 = n1 - 1
        gl2 = n2 - 1
        
        F_calc = var1 / var2
        
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
            'gl1': gl1,
            'gl2': gl2,
            'varianza1': var1,
            'varianza2': var2,
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
                resultado = pruebas.intervalo_confianza_varianza(datos, confianza)
                
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

# ============================================================================
# PRUEBA DE HIPÓTESIS (1 VARIANZA)
# ============================================================================
elif tipo_prueba == "Prueba de Hipótesis (1 Varianza)":
    st.header("🎯 Prueba de Hipótesis para Una Varianza")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        datos_input = st.text_area(
            "Ingresa los datos (separados por comas o espacios):",
            "12.5, 13.2, 11.8, 12.9, 13.5, 12.1, 13.8, 12.4, 13.1, 12.7"
        )
    
    with col2:
        varianza_h0 = st.number_input("Varianza bajo H₀ (σ₀²):", value=0.5, step=0.1, format="%.4f")
        hipotesis = st.selectbox("Tipo de hipótesis:", ["bilateral", "mayor", "menor"])
        alpha = st.slider("Nivel de significancia (α):", 0.01, 0.10, 0.05, 0.01)
    
    if st.button("Realizar Prueba", key="ph"):
        try:
            datos = np.array([float(x.strip()) for x in datos_input.replace(',', ' ').split()])
            
            if len(datos) < 2:
                st.error("Se necesitan al menos 2 datos")
            else:
                resultado = pruebas.prueba_hipotesis_varianza(datos, varianza_h0, hipotesis, alpha)
                
                st.subheader("Hipótesis:")
                col1, col2 = st.columns(2)
                col1.markdown(f"**H₀:** {resultado['h0']}")
                col2.markdown(f"**H₁:** {resultado['h1']}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("χ² calculado", f"{resultado['chi2_calculado']:.4f}")
                col2.metric("p-valor", f"{resultado['p_valor']:.6f}")
                col3.metric("Varianza muestral", f"{resultado['varianza_muestral']:.4f}")
                
                if resultado['rechazar_h0']:
                    st.error(f"❌ {resultado['conclusion']}")
                else:
                    st.success(f"✅ {resultado['conclusion']}")
                
                # Gráfico
                fig, ax = plt.subplots(figsize=(10, 5))
                x = np.linspace(0, stats.chi2.ppf(0.999, resultado['grados_libertad']), 1000)
                y = stats.chi2.pdf(x, resultado['grados_libertad'])
                
                ax.plot(x, y, 'b-', linewidth=2, label=f'χ²({resultado["grados_libertad"]})')
                ax.axvline(resultado['chi2_calculado'], color='red', linestyle='--', linewidth=2, label='χ² calculado')
                
                ax.set_xlabel('Valor χ²')
                ax.set_ylabel('Densidad')
                ax.set_title('Distribución Chi-cuadrada y Estadístico de Prueba')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error: {e}")

# ============================================================================
# PRUEBA F (2 VARIANZAS)
# ============================================================================
else:  # Prueba F
    st.header("⚖️ Prueba F para Dos Varianzas")
    
    st.markdown("Compara las varianzas de dos poblaciones independientes.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Muestra 1")
        datos1_input = st.text_area(
            "Datos de la muestra 1:",
            "12.5, 13.2, 11.8, 12.9, 13.5, 12.1",
            key="d1"
        )
    
    with col2:
        st.subheader("Muestra 2")
        datos2_input = st.text_area(
            "Datos de la muestra 2:",
            "15.2, 14.8, 16.1, 15.5, 14.3, 15.9, 16.2",
            key="d2"
        )
    
    col1, col2 = st.columns(2)
    with col1:
        hipotesis = st.selectbox("Tipo de hipótesis:", ["bilateral", "mayor", "menor"], key="hip_f")
    with col2:
        alpha = st.slider("Nivel de significancia (α):", 0.01, 0.10, 0.05, 0.01, key="alpha_f")
    
    if st.button("Realizar Prueba F", key="pf"):
        try:
            datos1 = np.array([float(x.strip()) for x in datos1_input.replace(',', ' ').split()])
            datos2 = np.array([float(x.strip()) for x in datos2_input.replace(',', ' ').split()])
            
            if len(datos1) < 2 or len(datos2) < 2:
                st.error("Cada muestra necesita al menos 2 datos")
            else:
                resultado = pruebas.prueba_dos_varianzas(datos1, datos2, hipotesis, alpha)
                
                st.subheader("Hipótesis:")
                col1, col2 = st.columns(2)
                col1.markdown(f"**H₀:** {resultado['h0']}")
                col2.markdown(f"**H₁:** {resultado['h1']}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("F calculado", f"{resultado['F_calculado']:.4f}")
                col2.metric("p-valor", f"{resultado['p_valor']:.6f}")
                col3.metric("α", alpha)
                
                col1, col2 = st.columns(2)
                col1.metric("Varianza 1", f"{resultado['varianza1']:.4f}")
                col2.metric("Varianza 2", f"{resultado['varianza2']:.4f}")
                
                if resultado['rechazar_h0']:
                    st.error(f"❌ {resultado['conclusion']}")
                else:
                    st.success(f"✅ {resultado['conclusion']}")
                
                # Gráficos
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                # Boxplot comparativo
                ax1.boxplot([datos1, datos2], labels=['Muestra 1', 'Muestra 2'])
                ax1.set_title('Comparación de Muestras')
                ax1.set_ylabel('Valores')
                ax1.grid(True, alpha=0.3)
                
                # Distribución F
                x = np.linspace(0.01, stats.f.ppf(0.99, resultado['gl1'], resultado['gl2']), 1000)
                y = stats.f.pdf(x, resultado['gl1'], resultado['gl2'])
                
                ax2.plot(x, y, 'b-', linewidth=2, label=f'F({resultado["gl1"]}, {resultado["gl2"]})')
                ax2.axvline(resultado['F_calculado'], color='red', linestyle='--', linewidth=2, label='F calculado')
                ax2.set_xlabel('Valor F')
                ax2.set_ylabel('Densidad')
                ax2.set_title('Distribución F')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>📊 Desarrollado para análisis estadístico de varianzas</p>
    <p>Utiliza distribuciones Chi-cuadrada (χ²) y F</p>
</div>
""", unsafe_allow_html=True)
