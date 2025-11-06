"""
Pruebas de Hipótesis e Intervalos de Confianza para Varianzas
Autor: [Tu nombre]
Descripción: Implementación de pruebas estadísticas para una y dos varianzas
             usando distribuciones Chi-cuadrada y F, con prueba de normalidad.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class PruebasVarianza:
    """Clase para realizar pruebas de hipótesis sobre varianzas."""
    
    @staticmethod
    def prueba_normalidad(datos: np.ndarray, alpha: float = 0.05) -> dict:
        """
        Prueba de Shapiro-Wilk para normalidad (sencilla y potente).
        
        Parámetros:
        -----------
        datos : array
            Muestra de datos a evaluar
        alpha : float
            Nivel de significancia (default: 0.05)
            
        Retorna:
        --------
        dict con estadístico, p-valor y decisión
        """
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
    def intervalo_confianza_varianza(datos: np.ndarray, 
                                     confianza: float = 0.95) -> dict:
        """
        Intervalo de confianza para la varianza usando Chi-cuadrada.
        
        Parámetros:
        -----------
        datos : array
            Muestra de datos
        confianza : float
            Nivel de confianza (default: 0.95)
            
        Retorna:
        --------
        dict con varianza muestral, límites del intervalo y desviación estándar
        """
        n = len(datos)
        varianza_muestral = np.var(datos, ddof=1)
        desv_std = np.std(datos, ddof=1)
        
        # Grados de libertad
        gl = n - 1
        
        # Valores críticos de Chi-cuadrada
        alpha = 1 - confianza
        chi2_inf = stats.chi2.ppf(alpha/2, gl)
        chi2_sup = stats.chi2.ppf(1 - alpha/2, gl)
        
        # Intervalo de confianza para varianza
        ic_varianza_inf = (gl * varianza_muestral) / chi2_sup
        ic_varianza_sup = (gl * varianza_muestral) / chi2_inf
        
        # Intervalo de confianza para desviación estándar
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
    def prueba_hipotesis_varianza(datos: np.ndarray, 
                                  varianza_h0: float,
                                  hipotesis: str = 'bilateral',
                                  alpha: float = 0.05) -> dict:
        """
        Prueba de hipótesis para una varianza usando Chi-cuadrada.
        
        Parámetros:
        -----------
        datos : array
            Muestra de datos
        varianza_h0 : float
            Varianza bajo la hipótesis nula
        hipotesis : str
            Tipo de hipótesis: 'bilateral', 'menor', 'mayor'
        alpha : float
            Nivel de significancia
            
        Retorna:
        --------
        dict con estadístico, p-valor y decisión
        """
        n = len(datos)
        varianza_muestral = np.var(datos, ddof=1)
        gl = n - 1
        
        # Estadístico Chi-cuadrada
        chi2_calc = (gl * varianza_muestral) / varianza_h0
        
        # P-valor según tipo de hipótesis
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
        else:  # mayor
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
    def prueba_dos_varianzas(datos1: np.ndarray, 
                            datos2: np.ndarray,
                            hipotesis: str = 'bilateral',
                            alpha: float = 0.05) -> dict:
        """
        Prueba F para comparar dos varianzas.
        
        Parámetros:
        -----------
        datos1, datos2 : array
            Muestras a comparar
        hipotesis : str
            Tipo de hipótesis: 'bilateral', 'menor', 'mayor'
        alpha : float
            Nivel de significancia
            
        Retorna:
        --------
        dict con estadístico F, p-valor y decisión
        """
        n1 = len(datos1)
        n2 = len(datos2)
        var1 = np.var(datos1, ddof=1)
        var2 = np.var(datos2, ddof=1)
        
        gl1 = n1 - 1
        gl2 = n2 - 1
        
        # Estadístico F (varianza mayor en numerador por convención)
        F_calc = var1 / var2
        
        # P-valor según tipo de hipótesis
        if hipotesis == 'bilateral':
            # Para prueba bilateral, siempre ponemos la mayor varianza arriba
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
        else:  # menor
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


def imprimir_resultados(resultados: dict, titulo: str = ""):
    """Función auxiliar para imprimir resultados de forma legible."""
    print("\n" + "="*70)
    print(f"  {titulo}")
    print("="*70)
    for key, value in resultados.items():
        if isinstance(value, tuple):
            print(f"{key}: ({value[0]:.6f}, {value[1]:.6f})")
        elif isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print("="*70 + "\n")


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Configurar semilla para reproducibilidad
    np.random.seed(42)
    
    # Crear instancia de la clase
    pruebas = PruebasVarianza()
    
    # ========================================================================
    # EJEMPLO 1: Prueba de normalidad
    # ========================================================================
    print("\n" + "█"*70)
    print("  EJEMPLO 1: PRUEBA DE NORMALIDAD (Shapiro-Wilk)")
    print("█"*70)
    
    # Datos que siguen distribución normal
    datos_normales = np.random.normal(100, 15, 50)
    resultado_norm = pruebas.prueba_normalidad(datos_normales)
    imprimir_resultados(resultado_norm, "Datos Normales")
    
    # Datos que NO siguen distribución normal (exponencial)
    datos_no_normales = np.random.exponential(2, 50)
    resultado_no_norm = pruebas.prueba_normalidad(datos_no_normales)
    imprimir_resultados(resultado_no_norm, "Datos No Normales")
    
    # ========================================================================
    # EJEMPLO 2: Intervalo de confianza para UNA varianza
    # ========================================================================
    print("\n" + "█"*70)
    print("  EJEMPLO 2: INTERVALO DE CONFIANZA PARA VARIANZA")
    print("█"*70)
    
    datos = np.array([12.5, 13.2, 11.8, 12.9, 13.5, 12.1, 13.8, 12.4, 13.1, 12.7])
    resultado_ic = pruebas.intervalo_confianza_varianza(datos, confianza=0.95)
    imprimir_resultados(resultado_ic, "IC al 95% para Varianza")
    
    # ========================================================================
    # EJEMPLO 3: Prueba de hipótesis para UNA varianza
    # ========================================================================
    print("\n" + "█"*70)
    print("  EJEMPLO 3: PRUEBA DE HIPÓTESIS PARA UNA VARIANZA")
    print("█"*70)
    
    # H0: σ² = 0.5  vs  H1: σ² ≠ 0.5
    resultado_ph = pruebas.prueba_hipotesis_varianza(
        datos, 
        varianza_h0=0.5, 
        hipotesis='bilateral',
        alpha=0.05
    )
    imprimir_resultados(resultado_ph, "Prueba Chi-cuadrada (bilateral)")
    
    # ========================================================================
    # EJEMPLO 4: Prueba F para DOS varianzas
    # ========================================================================
    print("\n" + "█"*70)
    print("  EJEMPLO 4: PRUEBA F PARA DOS VARIANZAS")
    print("█"*70)
    
    # Dos muestras con varianzas diferentes
    muestra1 = np.random.normal(50, 5, 25)   # media=50, desv=5
    muestra2 = np.random.normal(50, 8, 30)   # media=50, desv=8
    
    resultado_f = pruebas.prueba_dos_varianzas(
        muestra1, 
        muestra2,
        hipotesis='bilateral',
        alpha=0.05
    )
    imprimir_resultados(resultado_f, "Prueba F para Igualdad de Varianzas")
    
    # ========================================================================
    # VISUALIZACIÓN
    # ========================================================================
    print("\n" + "█"*70)
    print("  GENERANDO GRÁFICOS...")
    print("█"*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Gráfico 1: Q-Q plot para normalidad
    stats.probplot(datos_normales, dist="norm", plot=axes[0, 0])
    axes[0, 0].set_title('Q-Q Plot: Datos Normales')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfico 2: Histograma de datos
    axes[0, 1].hist(datos, bins=7, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 1].axvline(np.mean(datos), color='red', linestyle='--', label='Media')
    axes[0, 1].set_title('Distribución de Datos')
    axes[0, 1].set_xlabel('Valores')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gráfico 3: Comparación de muestras
    axes[1, 0].boxplot([muestra1, muestra2], labels=['Muestra 1', 'Muestra 2'])
    axes[1, 0].set_title('Comparación de Dos Muestras')
    axes[1, 0].set_ylabel('Valores')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gráfico 4: Distribución Chi-cuadrada
    x = np.linspace(0, 30, 1000)
    gl = len(datos) - 1
    y = stats.chi2.pdf(x, gl)
    axes[1, 1].plot(x, y, 'b-', linewidth=2, label=f'χ²({gl})')
    axes[1, 1].axvline(resultado_ph['chi2_calculado'], color='red', 
                       linestyle='--', label='Estadístico calculado')
    axes[1, 1].set_title('Distribución Chi-cuadrada')
    axes[1, 1].set_xlabel('Valor')
    axes[1, 1].set_ylabel('Densidad')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('resultados_pruebas_varianza.png', dpi=300, bbox_inches='tight')
    print("✓ Gráficos guardados en 'resultados_pruebas_varianza.png'")
    plt.show()
    
    print("\n" + "█"*70)
    print("  ¡ANÁLISIS COMPLETADO!")
    print("█"*70 + "\n")
