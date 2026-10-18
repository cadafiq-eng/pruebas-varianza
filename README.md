# 📊 Pruebas de Hipótesis para Varianzas

> **🌐 Versión web disponible (recomendada)**
>
> Esta herramienta ya existe como app web: funciona en cualquier navegador y se instala en el celular, sin Python ni servidor.
> 👉 **[Muestreo e Inferencia — HumAIno®](https://cadafiq-eng.github.io/muestreo-inferencia/)** · pestaña *Pruebas de Varianza*
>
> Este repositorio se conserva como la versión original en Python/Streamlit y como referencia del código. La app web se distribuye bajo [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.es); el código de este repositorio conserva su licencia original.

Implementación en Python de pruebas estadísticas para análisis de varianzas utilizando distribuciones Chi-cuadrada (χ²) y F, con prueba de normalidad incluida.

## 🎯 Características

- ✅ **Prueba de Normalidad**: Test de Shapiro-Wilk
- ✅ **Intervalo de Confianza**: Para una varianza (distribución χ²)
- ✅ **Prueba de Hipótesis**: Para una varianza (distribución χ²)
- ✅ **Prueba F**: Comparación de dos varianzas
- ✅ **Visualizaciones**: Q-Q plots, histogramas, boxplots y distribuciones
- ✅ **Código documentado**: Con ejemplos completos de uso

## 📋 Requisitos

```bash
Python 3.7+
numpy
scipy
matplotlib
```

## 🚀 Instalación

1. Clona este repositorio:
```bash
git clone https://github.com/tu-usuario/pruebas-varianza.git
cd pruebas-varianza
```

2. Instala las dependencias:
```bash
pip install numpy scipy matplotlib
```

## 💻 Uso Rápido

```python
from pruebas_varianza import PruebasVarianza
import numpy as np

# Crear instancia
pruebas = PruebasVarianza()

# Ejemplo 1: Probar normalidad
datos = np.random.normal(100, 15, 50)
resultado = pruebas.prueba_normalidad(datos)
print(resultado['conclusion'])

# Ejemplo 2: Intervalo de confianza para varianza
datos = np.array([12.5, 13.2, 11.8, 12.9, 13.5, 12.1])
ic = pruebas.intervalo_confianza_varianza(datos, confianza=0.95)
print(f"IC 95%: {ic['ic_varianza']}")

# Ejemplo 3: Prueba de hipótesis para una varianza
# H0: σ² = 0.5  vs  H1: σ² ≠ 0.5
resultado = pruebas.prueba_hipotesis_varianza(
    datos, 
    varianza_h0=0.5, 
    hipotesis='bilateral',
    alpha=0.05
)
print(resultado['conclusion'])

# Ejemplo 4: Comparar dos varianzas (Prueba F)
muestra1 = np.random.normal(50, 5, 25)
muestra2 = np.random.normal(50, 8, 30)
resultado = pruebas.prueba_dos_varianzas(muestra1, muestra2)
print(resultado['conclusion'])
```

## 📚 Documentación de Métodos

### `prueba_normalidad(datos, alpha=0.05)`
Realiza la prueba de Shapiro-Wilk para evaluar normalidad.

**Parámetros:**
- `datos` (array): Muestra de datos a evaluar
- `alpha` (float): Nivel de significancia (default: 0.05)

**Retorna:** dict con estadístico, p-valor y conclusión

---

### `intervalo_confianza_varianza(datos, confianza=0.95)`
Calcula el intervalo de confianza para la varianza usando χ².

**Parámetros:**
- `datos` (array): Muestra de datos
- `confianza` (float): Nivel de confianza (default: 0.95)

**Retorna:** dict con varianza muestral, límites del IC y desviación estándar

---

### `prueba_hipotesis_varianza(datos, varianza_h0, hipotesis='bilateral', alpha=0.05)`
Prueba de hipótesis para una varianza usando χ².

**Parámetros:**
- `datos` (array): Muestra de datos
- `varianza_h0` (float): Varianza bajo H₀
- `hipotesis` (str): 'bilateral', 'menor' o 'mayor'
- `alpha` (float): Nivel de significancia

**Retorna:** dict con estadístico χ², p-valor y decisión

---

### `prueba_dos_varianzas(datos1, datos2, hipotesis='bilateral', alpha=0.05)`
Prueba F para comparar dos varianzas.

**Parámetros:**
- `datos1`, `datos2` (array): Muestras a comparar
- `hipotesis` (str): 'bilateral', 'menor' o 'mayor'
- `alpha` (float): Nivel de significancia

**Retorna:** dict con estadístico F, p-valor y decisión

## 📊 Ejemplo Completo

Ejecuta el script principal para ver todos los ejemplos:

```bash
python pruebas_varianza.py
```

Esto generará:
- Salida detallada de cada prueba estadística
- Gráfico `resultados_pruebas_varianza.png` con 4 visualizaciones

## 🎓 Fundamento Teórico

### Distribución Chi-cuadrada (χ²)
Para una muestra de tamaño n de una población normal con varianza σ²:

$$\chi^2 = \frac{(n-1)s^2}{\sigma^2} \sim \chi^2_{(n-1)}$$

### Distribución F
Para comparar dos varianzas muestrales s₁² y s₂²:

$$F = \frac{s_1^2}{s_2^2} \sim F_{(n_1-1, n_2-1)}$$

### Prueba de Shapiro-Wilk
Evalúa si una muestra proviene de una distribución normal:
- H₀: Los datos siguen una distribución normal
- H₁: Los datos no siguen una distribución normal

## 📈 Visualizaciones

El script genera automáticamente:
1. **Q-Q Plot**: Para evaluar normalidad visualmente
2. **Histograma**: Distribución de los datos
3. **Boxplots**: Comparación de dos muestras
4. **Distribución χ²**: Con estadístico calculado

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/NuevaCaracteristica`)
3. Commit tus cambios (`git commit -m 'Agrega nueva característica'`)
4. Push a la rama (`git push origin feature/NuevaCaracteristica`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para más detalles.

## 👤 Autor

**[Luis Chel-Guerrero]**
- GitHub: [@tu-usuario](https://github.com/tu-usuario)
- Email: tu-email@ejemplo.com

## 📞 Contacto y Soporte

Si tienes preguntas o sugerencias, por favor abre un [Issue](https://github.com/tu-usuario/pruebas-varianza/issues).

## ⭐ Referencias

- Walpole, R. E., et al. (2012). *Probability & Statistics for Engineers & Scientists*
- Montgomery, D. C., & Runger, G. C. (2010). *Applied Statistics and Probability for Engineers*
- Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for normality

---

⭐ Si este proyecto te fue útil, ¡dale una estrella en GitHub!
