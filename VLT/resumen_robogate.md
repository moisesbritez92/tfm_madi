# Resumen del Paper: ROBOGATE (Azuki Kim, 2026)

**Título:** ROBOGATE: Adaptive Failure Discovery for Safe Robot Policy Deployment via Two-Stage Boundary-Focused Sampling  
**Autor:** Azuki Kim (AgentAI Co., Ltd.)  
**Fecha:** Marzo 2026  

## 1. Problema que Resuelve
El despliegue de políticas robóticas aprendidas (como Imitation Learning o modelos fundacionales VLA) en entornos industriales requiere una validación exhaustiva de seguridad antes de su puesta en producción. Sin embargo, evaluar el inmenso espacio de parámetros físicos y geométricos posibles es computacionalmente intratable con muestreo aleatorio. 
La evaluación actual a menudo ocurre en una única morfología robótica o bajo condiciones nominales, ignorando las "zonas límite" (transition zones) donde ocurren los fallos críticos.

## 2. Contribución Principal: ROBOGATE
ROBOGATE es un framework open-source de gestión de riesgo para despliegue que utiliza simulación basada en físicas (NVIDIA Isaac Sim) y una **estrategia de muestreo adaptativo en dos etapas** para descubrir de manera eficiente las fronteras de fallo de una política robótica.

### 2.1 El Muestreo Adaptativo en Dos Etapas
1. **Etapa 1 (Exploración Uniforme):** Se realizan 20,000 simulaciones utilizando Latin Hypercube Sampling (LHS) a lo largo de un espacio de 8 dimensiones continuas y discretas (fricción, masa, tamaño del objeto, ruido de cinemática inversa, número de obstáculos, etc.). Esto crea un mapa general de éxitos y fallos.
2. **Etapa 2 (Muestreo Enfocado en la Frontera):** A partir de los resultados de la Etapa 1, se identifica la zona de transición donde la tasa de éxito (Success Rate, SR) está entre el 30% y el 70%. Se lanzan otras 10,000 simulaciones concentradas exclusivamente en esta zona fronteriza.

Este método mejoró la exhaustividad del modelo de predicción de riesgo (pasando de un AUC de 0.754 en la etapa 1 a 0.780 en total).

### 2.2 Evaluación Multi-Embodiment
ROBOGATE diferencia fallos propios de la política de las limitaciones del hardware al evaluar simultáneamente en dos robots diferentes:
- **Franka Panda** (7-DOF con pinza paralela). 
- **UR5e** (6-DOF con pinza de succión).

El framework identificó **cuatro zonas de peligro universales** (ej. objetos de masa >0.935 kg) donde el éxito cae por debajo del 40% sin importar qué robot se esté usando, indicando que la dificultad radica en la tarea/física y no en el hardware.

## 3. Modelo de Riesgo e Interpretabilidad
En lugar de usar modelos de caja negra (como redes neuronales o random forests), ROBOGATE ajusta un modelo de **Regresión Logística**. Esto tiene una ventaja industrial enorme:
- Extrae una **ecuación de frontera en formato cerrado** para predecir el fallo respecto a variables críticas (ej. ecuación de fricción vs. masa).
- Permite traducir estos resultados directamente a **restricciones operativas fáciles de interpretar** por operadores humanos (ej. "no desplegar si la fricción del objeto es < 0.49").

## 4. Hallazgos sobre el despliegue de Modelos VLA
Quizás la aportación más valiosa del framework se evidencia en su evaluación del modelo fundacional **Open-Source VLA Octo-Small**:
- Sometido a los 68 escenarios adversarios generados por ROBOGATE (iluminación baja, texturas transparentes, oclusiones, posiciones atípicas), Octo-Small obtuvo un **0.0% de éxito**.
- Sus fallos principales fueron: no agarrar el objetivo (79.4%) y colisiones directas (20.6%).
- Incluso en las configuraciones "nominales" (las más fáciles), falló al 100%.

**Conclusión crítica:** Existe una brecha masiva (*Sim-to-Real perception gap* y discordancia en el espacio de acciones) entre las demostraciones controladas de los VLAs generalistas en laboratorio y la robustez requerida para un despliegue industrial. Las políticas generalistas actuales como Octo-Small no superan un "deployment gate" industrial automatizado.