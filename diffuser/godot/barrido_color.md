# Barrido de color sobre V0: ¿pendiente o acantilado?

## La pregunta

En [perturbaciones.md](perturbaciones.md) se ve que pintar la pieza de rojo hunde
a V0, de `0,99` a `0,14`. Eso admite dos lecturas incompatibles:

- **Pendiente.** El color se aleja poco a poco de lo que la política vio en el
  entrenamiento y la política se degrada poco a poco. Sería un problema de
  distancia en el espacio de color.
- **Acantilado.** Hay una frontera, y al cruzarla la pieza deja de ser una pieza.
  La política estaría *clasificando* por tono.

Distinguirlas cambia lo que hay que hacer para arreglarlo, así que se midió.

**Esto sigue sin ser una medición reportable**: un episodio por celda, tres
semillas, solo V0. Las mismas advertencias que en la bitácora principal valen
aquí enteras.

## El montaje

Interpolación lineal en RGB de 8 bits entre el gris azulado del entrenamiento
(`LightSlateGray`, `#778899`) y el `firebrick` de la perturbación `t_roja`
(`#b22222`), en siete puntos. Lineal en RGB y no en espacio de tono porque **RGB
es literalmente lo que entra a la red**. El relleno se deriva del borde con el
mismo aclarado del renderizador original, así que a lo largo del barrido lo único
que varía es el color de la pieza.

Semillas 200023, 200024 y 200051, elegidas con `servidor/elegir_semillas.py`.
Condición B, V0, 27 episodios, 46 minutos.

### Los dos controles, y por qué hacen falta

El eje principal **confunde dos variables**. El firebrick no solo es más rojo: es
la mitad de luminoso que el gris de partida (64,6 frente a 133,6 en Rec. 709). Sin
controles, una caída gradual no distinguiría «se aleja el tono» de «se oscurece».

- **`gris oscuro`, `#414141`.** Gris neutro con la luminancia del firebrick. Nada
  de rojo. Si aquí también falla, parte del efecto es brillo.
- **`rojo isolum`, `#ce7272`.** Rojo desaturado con la luminancia del gris
  original. Si aquí falla igual, el efecto es tono.

```powershell
.venv_diffuser_infer\Scripts\python.exe diffuser\godot\servidor\servidor_politica.py `
    --variante v0 --obs godot --puerto 5563
.venv_diffuser_infer\Scripts\python.exe diffuser\godot\servidor\barrido_color.py `
    --puerto 5563 --semillas 200023 200024 200051
```

## Resultados

| tono | hex | distancia RGB | luminancia | 200023 | 200024 | 200051 | media |
|---|---|---|---|---|---|---|---|
| mezcla 0,00 | `#778899` | 0,0 | 133,6 | 0,9994 | 0,9793 | 1,0000 | **0,9929** |
| mezcla 0,17 | `#817785` | 28,1 | 122,1 | 0,9920 | 0,9370 | 1,0000 | **0,9763** |
| mezcla 0,33 | `#8b6671` | 56,2 | 110,7 | 0,5438 | 0,2551 | 0,9204 | **0,5731** |
| mezcla 0,50 | `#94555e` | 83,2 | 99,0 | 0,9104 | 0,1857 | 0,8104 | **0,6355** |
| mezcla 0,67 | `#9e444a` | 111,3 | 87,6 | 0,5609 | 0,1539 | 0,6799 | **0,4649** |
| mezcla 0,83 | `#a83336` | 139,4 | 76,1 | 0,7749 | 0,1052 | 0,0000 | **0,2934** |
| mezcla 1,00 | `#b22222` | 167,5 | 64,6 | 0,4063 | 0,0000 | 0,0000 | **0,1354** |
| *control* gris oscuro | `#414141` | 125,3 | 65,0 | 0,4056 | 0,0946 | 0,9092 | **0,4698** |
| *control* rojo isolum | `#ce7272` | 97,8 | 133,6 | 0,4590 | 0,4491 | 0,8458 | **0,5846** |

## Qué se ve

### 1. Es una pendiente, no un acantilado

La media cae de forma continua a lo largo del eje y la correlación con la
distancia RGB es **−0,97**. No hay ningún salto que separe «pieza» de «no pieza».
Dos de las tres semillas son monótonas paso a paso (6 de 6 descensos); la tercera,
200023, es ruidosa y no ordena (4 de 6), lo que da la medida del ruido que hay
detrás de cada celda.

Hay, eso sí, un **codo temprano**. Entre 28 y 56 de distancia la media pasa de
`0,976` a `0,573`: la mitad de la caída total ocurre en el primer tercio del eje.
Por debajo de ~30 la política está intacta; a partir de ~50 ya está tocada. Esa
tolerancia de unas 30 unidades RGB es estrecha, y es la cifra concreta que sale de
todo esto.

### 2. Lo que importa es la distancia, no el rojo

Este es el resultado que no esperaba, y lo dan los controles:

| control | distancia RGB | observado | lo que predice el eje a esa distancia | diferencia |
|---|---|---|---|---|
| gris oscuro `#414141` | 125,3 | 0,4698 | 0,3793 | +0,09 |
| rojo isolum `#ce7272` | 97,8 | 0,5846 | 0,5465 | +0,04 |

**Los dos caen sobre la misma curva.** Un gris neutro, sin una gota de rojo, a
distancia 125 degrada tanto como el color rojizo que está a esa misma distancia. Y
un rojo con el brillo original degrada lo que le toca por su distancia, ni más ni
menos. Las dos diferencias son menores que la dispersión entre semillas de una
misma fila.

Es decir: **la política no ha aprendido «la pieza es gris azulada» como
categoría**. Tiene rasgos sintonizados en un punto concreto del espacio de color
que responden cada vez menos según te alejas, en la dirección que sea. El fallo no
es de clasificación, es de sintonía.

### 3. El fallo no es ceguera total

Ni siquiera con el firebrick la política se queda quieta: en 200023 conserva
`0,41`. Empuja, y a veces acierta. Lo que pierde es fiabilidad, de forma creciente
con la distancia.

## Qué implica

Si fuera un acantilado, la solución natural sería enseñarle más categorías de
color. Siendo una pendiente en distancia, el diagnóstico es otro y más simple:
**el codificador nunca vio variación fotométrica**. El único aumento de datos de
V0 es el recorte aleatorio a 76 píxeles, que es geométrico. Un `ColorJitter` de
manual durante el entrenamiento atacaría exactamente esto, y es barato: no cambia
la arquitectura, ni el presupuesto de épocas, ni el protocolo.

Eso conecta con la línea de trabajo futuro sobre ablaciones de la memoria, y sería
un experimento bien planteado: V0 con aumento fotométrico frente a V0 tal cual,
las dos evaluadas sobre el bloque disjunto y sobre este mismo barrido. Nada de eso
está hecho aquí, y este documento no lo sustituye.

## Lo que este barrido no dice

- **Solo V0.** V3 no tiene línea base suficiente en Godot para barrerlo.
- **Un episodio por celda.** La fila de 200023 (`0,99 · 0,54 · 0,91 · 0,56 · 0,77 ·
  0,41`) enseña que la varianza entre repeticiones es del mismo orden que buena
  parte de las diferencias entre filas.
- **Un solo control por eje.** Que los dos caigan sobre la curva es sugerente, no
  concluyente: harían falta varios puntos isoluminantes y varios grises.
- **La observación cambia por Godot antes de que empiece el barrido.** El punto de
  mezcla 0,00 ya lleva encima el 2,7 % de píxeles que separan el renderizador de
  Godot del de pygame.
