# Contrato entre la app y el modelo de IA

Estado: propuesta de contrato v1 para el primer prototipo del clasificador.
Todavía no se ha entrenado ni integrado un modelo en este repositorio.
Este documento fija el significado de las etiquetas y el orden de imagen; los
detalles concretos de los tensores acompañarán al primer modelo exportado.

## Alcance y responsabilidades

El modelo clasifica casillas individuales de un tablero digital plano o un
diagrama impreso. La app se encarga de localizar el tablero, corregir la perspectiva
y extraer las casillas. El proyecto de IA también debe proporcionar una herramienta
de línea de comandos de referencia que reciba un tablero 8×8 ya recortado y
rectificado, extraiga sus casillas y devuelva predicciones en orden de imagen.
Es una herramienta de desarrollo, no un servidor que la app necesite.

Las imágenes iniciales de referencia contienen solo el tablero, sin coordenadas
exteriores ni márgenes. Definir cualquier recorte adicional por casilla en los
metadatos y ejemplos de prueba. No rotar, reflejar ni inferir coordenadas de ajedrez
silenciosamente a partir de la entrada.

## Orden de clases

El entrenamiento, las salidas exportadas y la app comparten este orden exacto,
con índices que empiezan en cero:

| Índice | Etiqueta | Símbolo FEN |
| --- | --- | --- |
| 0 | empty | ninguno (se codifica como una secuencia de casillas vacías en FEN) |
| 1 | white_pawn | P |
| 2 | white_knight | N |
| 3 | white_bishop | B |
| 4 | white_rook | R |
| 5 | white_queen | Q |
| 6 | white_king | K |
| 7 | black_pawn | p |
| 8 | black_knight | n |
| 9 | black_bishop | b |
| 10 | black_rook | r |
| 11 | black_queen | q |
| 12 | black_king | k |

El color representa el bando de la pieza, independientemente de su diseño o del
fondo de la casilla. Un resultado incierto es un estado de revisión derivado de
las puntuaciones, no una clase adicional. Una confianza baja nunca debe convertirse
en una casilla vacía.

## Orden de casillas y salidas

- Filas y columnas empiezan en cero y usan coordenadas de imagen.
- Índice = fila × 8 + columna; el índice 0 es la esquina superior izquierda
  y el 63 la inferior derecha.
- No es necesariamente el orden de a8 a h1: la orientación corresponde a la app.
- El resultado lógico contiene 64 vectores de 13 probabilidades de clase y la
  versión del modelo.
- La herramienta de referencia devuelve un objeto JSON con `model_version` y
  `squares`. Cada una de sus exactamente 64 entradas ordenadas contiene `row`,
  `column` y `probabilities`.
- Las probabilidades deben ser finitas, estar en [0, 1] y sumar uno dentro de una
  tolerancia numérica documentada. La puntuación más alta no es una garantía
  calibrada de acierto.
- Los adaptadores de ejecución descuantizan las salidas y aplican softmax solo si
  el archivo declara logits. No deben aplicar softmax dos veces.
- El tensor físico puede aceptar una casilla o un lote. Documentar las formas
  admitidas; la app puede procesar las 64 casillas en lotes o llamadas sucesivas.

La implementación de referencia debe especificar cómo dividir tableros cuyas
dimensiones no sean divisibles entre ocho, cómo tratar la orientación de imagen
y qué convenciones de redimensionado utiliza. Proporcionar ejemplos de prueba
para que Dart y Python puedan reproducir el mismo preprocesamiento.

## Paquete de entrega versionado

Contenido propuesto, que debe crear el proyecto de IA:

```text
export/<version-del-modelo>/
  model.tflite
  labels.json
  metadata.json
  evaluation.json
  MODEL_CARD.md
  fixtures/
```

`labels.json` es un array con las 13 cadenas en el orden anterior. Los metadatos
deben incluir la versión del contrato, la del modelo y la suma de comprobación
del archivo, además de:

- Nombres, formas, disposición y tipos de los tensores de entrada/salida;
  tamaños de lote admitidos.
- Dimensiones de casilla, orden de canales RGB/escala de grises, composición del
  canal alfa, interpolación de redimensionado, política de recorte y fórmula de
  normalización de entrada.
- Si la normalización está integrada en el modelo o debe aplicarla quien lo llama.
- Escalas, puntos cero y ejes de cuantización, cuando corresponda.
- Si las salidas son logits o probabilidades y cómo convertirlas a los vectores
  de probabilidades definidos arriba.
- Requisitos del modelo y su entorno de ejecución, incluidos operadores no estándar.
- Política de incertidumbre derivada de la validación, con sus limitaciones.

El tamaño de tensor, la arquitectura y la cuantización se decidirán cuando los
experimentos establezcan una base adecuada. La primera exportación puede usar
coma flotante. Cada candidato cuantizado necesita su propia evaluación de
precisión y equivalencia.

## Evidencias para aceptar la entrega

Proporcionar:

1. Comandos reproducibles de preparación de datos, entrenamiento, evaluación y
   exportación; versiones de dependencias, semillas aleatorias e identificadores
   de los conjuntos de datos y sus fuentes.
2. Procedencia, licencias y reglas de agrupación que eviten filtraciones de
   imágenes de origen o estilos entre entrenamiento, validación y prueba.
3. Métricas por clase, matriz de confusión y precisión de tableros completos en
   tableros reservados para prueba, separando capturas y fotos de diagramas,
   así como estilos vistos y no vistos.
4. Mediciones antes y después de exportar/cuantizar, tamaño del archivo y tiempos
   con hardware y entorno de ejecución identificados. La latencia de un ordenador
   no es la de un móvil.
5. Imágenes de referencia con licencia adecuada, tensores de preprocesamiento
   esperados o hashes reproducibles, vectores de probabilidades esperados
   y tolerancias de comparación justificadas.
6. Una ficha del modelo que describa estilos cubiertos, limitaciones, fallos
   y carencias de datos.

Mantener todas las variantes aumentadas y recortes de casillas de un tablero en
la misma partición. Cuando los datos no permitan evaluar estilos no vistos o
diagramas reales, indicar esa carencia en lugar de presentar el rendimiento
sintético como precisión general.

## Interpretación a cargo de la app

La app decide la orientación, destaca dudas y permite edición manual.
Antes de exportar un FEN completo, solicita o muestra explícitamente los valores
asumidos de turno, derechos de enroque, casilla al paso y contadores.
Estos datos no son salidas del clasificador de imágenes.

Las correcciones solo cambian la posición actual. Este contrato no incluye
reentrenamiento en el dispositivo, subida implícita de ejemplos ni un servicio
externo de inferencia.
