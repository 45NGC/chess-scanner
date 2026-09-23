# Roadmap: reconocimiento local de ajedrez

## Decisión de producto

Reconocer tableros digitales y diagramas planos, tanto desde imágenes importadas
como desde la cámara del móvil. El reconocimiento se ejecutará en el dispositivo
con un modelo entrenado en un proyecto independiente. La interfaz será bilingüe,
en español e inglés.

La experiencia de cámara será similar a leer un QR: encuadrar, esperar una lectura
estable, recibir una confirmación y revisar la posición. Un diagrama no tiene la
corrección de errores de un QR; siempre debe ser posible corregir el resultado.

Esta decisión sustituye el aprendizaje inicial por plantillas. El usuario no
tendrá que etiquetar piezas de referencia para empezar. Las correcciones cambian
la posición actual; no reentrenan el modelo automáticamente.

## Estado actual

Implementado: estructura Flutter para Android, iOS y web, pantalla inicial,
localización y preferencia de idioma persistente.

Pendiente: importación, recorte, detección, modelo, inferencia, cámara, editor de
posición y exportación FEN. Los botones de la pantalla inicial están desactivados
y se indica que estas funciones están en desarrollo.

## Alcance

- Capturas de tableros digitales, imágenes de diagramas y fotos de diagramas
  impresos o mostrados en una pantalla.
- Inferencia local, sin cuenta, sin servidor y sin subir imágenes.
- Clasificación en 13 clases: casilla vacía y seis piezas de cada color.
- Revisión de casillas dudosas, edición de cualquier casilla y orientación.
- Exportación FEN con metadatos confirmados por el usuario.

Quedan fuera los tableros con piezas físicas tridimensionales, un motor de ajedrez
propio, OCR general, entrenamiento en el móvil y reconocimiento universal
garantizado. Historial y apertura de análisis externo serán mejoras posteriores.

## Reparto entre proyectos

El proyecto de IA prepara los datos, entrena, evalúa y entrega un modelo versionado,
su preprocesamiento, etiquetas y ejemplos de referencia. Esta app gestiona la
imagen/cámara, geometría del tablero, integración local, revisión y FEN.

El punto de integración se describe en [MODEL_CONTRACT.md](MODEL_CONTRACT.md).

## Fase 1: clasificador independiente

1. Crear un conjunto reproducible de casillas y tableros etiquetados.
2. Combinar diseños digitales variados con diagramas reales.
3. Separar entrenamiento, validación y prueba por origen y estilo visual; los
   recortes o variantes de una misma imagen no pueden cruzar las particiones.
4. Entrenar un clasificador compacto y evaluar estilos no vistos.
5. Exportar el modelo móvil candidato, metadatos y ejemplos de paridad.

Criterio de salida: resultados reproducibles por clase y por tablero completo en
un conjunto de prueba independiente, limitaciones documentadas y artefacto que
pueda probarse en un dispositivo. No basta la precisión del entrenamiento.

## Fase 2: primer flujo útil en móvil

1. Importar una captura o foto de un diagrama.
2. Permitir recorte/ajuste manual de las esquinas y normalizar el tablero.
3. Extraer 64 casillas y aplicar el preprocesamiento del modelo.
4. Ejecutar inferencia local con el artefacto exportado.
5. Mostrar la posición, destacar dudas y permitir corregir cualquier casilla.
6. Confirmar orientación y metadatos; copiar FEN.
7. Medir latencia y memoria en un móvil real y comprobar el funcionamiento sin red.

Criterio de salida: imagen importada -> revisión -> FEN en modo avión, sin ejemplos
previos aportados por el usuario. Este hito precede al escaneo automático.

## Fase 3: detección y perspectiva

1. Localizar el contorno del tablero y sus cuatro esquinas.
2. Corregir perspectiva y tratar coordenadas/márgenes fuera de las casillas.
3. Rechazar tableros incompletos o imágenes demasiado borrosas.
4. Mantener ajuste manual cuando falle la detección.

Criterio de salida: evaluar localización y reconocimiento conjuntamente en capturas
y fotos de diagramas con distintas condiciones, incluyendo fallos de detección.

## Fase 4: cámara y captura automática

1. Añadir permisos, vista previa y guía de encuadre.
2. Procesar fotogramas seleccionados sin bloquear la interfaz ni acumular trabajo.
3. Comprobar nitidez, encuadre y estabilidad de las esquinas.
4. Exigir acuerdo entre lecturas consecutivas y confianza suficiente.
5. Capturar una vez, detener el procesamiento y dar confirmación visual/háptica.
6. Permitir captura manual, revisión y reintento cuando no haya acuerdo.
7. Liberar cámara y recursos de inferencia al salir o pasar a segundo plano.

Los umbrales y la frecuencia de muestreo se elegirán con medidas; repetir un error
en varios fotogramas no lo convierte en un resultado correcto.

Criterio de salida: captura estable con tasa de falsos positivos, tiempo hasta
captura, memoria y consumo medidos en dispositivos representativos.

## Fase 5: plataformas y cobertura

- Validar Android e iOS con los mismos ejemplos de paridad.
- Integrar un runtime local para web y comprobar sus límites de rendimiento.
- Definir caché de app/modelo para uso web sin red tras la primera carga;
  no asumir que inferencia local equivale a disponibilidad offline del sitio.
- Ampliar pruebas a diagramas con sombras, reflejos, moiré, compresión y anotaciones.
- Entregar versiones nuevas del modelo con la app y comprobar compatibilidad.

La primera plataforma móvil se elegirá según los dispositivos de prueba
disponibles. No se da por hecha la compatibilidad de un paquete Flutter con todos
los destinos ni se sustituye la inferencia local por una API remota.

## Orientación, reglas y FEN

El clasificador devuelve casillas en orden de imagen. La app debe confirmar cómo
se corresponden con las coordenadas de ajedrez, permitiendo rotación manual.

Una imagen no determina el turno, los derechos de enroque, la casilla al paso ni
los contadores. Deben solicitarse o mostrarse como valores asumidos editables,
nunca presentarse como datos reconocidos. Las comprobaciones de coherencia
ayudan a detectar posibles errores; no deben cambiar piezas silenciosamente ni
inventar información histórica.

## Calidad y riesgos

Medir precisión por clase, porcentaje de tableros completos correctos antes de
corregir, número de correcciones, fallos de localización y latencia extremo a
extremo. Separar resultados de capturas digitales y fotos de diagramas, así como
estilos conocidos y no vistos. Calibrar el tratamiento de las puntuaciones bajas.

Riesgos principales: diseños nuevos, casillas resaltadas, flechas, coordenadas,
diagramas degradados, reflejos y lectura estable pero equivocada. El conjunto de
prueba debe incluir estos casos, no solo imágenes sintéticas limpias.

Las imágenes y correcciones permanecen locales. Cualquier futura función para
compartir ejemplos de entrenamiento requerirá una acción explícita del usuario;
no forma parte del flujo inicial.
