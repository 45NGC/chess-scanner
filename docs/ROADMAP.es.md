# ROADMAP

## Vision

`chess_scanner` sera una aplicacion Flutter para movil y web que:

- recibe una captura o imagen de un tablero digital o un diagrama de libro
- detecta el tablero
- separa las 64 casillas
- intenta reconocer automaticamente las piezas
- pregunta al usuario solo en los casos dudosos
- aprende de esas correcciones para futuros escaneos
- funciona en espanol e ingles

## Alcance de la V1

Incluido:

- tableros digitales de sitios como lichess y chess.com
- diagramas de libros de ajedrez
- aprendizaje incremental por plantillas
- correccion manual asistida
- exportacion de FEN
- apertura de la posicion en un tablero de analisis
- interfaz bilingue `es/en`

No incluido en esta fase:

- tableros fisicos fotografiados
- reconocimiento universal sin entrenamiento previo
- motor de ajedrez propio
- OCR avanzado de texto ajeno al tablero

## Idea Base

El sistema no intentara saber todo desde el primer dia.

Funcionara asi:

1. detecta el tablero
2. extrae las casillas
3. compara cada casilla con plantillas ya aprendidas
4. asigna una confianza a cada resultado
5. pregunta al usuario solo cuando la confianza sea baja
6. guarda las respuestas para mejorar futuros escaneos

## Principios del Proyecto

- primero utilidad real, luego sofisticacion
- minimizar preguntas al usuario
- aprender por estilo visual
- separar bien deteccion, reconocimiento y aprendizaje
- guardar conocimiento persistente, no solo en memoria temporal
- internacionalizacion desde el inicio

## Arquitectura Propuesta

### Frontend

Flutter con soporte:

- Android
- iOS
- Web

Idiomas:

- Espanol
- Ingles

Pantallas iniciales:

- Home
- Importar imagen
- Resultado del escaneo
- Correccion manual
- Historial
- Ajustes

### Backend local de la app

Capas internas:

1. `board_detector`
2. `board_normalizer`
3. `square_extractor`
4. `square_matcher`
5. `confidence_engine`
6. `interactive_labeling`
7. `template_store`
8. `fen_builder`

## Modelo de Aprendizaje

El sistema aprendera por plantillas confirmadas por el usuario.

Cada plantilla deberia guardar:

- tipo de pieza
- color de pieza
- color de casilla
- estilo visual
- imagen normalizada de la casilla
- metadatos del origen

Ejemplos de estilos:

- `lichess_blue`
- `lichess_brown`
- `chesscom_green`
- `book_diagram_serif_01`

## Flujo Ideal

1. el usuario carga una imagen
2. la app detecta el tablero
3. la app lo normaliza
4. la app separa las 64 casillas
5. la app agrupa casillas visualmente parecidas
6. la app intenta reconocerlas con lo ya aprendido
7. la app resuelve automaticamente las casillas claras
8. la app pregunta solo por grupos o casillas dudosas
9. la app reconstruye el FEN
10. la app guarda el nuevo conocimiento

## Estrategia de Preguntas

No preguntar casilla por casilla salvo ultimo recurso.

Prioridad:

1. intentar reconocer automaticamente
2. agrupar casillas similares
3. preguntar por grupo
4. propagar la respuesta al grupo completo
5. permitir corregir casillas individuales

Ejemplo:

- en vez de preguntar 8 veces por peones blancos
- preguntar una vez por un grupo de casillas que parecen iguales

## Fase 1: Base del Proyecto

Objetivo: tener la app arrancada y el flujo principal definido.

1. crear repo nuevo `chess_scanner`
2. crear app Flutter con soporte movil y web
3. definir estructura del proyecto
4. crear navegacion basica
5. crear flujo de importar imagen
6. crear vista de resultado vacia
7. configurar localizacion `es/en`
8. definir modelo de datos para:
   - tablero
   - casilla
   - plantilla
   - escaneo
   - correccion

## Fase 2: Deteccion del Tablero

Objetivo: encontrar y normalizar correctamente el tablero.

1. detectar region 8x8
2. corregir perspectiva si hace falta
3. normalizar a tamano fijo
4. detectar orientacion
5. soportar:
   - capturas limpias digitales
   - diagramas de libros escaneados
6. guardar imagenes de debug

Criterio de salida:

- el tablero se extrae de forma estable en la mayoria de casos conocidos

## Fase 3: Extraccion de Casillas

Objetivo: obtener 64 casillas consistentes.

1. dividir tablero normalizado en 64 regiones
2. recortar margenes utiles
3. clasificar tono de casilla
4. generar representacion normalizada por casilla
5. almacenar snapshots de debug

Criterio de salida:

- cada casilla queda lista para matching visual

## Fase 4: Matching Basico

Objetivo: reconocer piezas ya aprendidas sin preguntar.

1. implementar matching por similitud
2. distinguir vacia vs ocupada
3. distinguir color de pieza
4. distinguir tipo de pieza
5. calcular confianza por casilla
6. escoger mejor plantilla por perfil visual

Criterio de salida:

- si el estilo ya fue aprendido, la mayoria de casillas se resuelven solas

## Fase 5: Aprendizaje Interactivo

Objetivo: que el sistema mejore con cada uso.

1. guardar respuestas del usuario
2. crear nuevas plantillas a partir de casillas confirmadas
3. evitar duplicados inutiles
4. puntuar calidad de plantilla
5. permitir reentrenamiento ligero del perfil visual

Criterio de salida:

- tras varios escaneos del mismo estilo, bajan claramente las preguntas

## Fase 6: Agrupacion Inteligente

Objetivo: reducir friccion.

1. agrupar casillas similares
2. preguntar por grupos antes que por casillas
3. mostrar sugerencias visuales
4. aplicar respuesta al grupo completo
5. permitir deshacer

Criterio de salida:

- el numero medio de preguntas por escaneo baja mucho

## Fase 7: FEN y Analisis

Objetivo: cerrar el flujo de usuario.

1. construir FEN final
2. mostrar tablero reconstruido
3. copiar FEN
4. abrir tablero de analisis
5. guardar historial de escaneos

## Fase 8: Soporte de Estilos

Objetivo: soportar varios entornos.

1. perfiles para lichess
2. perfiles para chess.com
3. perfiles para diagramas de libros
4. seleccion automatica del perfil mas probable
5. creacion de perfil nuevo si no encaja ninguno

## Fase 9: Persistencia

Objetivo: no perder aprendizaje.

1. almacenar plantillas localmente
2. almacenar perfiles visuales
3. almacenar historial
4. exportar e importar perfiles
5. backup del conocimiento del usuario

## Fase 10: Calidad

Metricas principales:

- precision del FEN final
- precision por casilla
- porcentaje de casillas resueltas sin ayuda
- numero medio de preguntas por escaneo
- tiempo medio por escaneo

Tests minimos:

- tests unitarios de FEN
- tests de matching
- tests con imagenes conocidas
- tests por perfil visual
- tests de localizacion en `es/en`

## Entregas Recomendadas

### V0.1

- app Flutter base
- importar imagen
- pantalla de resultado vacia
- idiomas `es/en`

### V0.2

- deteccion de tablero
- extraccion de 64 casillas
- debug visual

### V0.3

- matching simple con plantillas
- FEN en casos faciles

### V0.4

- preguntas al usuario
- guardado de plantillas

### V0.5

- agrupacion de casillas similares
- menos friccion en correcciones

### V1.0

- soporte util para lichess, chess.com y algunos diagramas de libros
- historial
- exportacion de FEN
- apertura de analisis
- interfaz bilingue estable

## Riesgos

- cambios grandes de tema visual
- highlights de ultima jugada
- flechas y anotaciones sobre el tablero
- coordenadas visibles que contaminen bordes
- diagramas de libros con mala calidad o escaneo torcido

## Decision Tecnica Actual

Para esta version:

- no usar IA pesada de inicio
- usar plantillas + confianza + aprendizaje incremental
- preguntar solo en dudas
- aprender por perfil visual
- mantener toda la UI en espanol e ingles

## Criterio de Exito

El proyecto sera un exito si:

- un usuario puede ensenar un estilo visual en pocos escaneos
- la app reduce progresivamente las preguntas
- el FEN final termina siendo rapido y fiable en los estilos ya aprendidos
- el mismo flujo funciona bien tanto en espanol como en ingles
