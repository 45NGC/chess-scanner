# Hoja de ruta

Implementado: interfaz inglés/español, fotos con cámara Android/iOS, importación, reconocimiento HTTP BoardSnap,
modo captura/foto y orientación explícitos, revisión editable y copia de colocación.

Siguientes pasos:
1. Probar imágenes reales en Android, iOS y web contra el servidor configurado.
2. Desplegar BoardSnap con HTTPS, CORS, acceso y capacidad configurados.
3. Completar editor/metadatos para FEN completo e integrar análisis con Stockfish.
4. Añadir historial, exportación/análisis y mejoras de uso según necesidades.
5. Revisar reconocimiento sin conexión y diagramas impresos como trabajo futuro.

El reconocimiento fotográfico sigue siendo experimental. La compatibilidad depende
del modelo del servidor y sus condiciones medidas; integrar Flutter no mejora su
precisión. No declarar estilos o plataformas compatibles sin probarlos. Las
correcciones permanecen locales salvo petición explícita del usuario.
