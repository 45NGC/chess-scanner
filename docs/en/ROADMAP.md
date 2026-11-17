# Roadmap

Implemented: English/Spanish shell, image import, BoardSnap HTTP recognition,
explicit screenshot/photo mode and orientation, editable position and placement copy.

Next:
1. Test real images on Android, iOS and web against the configured backend.
2. Deploy BoardSnap with HTTPS, CORS, access controls and capacity limits.
3. Add camera capture and the full board editor/FEN metadata workflow.
4. Add history, export/analysis integration and UX improvements as needed.
5. Revisit offline inference and printed diagrams as separate future work.

Photographic recognition remains experimental. Compatibility depends on the
server model and measured conditions; adding the Flutter client does not improve
recognition accuracy. Do not claim native mobile or unseen-style coverage without
verification. Keep corrections local unless the user explicitly requests upload.
