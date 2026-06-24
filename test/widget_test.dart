import 'package:flutter_test/flutter_test.dart';

import 'package:chess_scanner/app/app.dart';

void main() {
  testWidgets('renders localized home shell', (WidgetTester tester) async {
    await tester.pumpWidget(const ChessScannerApp());
    await tester.pumpAndSettle();

    expect(find.text('Chess Scanner'), findsOneWidget);
    expect(find.text('Choose image'), findsOneWidget);
  });
}
