/// Canonical a8-to-h1 square order. Empty squares use an empty string.
List<String> decodePiecePlacement(String placement) {
    final ranks = placement.split('/');
    if (ranks.length != 8) throw const FormatException('Expected eight ranks.');
    final squares = <String>[];
    for (final rank in ranks) {
        final row = <String>[];
        for (final symbol in rank.split('')) {
            if (RegExp(r'^[1-8]$').hasMatch(symbol)) {
                row.addAll(List.filled(int.parse(symbol), ''));
            }
            else if ('PNBRQKpnbrqk'.contains(symbol)) {
                row.add(symbol);
            }
            else {
                throw const FormatException('Invalid piece placement.');
            }
        }
        if (row.length != 8) throw const FormatException('Expected eight files.');
        squares.addAll(row);
    }
    return squares;
}

String encodePiecePlacement(List<String> squares) {
    if (squares.length != 64 ||
        squares.any(
            (s) => s.isNotEmpty && (s.length != 1 || !'PNBRQKpnbrqk'.contains(s)),
        )) {
        throw const FormatException('Expected 64 valid squares.');
    }
    return List.generate(8, (row) {
        final result = StringBuffer();
        var empty = 0;
        for (final piece in squares.skip(row * 8).take(8)) {
            if (piece.isEmpty) {
                empty++;
            }
            else {
                if (empty > 0) result.write(empty);
                empty = 0;
                result.write(piece);
            }
        }
        if (empty > 0) result.write(empty);
        return result.toString();
    }
    ).join('/');
}
