# Datensatz für überwachtes Lernen

Der erste M5-Datensatz verwendet vier Zahlenmerkmale pro Entscheidung und eine
zukünftige Bruttokursrendite als Regressionsziel. Die Python-Schnittstelle
`build_training_data(dataset, config)` arbeitet mit einer normalisierten Serie
und liefert unveränderliche Zeilen in drei chronologischen Abschnitten.

## Merkmale und Ziel

Die Entscheidung fällt zum Schluss der letzten Kerze im Lookback. Jede Kerze
des Lookbacks muss bis dahin gemäß `available_at` verfügbar sein.

| Merkmal | Berechnung |
| --- | --- |
| `return_1` | Änderung des Schlusskurses gegenüber dem vorherigen Schlusskurs, geteilt durch den vorherigen Schlusskurs. |
| `window_return` | Änderung vom ersten zum letzten Schlusskurs im Lookback, geteilt durch den ersten Schlusskurs. |
| `range_fraction` | Hoch minus Tief der Entscheidungskerze, geteilt durch ihren Schlusskurs. |
| `close_vs_mean` | Abweichung des letzten Schlusskurses vom Mittel der Schlusskurse im Lookback, geteilt durch das Mittel. |

Der Referenzeinstieg ist das Open der unmittelbar folgenden Kerze. Der
Referenzausstieg ist das Close der letzten Kerze im eingestellten Zielhorizont,
gezählt ab der Einstiegskerze. `target_return` ist die Kursdifferenz geteilt
durch den Einstiegskurs. Die Berechnung verwendet Dezimalzahlen mit 28
signifikanten Stellen und Rundung zur geraden Ziffer bei Gleichstand. Der
Zielwert enthält keine Ausführungskosten. `target_available_at` ist die
späteste Verfügbarkeit der Kerzen im Zielhorizont.

Das vollständige Fenster aus Lookback und Zielhorizont muss innerhalb einer
expliziten Sitzung liegen. Für den kontinuierlichen Kalender gilt die
durchgehende Serie. Aktive Datenlücken und nichtpositive Open-/Close-Kurse
führen zu einem Fehler. Unvollständige Fenster an den Sitzungs- oder
Datensatzrändern entfallen.

## Zeitliche Abschnitte

`validation_start` und `test_start` sind aufsteigende UTC-Grenzen. Der
Entscheidungszeitpunkt ordnet eine Zeile dem Training vor der ersten Grenze,
der Validierung ab der ersten bis vor der zweiten Grenze oder dem Test ab
der zweiten Grenze zu. Trainings- und Validierungszeilen entfallen, wenn ihr
Zielwert erst an oder nach der nächsten Grenze verfügbar wird. Verfügbare
historische Kerzen dürfen für ein späteres Merkmal verwendet werden, sofern
das Fenster innerhalb der Sitzung liegt. Jeder Abschnitt muss Beispiele
enthalten.

Der Export enthält die berechneten Merkmale ohne gelernte Skalierung. Beim
Modelltraining werden Modell und Skalierung nur auf dem Trainingsabschnitt
angepasst. Die Validierung dient dem Modellvergleich. Der Test bleibt bis zur
abschließenden Auswertung des eingefrorenen Modells reserviert.

## Lokaler Export

```toml
version = 1
dataset_artifact = "dataset.json"
lookback_bars = 8
horizon_bars = 2
validation_start = 2026-01-22T08:00:00Z
test_start = 2026-01-27T08:00:00Z
```

Der Eingabepfad ist relativ zur TOML-Datei oder absolut. Der Lookback umfasst
mindestens zwei Kerzen und der Zielhorizont mindestens eine Kerze.

```sh
market-trader data prepare-training --config prepare.toml --output training-data.json
```

Die CLI verifiziert das Eingabeartefakt und legt eine neue Ausgabedatei an.
Das JSON-Format `market-trader.training-data.v1` enthält die Kennungen des
Eingabedatensatzes und seines Artefakts, Instrument, Intervall in Mikrosekunden,
Konfiguration, geordnete Merkmalsnamen, Zieldefinition und die drei Abschnitte.
Die Zeilen speichern Zeitstempel als ISO-8601-Zeichenketten und Dezimalwerte
als Zeichenketten. `training_data_id` ist der SHA-256-Wert des kompakten
UTF-8-JSON mit sortierten Schlüsseln, ohne ASCII-Escaping, ohne das Feld
`training_data_id` und ohne abschließenden Zeilenumbruch.

## CPU-Basismodell

```sh
market-trader model train-baseline --training-data training-data.json --output ridge-validation.json
```

Das Modell zentriert die vier Merkmale mit ihrem Mittelwert im Training
und teilt sie durch die jeweilige Populationsstandardabweichung. Bei
konstanten Merkmalen ist der Divisor eins. Das Modell minimiert den
mittleren quadratischen Fehler plus `0.1 * sum(coefficients²)`. Der
Achsenabschnitt bleibt unbestraft. `alpha = 0.1` ist vorab festgelegt.

Die Validierung vergleicht den mittleren absoluten Fehler (MAE) und die Wurzel
des mittleren quadratischen Fehlers (RMSE) mit einer konstanten Vorhersage von
null Rendite. Beide Kennzahlen stehen im Bericht als Renditeanteil, wobei
`0.0001` einem Basispunkt entspricht. Der Befehl prüft die Prüfsumme des
gesamten Eingabeartefakts und verwendet für Fit und Auswertung ausschließlich
die zugehörigen Trainings- und Validierungszeilen.

Der Bericht `market-trader.ridge-validation.v1` enthält die Herkunft,
Zeitgrenzen, Merkmalsreihenfolge, Skalierung, Koeffizienten, Achsenabschnitt
und Validierungskennzahlen. `report_id` folgt denselben SHA-256-Regeln wie
`training_data_id` und lässt das Feld `report_id` aus. Der Befehl legt eine
neue Datei an. Das gespeicherte Modell sagt für einen Merkmalsvektor
`intercept + sum(coefficients[i] * (features[i] - means[i]) / scales[i])` voraus.

## Auswertung auf dem Testabschnitt

```sh
market-trader model evaluate-baseline \
  --model ridge-validation.json \
  --training-data training-data.json \
  --output ridge-test.json
```

Die Auswertung prüft die Prüfsummen, die Datensatzkennung und die
Modellmetadaten. Sie verwendet die gespeicherte Skalierung und die
gespeicherten Koeffizienten für Entscheidungen ab `test_start`. Der Bericht
`market-trader.ridge-test.v1` enthält die Modellbericht-Kennung, die
Datensatzkennungen, Anzahl und Zeitraum der Testbeispiele sowie MAE und RMSE
für das Ridge-Modell und die Nullvorhersage. Der Befehl legt eine neue Datei an.
