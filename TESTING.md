# Ausführbare Tests der Datenbasis

Stand 2026-11-11 · Datenbasis und Offline-Daten-CLI geprüft

## Einrichtung

Die Testbasis ist CPython 3.14 mit pytest und uv. `.python-version` wählt die
Python-Minorversion; `uv.lock` fixiert die Testabhängigkeiten samt Prüfsummen.
Der Kern hat zurzeit keine externen Laufzeitabhängigkeiten. Python und uv sind
Werkzeugvoraussetzungen, kein mitgelieferter Teil des Repositorys.
Geprüft wurde mit CPython 3.14.8, uv 0.13.6 und pytest 9.2.0 auf Linux.

uv nach der [offiziellen Anleitung](https://docs.astral.sh/uv/getting-started/installation/)
installieren. Anschließend vom Repository-Stamm aus:

```sh
uv sync --locked --group test
uv run --locked --group test pytest --collect-only -q
uv run --locked --group test pytest -q --tb=short
```

Die Installation legt eine lokale `.venv` an. Einmalig werden Testpakete und der
festgelegte Build-Backend heruntergeladen; bei fehlendem Python kann uv auch den
Interpreter installieren. Die eigentliche Testsuite benötigt danach weder
Netzwerkzugriff noch API-Schlüssel, GPU, Marktdaten oder Modellgewichte.

Für einen Lauf ohne erneute Auflösung oder Synchronisierung nach Einrichtung:

```sh
.venv/bin/python -m pytest -q --tb=short
```

## Offline-Datenablauf über CLI

Die Befehle laufen nach `uv sync --locked --group test` vom Repository-Stamm aus.

```sh
.venv/bin/market-trader data import --config configs/data-example.toml --output datasets/example.json
.venv/bin/market-trader data verify --artifact datasets/example.json
```

Die [Beispielkonfiguration](configs/data-example.toml) verwendet eine synthetische
CSV. TOML-Version 1 definiert eine kontinuierliche Instrument-/Intervallserie mit
explizitem UTC-Zeitraum und der Lückenregel `report` oder `reject`. Der CSV-Pfad
ist absolut oder relativ zur Konfigurationsdatei. Pfade in Befehlsargumenten
beziehen sich auf das aktuelle Arbeitsverzeichnis.

Der Import erfasst Größe und SHA-256 der eingelesenen CSV, normalisiert die
Kerzen und prüft den gespeicherten Datensatz vor der Veröffentlichung. Für einen
erneuten Import einen neuen Zielpfad angeben. Vorhandene Ziele werden abgewiesen.
`data verify` prüft das gespeicherte Artefakt anhand seines Formats, seiner
Prüfsumme und der fachlichen Datenverträge. Beide Befehle geben bei Erfolg eine
JSON-Zusammenfassung mit Kennungen, Kerzenzahl und Anzahl fehlender Kerzen aus.
Bei Verarbeitungsfehlern steht der betroffene Schritt auf stderr und der Exitcode
ist 1. Fehler in den Befehlsargumenten führen zu Exitcode 2.

## Offline-Backtest über CLI

Der Beispielbefehl läuft nach der Einrichtung vom Repository-Stamm aus.

```sh
.venv/bin/market-trader backtest --config configs/backtest-example.toml
.venv/bin/market-trader backtest --config configs/backtest-sma-example.toml
.venv/bin/market-trader backtest --config configs/backtest-pullback-example.toml
```

Die ersten beiden TOML-Konfigurationen verwenden dieselben Daten und dasselbe
Ausführungsprofil. Der Abschnitt `[strategy]` wählt `single_long` oder
`sma_crossover` und legt dessen Parameter fest. Der Bericht erscheint
auf stdout. Bei Verarbeitungsfehlern meldet der Befehl den betroffenen Schritt
auf stderr und endet mit Exitcode 1. Der Beispiellauf kauft 3 Einheiten zu
100,2 Q, schließt sie zu 104,8 Q und endet nach Gebühren bei 1.013,185 Q Equity.
Beim SMA-Beispiel beträgt der Einstieg 103,2 Q und das Equity am Ende 1.004,176 Q.
Das Pullback-Beispiel verwendet eine eigene synthetische Sequenz und ein
Preisraster von 0,25 Q. Der Einstieg erfolgt bei 100,25 Q, der Zielausstieg bei
102,50 Q und das Equity am Ende beträgt 1.018 Q.

Für einen gespeicherten Datensatz wird in der Backtest-TOML der Eintrag
`dataset_artifact = "pfad/zum/dataset.json"` verwendet. Er ersetzt
`data_config`. Der Pfad ist relativ zur Backtest-TOML oder absolut.
Die Artefaktprüfung und der gespeicherte Sitzungskalender gelten für den
gesamten Lauf. Der JSON-Bericht enthält zusätzlich `dataset_artifact_id`.

## Wiederverwendbarer HTML-Bericht

Der Bericht liest bereits gespeicherte Ergebnisse. Die HTML-Datei lässt sich
direkt im Browser öffnen. Bei der Erzeugung werden weder ein Modell trainiert
noch ein Backtest ausgeführt. Für die Vorhersagepunkte und den einzelnen
Ridge-Trade wertet der Befehl das eingefrorene Modell aus.

```sh
.venv/bin/python -m market_trader.cli report \
  --backtest /path/to/backtest.json --output /path/to/report.html
```

`--compare` ergänzt einen zweiten Backtest mit demselben Datensatz, denselben
Kontostand-Zeitpunkten und demselben Ausführungsprofil. `--evaluation` zeigt
die Fehlermaße eines passenden Ridge-Testberichts. `--model` zusammen mit
`--training-data` liefert die einzelnen Testprognosen. `--model` zusammen
mit `--dataset` rekonstruiert die Prognose und Merkmalsbeiträge jedes
abgeschlossenen Ridge-Trades. Der Zielpfad muss neu sein.

Vollständiger Vergleich mit den bereits lokal erzeugten GER30-Artefakten.

```sh
trade_run=datasets/ger30-15m-2026-02-17-to-03-06
training=datasets/ger30-15m-2025-10-02-to-2026-02-13/training
.venv/bin/python -m market_trader.cli report \
  --backtest "$trade_run/ridge-long.json" \
  --compare "$trade_run/sma-crossover.json" \
  --dataset "$trade_run/dataset.json" \
  --evaluation "$training/ridge-test.json" \
  --model "$training/ridge-validation.json" \
  --training-data "$training/training-data.json" \
  --output "$trade_run/ridge-vs-sma-report.html"
```

## Überwachter Datensatz über CLI

`market-trader data prepare-training --config prepare.toml --output training-data.json`
lädt ein verifiziertes Kerzenartefakt und speichert chronologische Beispiele
mit verfügbaren Merkmalen und einem vorwärts gerichteten Renditeziel. Der
[Datensatzvertrag](TRAINING_DATA_CONTRACT.md) enthält die TOML-Konfiguration,
Grenzregeln und das JSON-Format.

## CPU-Basismodell über CLI

`market-trader model train-baseline --training-data training-data.json --output ridge-validation.json`
prüft einen gespeicherten Trainingsdatensatz, passt ein Ridge-Modell an
und speichert Modellparameter und Validierungsfehler. `model evaluate-baseline`
lädt diese Parameter und vergleicht die Testfehler mit einer Nullvorhersage.
[Datensatzvertrag](TRAINING_DATA_CONTRACT.md)

## Aktueller Zustand mit 420 bestandenen Prüfungen

CSV-Import, Validierung, Resampling, historische Fenster und numerische
Merkmalsberechnung sowie Normalisierung, Qualitätsberichte und explizite
UTC-Sitzungskalender, Prüfung angefragter Zeiträume/Serien und lokale Speicherung
mit Herkunftsmanifest sind implementiert. Die CLI verbindet Import, Normalisierung
und geprüfte Veröffentlichung. Der R0-Backtest läuft über die Offline-CLI.
Am 2026-11-12 in der lokalen Testumgebung ausgeführt.
**420 bestanden, 0 fehlgeschlagen, 0 Sammlungs-/Fixture-Fehler, 0 übersprungen**,
Exitcode 0. Es gibt keine `skip`-/`xfail`-Markierungen für offene Verarbeitung.

Die ursprünglichen 48 Prüfungen wurden zuerst als Rotphase ausgeführt. Vor der
Implementierung kamen 42 Randprüfungen hinzu und wurden ebenfalls rot ausgeführt.
Eine anschließend ergänzte Regression für fehlerhafte CSV-Header-Zitate schlug
zunächst mit falscher Fehlerzuordnung fehl und besteht nach der Korrektur.
Im zweiten Arbeitspaket wurden zuerst 39 weitere Prüfungen in
`tests/test_normalization.py` rot ausgeführt: fehlende Normalisierungslogik,
keine Import-/Fixture-Fehler. Anschließend wurde die Normalisierung implementiert.
Im dritten Arbeitspaket folgten 46 rote Prüfungen in
`tests/test_session_calendars.py`, ebenfalls ohne Import-/Fixture-Fehler.
Nach Implementierung bestehen sie gemeinsam mit den bisherigen 130 Prüfungen.
Das vierte Arbeitspaket ergänzt 47 zunächst rote Prüfungen in
`tests/test_requested_ranges.py`, ohne Import-/Fixture-Fehler. Nach Implementierung
bestehen sie gemeinsam mit den bisherigen 176 Prüfungen.
Im fünften Paket wurden zuerst 55 Speicherprüfungen in
`tests/test_dataset_storage.py` rot ausgeführt, ohne Import-/Fixture-Fehler.
Zwei ergänzte Unicode-Regressionsfälle schlugen vor ihrer Korrektur ebenfalls
fehl. Der Nachweis der ausdrücklich gespeicherten Verarbeitungsversion wurde
vor seiner Ergänzung im bestehenden Roundtrip-Fall rot geprüft. Abschließend
bestehen alle 57 Varianten zusammen mit den bisherigen 223 Prüfungen.
Im sechsten Paket schlugen zuerst alle 17 CLI-Prüfungen in
`tests/test_data_cli.py` wegen des fehlenden Verhaltens fehl. Nach der Umsetzung
bestehen sie zusammen mit den bisherigen 280 Prüfungen. Der installierte Befehl
erzeugte aus zwei Arbeitsverzeichnissen bytegleiche Artefakte.

Ein zukünftiger Exitcode 1 ist wieder ein zu untersuchender Testfehler. Die
[pytest-Exitcodes](https://docs.pytest.org/en/stable/reference/exit-codes.html)
unterscheiden fehlgeschlagene Prüfungen und Fehler der Testausführung.

Die Testsuche ist auf `tests/` begrenzt. Das bestehende
`strategies/test_csv_loader.py` ist ein manueller Legacy-Versuch mit lokalen
Daten und wird von diesem neuen Paket nicht automatisch ausgeführt.

## Zuordnung der ersten Fälle

| Anforderung | Automatisierte Testfunktion(en) | Umfang |
| --- | --- | --- |
| DAT-001 | `tests/test_data_contract.py::test_dat_001_csv_import_preserves_values_units_and_timestamps` | Kanonischer Import mit explizitem Soll-Datensatz. |
| DAT-003 | `test_dat_003_missing_close_reports_source_row_and_field`, `test_dat_003_invalid_numeric_values_are_rejected`, `test_dat_003_csv_import_rejects_invalid_numbers` in derselben Datei | Fehlerzuordnung sowie numerische Grenzen im Import und in der Validierung. |
| DAT-007 | `tests/test_data_contract.py::test_dat_007_resampling_has_hand_calculated_ohlcv_and_time_bounds` | Vollständiges synthetisches Intervall. |
| CAU-001 | `tests/test_causality_contract.py::test_cau_001_only_full_historical_windows_are_yielded` | Vier Historienlängen; Strategie-Runner folgt später. |
| CAU-003 | `tests/test_causality_contract.py::test_cau_003_future_prices_do_not_change_historical_numeric_features` | Zahlenmerkmale; Bild- und Signalprüfung folgt später. |

`tests/test_data_boundaries.py` ergänzt die ursprünglichen fünf IDs um
Zeitstempel-, Schema-, Aggregations-, Größen- und Präzisionsfälle sowie Teile
weiterer Anforderungen. `tests/test_normalization.py` ergänzt Sortierung,
Dublettenaudit, Konflikte, Lückenregeln, Datensatzkennungen und Gruppenauswahl.
`tests/test_session_calendars.py` prüft aktive Sitzungen, geschlossene Zeiten,
Konfigurationsfehler und die Kalender-Metadaten in der Datenkennung.
`tests/test_requested_ranges.py` ergänzt Aufträge für Zeiträume und fehlende
Serien einschließlich Randlücken, erwarteter Anzahlen und Fehler ohne Quellzeile.
`tests/test_dataset_storage.py` prüft Speicherung, Wiederherstellung, ursprüngliche
Zeilenbezüge, Herkunft, ungültige Artefakte und Dateifehler:

| Anforderung | Bestandene Varianten insgesamt | Abgedeckter Umfang |
| --- | --- | --- |
| SET-004 | 1 | Daten-CLI mit derselben Konfiguration aus zwei Arbeitsverzeichnissen. |
| DAT-001 | 1 | Kanonischer Import und Nachkommastellen. |
| DAT-002 | 7 | UTC-Normalisierung und Abweisung naiver/ungültiger Zeitstempel. |
| DAT-003 | 54 | Zahlen, OHLC-Grenzen, Metadaten, CSV-Struktur und Fehlerzeilen, auch vor Normalisierung. |
| DAT-004 | 8 | Identische Dubletten mit Zeilen-Audit; Konflikte bei OHLCV und Verfügbarkeit. |
| DAT-005 | 46 | Sortierung, aktive/angefragte Lücken, Randlücken, leere Serien, strikte Abweisung, Raster, Auftragsvalidierung und CLI-Lückenregeln. |
| DAT-006 | 44 | Explizite UTC-Sitzungen, geschlossene Zeiten, aktive Auftragsgrenzen, Erwartungszahlen, Gültigkeitsbereich und ungültige Zeitpläne. |
| DAT-007 | 6 | Handgerechnete Aggregate, Grenzen, Intervallparameter und exakte Volumensumme. |
| DAT-008 | 3 | Explizite Abweisung unvollständiger Zielblöcke. |
| DAT-009 | 79 | Inhalts-/Auftragskennung, verlustfreie lokale Artefakte, Herkunftsmanifest, Quellprüfung, Integritäts-/Schemafehler, unveränderte Rohdaten und reproduzierbarer CLI-Import. |
| DAT-010 | 10 | Gruppen, Auswahl, vollständig fehlende angefragte Serien und isolierte Kennungen/Lücken; nicht angefragte Eingaben scheitern ausdrücklich. |
| CAU-001 | 12 | Fenster, Warmup, Größen und einheitlich geordnete Eingaben; Runner offen. |
| CAU-002 | 1 | Verfügbarkeitsmetadaten des Zahlenmerkmals; Freigabe im Runner offen. |
| CAU-003 | 6 | Numerische Kausalität, Lookback und unabhängiger Dezimalkontext; Bilder/Signale offen. |
| CAU-004 | 19 | Datei-, Speicher- und CLI-Fehlerfälle mit Diagnose, darunter ungültige Konfigurationen, fehlerhafte Rohdaten und veränderte Artefakte. |
| CAU-005 | 1 | Unveränderliche Momentaufnahme der Fenster-Eingabe; Strategieintegration offen. |

pytest-Marker `case` verknüpfen jede Funktion mit ihrer fachlichen ID.
Parametrisierte Varianten sind mehrere ausführbare Prüfungen derselben
Anforderung und erhöhen nicht die Zahl der 69 fachlichen Fall-IDs.

Die Beispieldaten in `tests/conftest.py` sind von Hand vorgegeben und verwenden
keinen Produktionsloader. Fenster und Resampling werden dadurch unabhängig vom
Dateiimport geprüft. Siehe [DATA_CONTRACT.md](DATA_CONTRACT.md)
für Felder, Einheiten und öffentliche Schnittstellen.
Speicherformat und Dateiverhalten stehen in [STORAGE_CONTRACT.md](STORAGE_CONTRACT.md).

## M2-Abnahmeprüfung

Am 2026-11-11 gegen die Fallzuordnung in [TEST_CASES.md](TEST_CASES.md) geprüft.

| Pflichtfälle | Nachweis und Stand |
| --- | --- |
| SET-001 | Bestanden für eine frische Projektkopie der aktuellen Arbeitsdateien. `uv sync --locked --group test` und die installierte CLI funktionierten mit Python 3.14.8. Der Nachweis aus einem Git-Checkout setzt einen Commit mit diesen Dateien voraus. |
| SET-002 | Bestanden in einer separaten Umgebung mit `uv sync --locked --no-dev`. Domain, CSV-Import und CLI ließen sich importieren. Als Distribution war allein `market-trader` installiert. |
| SET-003 | Die Tests bestanden mit lokalen Fixtures und eingeschränktem Netzwerkzugriff. |
| SET-004 | Der Datenablauf erzeugte aus zwei Arbeitsverzeichnissen bytegleiche Artefakte. Der installierte Backtest-Befehl erzeugte entsprechend bytegleiche JSON-Berichte. |
| DAT-001–004 und DAT-006–010 | Der Datenumfang ist durch die oben zugeordneten Tests belegt. DAT-009 prüft zusätzlich eine wiederholte Aggregation und den Einfluss des Zielintervalls auf die Datensatzkennung. |
| DAT-005 | Normalisierung erkennt und verwirft im strikten Modus aktive Lücken. Der Runner weist lückenhafte Datensätze vor dem ersten Strategieaufruf ab. |
| CAU-001–005 | Fenster, Warmup, verspätete Daten, unveränderliche Eingaben, kausale Zahlenmerkmale und Signale sowie weitergereichte Strategiefehler sind geprüft. Die Bildprüfung folgt in M6. |

## Erstes M3-Paket

`tests/test_backtesting.py` enthielt zunächst 17 Varianten. Alle schlugen mit der
Funktionshülle zunächst an fehlendem Verhalten fehl und bestanden nach der
Umsetzung. Die Fälle prüfen Kontoführung für Long und Short, Gebühren,
Marktorders, Stop- und Zielfälle, Ausführung bei Kurslücken, Warmup und
verspätete Daten. Profil- und Kapitalschranken werden vor einem Fill geprüft.

## Zweites M3-Paket

29 zusätzliche Prüfungen decken Endschließung, Spread, Slippage, Zielpreis mit
Kosten, risikobasierte Mengen, Drawdown, Berichtsabgleich und reproduzierbare
CLI-Läufe ab. Die neuen Funktionen wurden mit fehlgeschlagenen
Verhaltensprüfungen begonnen. Die späteren Regressionen sichern exakte
Tick-Rundung und das Verhalten fester Mengen bei einem abweichenden Open.
Der [Backtest-Vertrag](BACKTEST_CONTRACT.md) beschreibt die Python-Schnittstelle
und den JSON-Bericht.

## Erstes M4-Paket

Zehn zusätzliche Prüfungen decken SIG-003–005 und den Strategiewechsel aus
REP-005 ab. Sie prüfen ungültige Namen und Parameter vor dem Datenzugriff,
die erwarteten SMA-Ein- und Ausstiege sowie isolierten Strategiezustand.
Die beiden CLI-Konfigurationen unterscheiden sich ausschließlich in
`[strategy]`. Datensatzkennung, Ausführungsprofil und Berichtsformat stimmen
überein. Die Prüfungen wurden vor und nach der Umsetzung ausgeführt.

## Zweites M4-Paket

15 zusätzliche Prüfungen decken die Pullback-Portierung ab. Sie prüfen
die Vier-Kerzen-Folge, Grenzfälle, verspätete Daten, den Einstieg am nächsten
Open, Ziel und Stop-Lücke, das Preisraster aus der Ausführungskonfiguration
sowie ungültige Strategieparameter. Alle 15 schlugen vor der Implementierung
fehl und bestanden danach.

## Backtest mit gespeichertem Datensatz

Sieben zusätzliche Prüfungen decken den artefaktbasierten Backtest ab.
Sie prüfen die Wiederholbarkeit aus verschiedenen Arbeitsverzeichnissen,
die Herkunft und Artefaktkennung im Bericht, einen gespeicherten
Sitzungskalender sowie fehlende oder veränderte Dateien, aktive Lücken und
ungültige Datenquellen in der Konfiguration. Alle sieben schlugen vor der
Implementierung fehl und bestanden danach.

## Erstes M5-Paket

Vierzehn Prüfungen in `tests/test_training_data.py` schlugen vor der
Implementierung fehl. Mit der zusätzlichen Variante für direkt angrenzende
Sitzungen bestehen 15 Prüfungen. Sie decken historische Verfügbarkeit,
Zielrendite, Grenzbereinigung, Sitzungswechsel und reproduzierbaren Export ab.
Die installierte CLI erzeugte den GER30-Export aus einem zweiten
Arbeitsverzeichnis bytegleich.

## Zweites M5-Paket

Neun Prüfungen in `tests/test_baseline.py` schlugen vor der Implementierung
fehl und bestanden danach. Die Fälle prüfen die Skalierung im Training,
die Modellkoeffizienten, zeitliche Grenzen, die Trennung von Validierungs-
und Testdaten, die Prüfsumme und die reproduzierbare CLI-Ausgabe.

## Drittes M5-Paket

Acht zusätzliche Prüfungen in `tests/test_baseline.py` schlugen vor der
Implementierung fehl und bestanden danach. Die Fälle prüfen gespeicherte
Modellparameter und Testfehler, abweichende Eingabedaten, ungültige
Modellparameter, fehlerhafte Testgrenzen sowie die Veröffentlichung des
Testberichts.

## Eingefrorenes Modell im Backtest

```sh
.venv/bin/python -m pytest -q tests/test_model_strategy.py
```

Der synthetische CLI-Fall prüft Einstiegs- und Ausstiegs-Open, Fill-Preise,
Gebühren und End-Equity. Weitere Varianten prüfen verspätete Kerzen,
Sitzungsgrenzen, Prognoseschwelle, Modellkompatibilität und Wiederholbarkeit.
Die Neuberechnung des 92-Sitzungs-Trainingsartefakts mit der gemeinsam
verwendeten Merkmalsfunktion ergab bytegleiche Daten.
[Backtest-Vertrag](BACKTEST_CONTRACT.md#eingefrorenes-ridge-modell)

## Feste SMA-Menge

Vier Varianten in `tests/test_backtesting.py` prüfen feste Mengen für
`sma_crossover`. Der bestehende Fall mit Risikobudget und Stop wird
weiter ausgeführt.
