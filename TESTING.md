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

Die Berichtserzeugung prüft nun die interne Konsistenz von Fills, Gebühren,
Brutto-/Netto-P&L, Ausführungsprofil und Kontoverlauf. Widersprüchliche Eingaben
werden vor der HTML-Veröffentlichung abgewiesen. Das ersetzt keine unabhängige
Prüfung der Marktdaten; insbesondere lässt sich die Bewertung einer offenen
Position ohne zugehörige Kerzen nicht vollständig nachrechnen.

Für den nur lesenden Zugriff im Browser siehe [REPORT_VIEWER.md](REPORT_VIEWER.md).
`market-trader serve-reports` bietet eine separate interne HTML-Übersicht;
Quellcode, JSON und CSV werden darüber nicht bereitgestellt.

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

## AP1: eingefrorenes echtes Referenzexperiment

Freigegebener Vertrag: [AP1_REFERENCE_EXPERIMENT.md](AP1_REFERENCE_EXPERIMENT.md).
Ausgeführte Ergebnisse, private Artefaktpfade, Kostenannahmen und Wiederholung:
[AP1_RESULTS.md](AP1_RESULTS.md).

```sh
/home/hermes/workspace/.venvs/market-trader-review/bin/python -m pytest -q
/home/hermes/workspace/.venvs/market-trader-ap1/bin/market-trader reference \
  --config configs/ap1-reference.toml \
  --output /home/hermes/workspace/var/market-trader/runs/ap1-new-reproduction
```

Nach der Audit-Grenzkorrektur: **653 Tests bestanden**. Die AP1-Tests liegen in
`test_dukascopy_import.py`, `test_open_target.py`, `test_sma_timed.py`,
`test_reference.py` und `test_reference_reporting.py`. Sie nutzen nur kleine
synthetische Eingaben, einschließlich veränderter 2025-Preise zum Leakage-Test.
Der echte Lauf erzeugt 18.528 Trainings- und 6.192 Validierungsbeispiele;
reservierte Testzeilen bleiben leer. Bestehende v1-Targets und Artefakte bleiben
getrennt vom neuen Open-to-Open-v2-Vertrag unterstützt.

Zwei vollständige echte Läufe aus gleicher Umgebung erzeugten alle 15 Dateien
bytegleich. Ein separater Wheel-Lauf bestätigte gleiche fachliche Ergebnisse.
Die JSON-/Quell-/Label-/Trade-Prüfung steht privat unter
`var/market-trader/acceptance/verify_ap1.py`; dort wird 2025 nicht modelliert.

## AP2: lokale Experimentläufe

Vertrag und exakte TOML-Tabellen: [AP2_EXPERIMENTS.md](AP2_EXPERIMENTS.md).
Echte Abnahme, Run-IDs und installierter Wheel-Aufruf: [AP2_RESULTS.md](AP2_RESULTS.md).

```sh
/home/hermes/workspace/.venvs/market-trader-review/bin/python -m pytest -q
/home/hermes/workspace/.venvs/market-trader-ap2/bin/market-trader experiment run \
  --config configs/ap2-reference.toml \
  --registry /home/hermes/workspace/var/market-trader/experiments/my-research
```

Nach abschließender Typprüfung beschädigter Run-Records: **786 Tests bestanden**.
Die vier AP2-Testmodule prüfen Konfiguration, Pipeline, Registry, CLI und Replay
mit synthetischen Kerzen. Zusätzlich wurden echte Referenz, bytegleiches Replay,
abweichendes Kostenprofil und ein absichtlich fehlschlagender Import ausgeführt.
Das separat installierte Wheel lieferte gleiche Modell-/Handelswerte. 2025 wurde
in keinem dieser Läufe modelliert oder bewertet.

Eine JSON-Datei kann syntaktisch gültig sein, ohne ein Run-Objekt zu enthalten:
Array, null, String und Zahl werden daher explizit mit verständlichem Fehler
abgewiesen, statt über einen AttributeError einen rohen CLI-Traceback auszugeben.

## AP3: erster neuronaler Vergleich

Vertrag: [AP3_NEURAL_REFERENCE.md](AP3_NEURAL_REFERENCE.md).
Isolierter, hashgeprüfter CPU-Stack: [AP3_ENVIRONMENT.md](AP3_ENVIRONMENT.md).
Reale Ergebnisse, Run-IDs, Replay und Einschränkungen: [AP3_RESULTS.md](AP3_RESULTS.md).

```sh
/home/hermes/workspace/.venvs/market-trader-ap3/bin/python -m pytest -q
/home/hermes/workspace/.venvs/market-trader-review/bin/python -m pytest -q
/home/hermes/workspace/.venvs/market-trader-ap3/bin/market-trader experiment run \
  --config configs/ap3-neural-reference.toml \
  --registry /home/hermes/workspace/var/market-trader/experiments/my-neural-runs
```

Finaler Stand: **891 Tests bestanden** mit dem Trainingsstack; ohne Torch
**885 bestanden, 6 reine Trainingstests übersprungen**. Modell-JSON, Inferenz,
Strategie und Berichte bleiben ohne Torch/NumPy nutzbar. Der echte Replay ist
in allen 22 Backend-Artefakten bytegleich; 6.192 Reload-Prognosen wurden exakt
geprüft. Eingabe-/Zielskalierer wurden separat aus Training nachgerechnet,
alle MLP-/Ridge-/SMA-Trades gegen Quell-Opens und Kosten reconciled.

Frühes Stoppen wählt den Checkpoint auf 2024; diese Werte sind ausdrücklich
Entwicklungs-/Selektionsmetriken, keine unabhängigen finalen Testwerte.
2025 wurde weder zum Training noch zur Modellbewertung verwendet.

## AP4: nur lesender Experimentvergleich im Browser

Der Umfang und die HTTP-/Datenverträge stehen in [AP4_WORKBENCH.md](AP4_WORKBENCH.md),
der Betrieb einschließlich opt-in Registry-Quellen in [REPORT_VIEWER.md](REPORT_VIEWER.md).
Die Serving-Umgebung bleibt ohne externe Python-Laufzeitabhängigkeiten; Torch,
NumPy und Node werden zum Lesen vorhandener Ergebnisse nicht benötigt.
**Node muss für die ausführbaren Frontend-Vertragstests im Test-PATH verfügbar sein.**
Die Node-Tests sind keine visuelle Browserprüfung.

```sh
/home/hermes/workspace/.venvs/market-trader-review/bin/python -m pytest -q \
  tests/test_workbench_data.py tests/test_workbench_server.py tests/test_workbench_template.py
/home/hermes/workspace/.venvs/market-trader-ap3/bin/python -m pytest -q
/home/hermes/workspace/.venvs/market-trader-review/bin/python -m pytest -q
```

Die neuen synthetischen Tests prüfen Integritätsumschläge und Cache-Invalidierung,
Quell-/Run-Auswahl, gespeicherte Metriken/Einheiten, echte Konto-zu-Kurven-Abbildung,
Sampling und Trainingshistorie, Vergleichssignaturen, endliche Zahlen, sichere
HTTP-Grenzen und die reale Inline-JavaScript-Logik mit einem kleinen Test-DOM.
Schreibende Tests verwenden ausschließlich temporäre synthetische Registries,
nicht die akzeptierten AP2-/AP3-Läufe. Neben korrupten Bytes sind auch kohärente
Versions-/Statuswechsel während eines Vergleichs abzusichern.

Die separate Abnahme nutzt die vorhandenen echten, unveränderten Referenzläufe.
HTTP-Projektionen werden gegen gespeicherte Summarys, vollständige Kontoverläufe,
Epochen und HTML-Bytes abgeglichen; alle Registry-Dateien werden vor/nachher
gehasht. Die private Automatisierung liegt unter `var/market-trader/acceptance/`:
`exercise_ap4.py` und `ap4_browser.mjs`. Vor Wiederverwendung Ziel-URL und neue
Belegpfade prüfen, bestehende Belege nicht überschreiben. Der Browserlauf prüft
Desktop/390px/320px, Filter, Auswahlgrenzen, kompatible/inkompatible Vergleiche,
Deep Links, Keyboard, Fehlerfälle und ausdrücklich markierte XSS-Text-Fixtures.
Er verwendet ein separat installiertes Wheel außerhalb des Checkouts; PNGs,
Konsole und Netzwerk sind eigenständige Nachweise zusätzlich zu pytest.

## AP5: lokale Varianten und begrenzte Trainingsjobs

Vertrag/Bedienung: [AP5_EXPERIMENT_CONTROL.md](AP5_EXPERIMENT_CONTROL.md).
Tatsächliche Fits, Replay, Wheel und Grenzen: [AP5_RESULTS.md](AP5_RESULTS.md).

```sh
/home/hermes/workspace/.venvs/market-trader-ap3/bin/python -m pytest -q
/home/hermes/workspace/.venvs/market-trader-review/bin/python -m pytest -q
/home/hermes/workspace/.venvs/market-trader-ap3/bin/python -m pytest -q \
  tests/test_ap5_model.py tests/test_experiment_variants.py tests/test_ap5_cli.py \
  tests/test_ap5_integration.py tests/test_experiment_jobs.py tests/test_experiment_registry.py
```

Stabiler Abschlussstand: **1.143 bestanden** mit Torch; **1.135 bestanden / 8
reine Trainingstests übersprungen** ohne Torch/NumPy. Die neuen Tests decken
strikte v3-Snapshots und MLP-JSON v2, echte synthetische variable Fits/Reload,
Legacy-Semantik, kausale Fenster, tatsächliche Architekturtexte, v3-SMA-Bedingungen,
verifizierte Parent-Pläne, eindeutige höchstens vier Varianten und Auswahl nach
Validierungs-RMSE ab. Bool/int-Aliase dürfen keine Provenienzprüfung passieren.

Lifecycle-Tests müssen echte Subprozesse ergänzen, nicht nur simulierte Exitcodes:
FIFO-basierten Abbruch/Wall-Timeout, nativen CPU-Hardkill plus gewöhnlichen
SIGKILL, Controller-Tod, geerbte Workerflocks, TERM→KILL, Queuekapazität und
Exception-Rollback. Crashfenster vor Config-/pending-/running-Publikation prüfen;
Run-Besitz muss schon im Allocation-Hook dauerhaft feststehen. Unveröffentlichte
Anlagen nicht als completed erfinden und unbeteiligte gleiche Konfigurationen
nicht über Hashsuche übernehmen. Kernel-CPU-Accounting muss sich auf den exakten
Child beziehen, nicht auf aggregate RUSAGE_CHILDREN-Werte.

Die reale Abnahme bleibt auf vier vorab begrenzte Fits außerhalb Git beschränkt.
Keine Fixtures als Marktresultate ausgeben, keine 2025-Features/-Metriken, keine
alten Registries/Live-Reports verändern. Wiederholungen brauchen frische Beleg-
und Ausgabepfade sowie ein neu freigegebenes tatsächliches Trainingsbudget.
Der installierte Wheel-Kontext wird außerhalb des Checkouts geprüft; echte
Modell-/Prognosegleichheit von abweichenden Git-/Lock-Metadaten unterscheiden.

## AP6: kausale Chartbilder und getrennte Auditvorschau

Vertrag: [AP6_IMAGE_DATA.md](AP6_IMAGE_DATA.md); tatsächliche Abnahmen und
Reviewstatus: [AP6_RESULTS.md](AP6_RESULTS.md). AP6 trainiert kein Bildmodell.

```sh
/home/hermes/workspace/.venvs/market-trader-review/bin/python -m pytest -q \
  tests/test_chart_images.py tests/test_image_dataset.py tests/test_image_publication.py \
  tests/test_image_gallery.py tests/test_image_gallery_capture.py tests/test_images_cli.py
```

Ausgeführter Abschlussstand: **152 Bildtests**, vollständige Suite **1.295 bestanden**
mit Torch; **1.287 bestanden / 8 Training-Skips** ohne Torch. Vollsuiten mit engen
Subprozess-Walllimits nicht gleichzeitig ausführen; ein Parallel-Lastfehler und
seine unverändert erfolgreichen isolierten/Vollsuiten-Nachläufe sind dokumentiert.

Die synthetischen Tests prüfen PNG-Chunks/CRC/DEFLATE und echte Pixel, kausale
Fenster, getrennte Zukunfts-/Labeldaten, Ressourcen- und Sessiongrenzen,
Original-Zielarithmetik und widerspruchsfreie mehrfach auftauchende Kerzen.
Dateisystemtests verwenden echte `renameat2`-Aufrufe mit gezielten Einfügungen:
private Staging-Container, vertauschte äußere Namen, ersetzte Vorfahren vor/nach
Publikation, konkurrierende leere Verzeichnisse/Dateien/Symlinks, kein Überschreiben
oder Löschen fremder Ziele. Private Staging- und Cache-Pins benötigen Linux.

Die eigenständige Vorschau muss dieselben verifizierten Dateibytes erfassen,
PNG-Base64 rückwärts bytegleich liefern, Metadaten escapen, Änderungen während
Capture/Rendering abweisen und auch über `..` außerhalb des Bundles bleiben.
Sie enthält menschliche Audit-Zielwerte; das HTML ist kein Modellinput.

Echte Abnahme separat außerhalb Git/Reports: 32 Trainings- und 32
Validierungsbilder, wiederholter kalter/warmer Export, alle 68 Dateien bytegleich.
Ein nicht-editierbar installiertes Wheel ohne Torch/NumPy erzeugt dieselben
Artefakte und dieselbe Vorschau. Chromium dekodiert alle 64 PNGs pixelgleich zum
unabhängigen Decoder; Desktop/390px/320px und Tastatur-/Detailansichten sind
separat geprüft. Dieser begrenzte Audit-Satz ersetzt keinen vollständigen
AP7-Trainingsdatensatz und erteilt keine Trainings-/Veröffentlichungsfreigabe.

## AP7: volle Bildkohorte und visuelles Referenznetz

Vertrag: [AP7_VISUAL_REFERENCE.md](AP7_VISUAL_REFERENCE.md); tatsächliche
Ergebnisse/Abnahme: [AP7_RESULTS.md](AP7_RESULTS.md). Die kanonische v4-Konfiguration
steht in `configs/ap7-cnn-reference.toml`; v1–v3 werden nicht neu interpretiert.

Vollständig ausgeführte, **sequentielle** Suiten auf dem eingefrorenen Stand:
**1.460 bestanden mit Torch**, **1.445 bestanden / 15 optionale Skips ohne Torch**.
Lange Vollsuiten als überwachte Hintergrundprozesse mit eigenen Logs und echten
Exitcodes ausführen; ein Transport-Zeitlimit ist kein Testsuite-Ergebnis.

```sh
/home/hermes/workspace/.venvs/market-trader-review/bin/python -m pytest -q \
  tests/test_ap7_config.py tests/test_ap7_integration.py tests/test_cnn_settings.py \
  tests/test_image_pixels_v1.py tests/test_image_training_data.py tests/test_image_neural.py \
  tests/test_ap7_reporting.py tests/test_ap7_trade_audit.py tests/test_ap7_workbench.py
```

Diese Tests bleiben synthetisch. Echte Fits benötigen das eingefrorene Budget;
bei der Abnahme wurden genau zwei CNN-Fits plus zwei Ridge-Kontrollfits, kein MLP
und keine Suche ausgeführt. 18.528 Trainings-/6.192 Validierungsbeispiele wurden
vollständig abgeglichen; 2025 bleibt für Modelleingaben/Bewertung ausgeschlossen.
Der Erstlauf und sein Replay liefern 27 bytegleiche Artefakte.

Das separat installierte Wheel ohne Torch/NumPy muss alle 6.192 tatsächlichen
PNG-/Fensterprognosen bitgleich rekonstruieren, die train-only Zielskalierung
unabhängig nachrechnen und identische Berichtsbytes/Bildauditdaten erzeugen.
Gemessene Framework-/Stdlib-Parität wird zusätzlich über alle Validierungszeilen
protokolliert; keine universelle Plattform-Gleichheit aus Seeds ableiten.

Die reale Browserabnahme prüft Desktop sowie 390px/320px, echte CNN-Historie,
kompatibles Erstlauf-/Replaypaar, abgewiesene AP5-/CNN-Vergleichbarkeit,
Berichtsbytes und die ersten acht chronologischen ausgeführten Trades samt
PNG-Browserpixeln, gespeicherter Prognose und Ledgerbindung. Read-only-, Host-,
Traversal-/JSON-Abweisung und unveränderte Forschungsinventare bleiben Pflicht.
Die Bildauditansicht ist absichtlich technisch/metadatenreich; sie ist keine
Salienz-/Attributionsdarstellung und nicht der Trainingsinput.
