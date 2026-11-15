# AP2 – lokale Experiment-Konfiguration und Laufverwaltung

## Auftrag und Umfang

Andreas hat nach Abschluss von AP1 dessen Commit/Push und anschließend die Umsetzung von AP2 freigegeben. AP1 wurde als `655b82cee81411eee404dd66bb4e47faf6bf361a` nach `origin/main` gepusht und per Remote-Readback bestätigt.

AP2 ergänzt den bestehenden Offline-Trainings-/Backtestpfad, keinen zweiten Simulator. Ziel sind konfigurierbare Ridge-Experimente mit nachvollziehbaren unveränderlichen Konfigurationssnapshots und sicherer Wiederholung. Das ist noch keine Browser-Trainingsoberfläche, Warteschlange, automatische Parametersuche oder neuronales Modell.

**AP2 ist implementiert und technisch abgenommen.** Echte Referenz-/Replay-Läufe, Kostenabweichung, Fehlerfall, installierter Build, Gesamttests und unabhängiges Review sind geprüft. Belege: [AP2_RESULTS.md](AP2_RESULTS.md).

## Feste fachliche Grenzen

- Quelle weiterhin Dukascopy `deuidxeur`, 15-Minuten-Bid-Daten und vollständige beobachtete Berliner Forschungsfenster.
- Training 2021–2023, Validierung 2024; 2025 bleibt reserviert und erreicht keine Modellmerkmale/-bewertung.
- Neues Open-to-Open-Ziel mit konservativer Labelverfügbarkeit, kein stiller Wechsel des alten v1-Vertrags.
- Ridge bleibt im ersten AP2-Umfang bei Alpha 0.1. Konfigurierbar werden kausales Lookback/Zielhorizont, Einstiegsschwelle, passende SMA-Kontrollperioden und hypothetische Ausführungskosten.
- Beide Regeln verwenden dieselbe zulässige Entscheidungskohorte; SMA-Slow muss dem Modell-Lookback entsprechen. Feste Menge 1, keine Shorts, Übernachtpositionen oder automatische Modell-/Regeloptimierung.
- Ein Seed wird gespeichert; der deterministische Ridge-Löser verwendet keine Zufallsinitialisierung. Ein Seed ist keine plattformübergreifende Bitgleichheitsgarantie.
- Das eingefrorene AP1-Kommando bleibt erhalten. Ein AP2-Variantenbericht darf keine abweichenden Parameter unter dem AP1-Namen oder mit festen alten Kosten beschriften.

## CLI-Vertrag

Die Registry ist ein privates Verzeichnis außerhalb von Git und Webverzeichnis.

```sh
market-trader experiment run --config configs/ap2-reference.toml --registry /private/experiment-runs
market-trader experiment list --registry /private/experiment-runs
market-trader experiment show --registry /private/experiment-runs --run-id RUN_ID
market-trader experiment replay --registry /private/experiment-runs --run-id RUN_ID
market-trader experiment compare --registry /private/experiment-runs --left RUN_ID --right OTHER_RUN_ID
```

`run` akzeptiert optional `--parent RUN_ID`. `replay` lädt den überprüften gespeicherten Snapshot und erstellt eine neue ID mit Herkunftsverweis; es überschreibt oder verändert den ursprünglichen Lauf nicht. CLI-Ausgaben sind JSON. Ein fachlich zulässiger, aber inkompatibler Vergleich liefert `compatible: false`, keinen verschwiegenen Gleichheitsanspruch.

## Laufakte und Wiederholung

Jeder Lauf erhält eine ausschließlich für diesen Lauf neu angelegte Kennung, Name, Hypothese, optionalen Parent, Konfiguration mit Prüfsumme, Seed, Status und Zeitstempel, Log und Artefaktpfade. Datensatz-, Modell-, Ziel-/Feature- und Code-/Runtime-Identitäten werden mitgeführt. Fehlgeschlagene Ausführungen müssen als Fehler sichtbar bleiben, nicht als fertige Läufe erscheinen.

Die Konfigurationssnapshot-Datei wird durch die Anwendung nicht nachträglich geändert; das ist keine kryptographisch manipulationssichere Ablage gegenüber dem Dateisystemeigentümer. Prüfsummen helfen, Inkonsistenzen zu erkennen. Ein harter Prozessabbruch kann einen zuletzt laufenden Status hinterlassen; Queue-/Recovery-Bedienung ist nicht Teil von AP2.

Vergleiche unterscheiden Parameter-/Hypothesenänderungen von Bedingungen: andere Quelldaten, Ziel-/Featurekohorten, Zeiträume oder Ausführungskosten sind kein uneingeschränkt fairer Modellvergleich und müssen explizit abweichend erscheinen. AP2 trifft keine automatische Profitabilitäts- oder Modellauswahlentscheidung.

## Abnahme

1. Ein dokumentiertes Konfigurationsfile löst tatsächlich Import, Training, eingefrorene Auswertung, Backtests und Bericht aus.
2. Jeder Lauf ist eindeutig; keine alten Dateien werden überschrieben. Parent-/Replay-Herkunft bleibt sichtbar.
3. Wiederholung aus dem Snapshot reproduziert die fachlichen Ergebnisse unter derselben Umgebung.
4. Konfiguration, Versionen, Status, Logs, Fehlermeldungen und Artefakte können über CLI inspiziert werden.
5. Ein absichtlich geändertes Kostenprofil wird als abweichende Vergleichsbedingung erkannt.
6. Bestehende AP1-/Legacy-Tests und neue Pflichtprüfungen bestehen; unabhängiges Review und Graphify sind aktualisiert.
7. Reale Abnahmeläufe verwenden nur Training/Validierung bis Ende 2024; keine reservierten 2025-Modellmetriken.

## Aktueller Prüfstand

AP1-Ausgangsbasis: 653 Tests bestanden. AP2-CLI und Module sind integriert; nach zusätzlicher Typprüfung beschädigter Run-Records besteht die Parent-Gesamtsuite **786 Tests**. Echte Referenz, Snapshot-Replay, bewusst abweichende Kosten und ein gezielt fehlschlagender Import wurden ausgeführt und geprüft. Ein separat installiertes Wheel hat ebenfalls einen echten Lauf außerhalb des Checkouts ausgeführt. Einzelbelege und Grenzen stehen in [AP2_RESULTS.md](AP2_RESULTS.md). Das unabhängige Review und die separate Nachprüfung der Run-Record-Typprüfung sind ohne offene Befunde abgeschlossen.

## Konfigurationsschema v1

`configs/ap2-reference.toml` enthält die sechs Pflichtfelder `version`, `name`, `hypothesis`, `seed`, `source_csv`, `source_sha256`. Relative CSV-Pfade gelten relativ zur TOML-Datei. Folgende Tabellen sind optional; hier die Defaultwerte:

```toml
[training]
lookback_bars = 8
horizon_bars = 2

[strategy]
entry_threshold = "0"
sma_fast_period = 4
sma_slow_period = 8

[execution]
spread = "2.0"
slippage = "0.1"
commission_rate = "0"
price_tick = "0.001"
initial_capital = "100000"
```

Dezimalwerte können als Zahlen oder Dezimalstrings eingegeben werden. Der vollständige Snapshot speichert diese als Strings und die Quelle als absoluten Pfad. Unbekannte Felder, Modell-/Datums-/Mengenüberschreibungen und nicht endliche Werte werden abgewiesen. Es gilt `lookback >= 2`, `horizon >= 1`, `lookback + horizon + 1 <= 34` und `SMA fast < slow == lookback`.

Pro Registry-Lauf liegen `run.json`, `config.json`, `run.log` und nach erfolgreichem Abschluss `artifacts/` vor. Das Log enthält Lebenszyklus und Fehler, noch keine detaillierte Trainingsfortschrittsoberfläche. `read/show/list/compare` verifizieren die fertig gespeicherten Artefakte; bei vielen großen Läufen ist die vollständige Integritätsprüfung bewusst aufwendiger als eine ungeprüfte Dateiliste.
