# Roadmap-Entwurf: ML-Experimentierumgebung für market-trader

## Zweck und Status

Andreas' Ziel ist eine reproduzierbare Umgebung zum Trainieren, Variieren und Vergleichen neuronaler Modelle auf tabellarischen Marktdaten und/oder Chartbildern. Der vorhandene Backtest ist der gemeinsame Prüfstand, die Webseite die Vergleichsoberfläche und später gegebenenfalls die Bedienoberfläche für Trainingsaufträge. Regelstrategien bleiben Referenzen, nicht der primäre Entwicklungsschwerpunkt.

Dieser Entwurf priorisiert die vorhandenen ROADMAP.md und MODEL_DEVELOPMENT_ROADMAP.md nach der aktuellen Zielklärung. Er ersetzt deren fachliche Verträge nicht und ist noch kein Auftrag, alle Pakete umzusetzen. FastAI ist im bisherigen Modellplan vorgeschlagen; ein konkreter Trainingsstack wird erst nach einem isolierten Kompatibilitäts-/Smoke-Test verbindlich gewählt. Kein Plattform-Neubau und kein Live-Trading in diesem Umfang.

## Vorhandene Grundlage

- Offline-Datenverarbeitung und Backtester mit gemeinsamen Ausführungs-/Kostenregeln.
- Chronologische Trainingsdaten, Ridge-Referenz, eingefrorene Modellinferenz und Modell-Backtest.
- HTML-Berichte; lokale Berichtskonsistenzprüfung und interner Viewer wurden ergänzt.
- Letzter ausgeführter Gesamttest nach AP3:891Tests bestanden; ohneTorch885bestanden und6reineTrainingstests übersprungen. Abnahmen: [AP1_RESULTS.md](AP1_RESULTS.md), [AP2_RESULTS.md](AP2_RESULTS.md), [AP3_RESULTS.md](AP3_RESULTS.md).
- Der Report-Service ist auf Hermes installiert und läuft separat aus einem Wheel; Andreas hat Browser- und VPN-Zugriff bestätigt.
- AP1-/Report-/Viewer-Stand ist unter `655b82c` committed und auf `origin/main` verifiziert. Andreas hat anschließend AP2 zur Umsetzung freigegeben; aktueller Stand in [AP2_EXPERIMENTS.md](AP2_EXPERIMENTS.md).

## Reihenfolge der Arbeitspakete

### AP0 — Stand sichern und nächsten Umfang vereinbaren

Änderungen prüfen und dokumentieren, ersten Forschungsmarkt und Datenbestand zuordnen, Entscheidungsspielräume und Freigaben für lokale Entwicklung, Training, Deployment und Git festhalten. Vorhandene Kernfunktionen nicht neu bauen.

**Abnahme:** Klarer Ausgangsstand, ein priorisiertes nächstes Paket und überprüfbare Fertig-Kriterien. Keine vermischten parallelen Änderungen auf denselben Dateien.

### AP1 — Echten Referenzdatensatz und Experimentvertrag herstellen

**Technisch abgeschlossen am 15. November 2026.** Ausgeführte Referenz, Reproduzierbarkeit, unabhängiges Review und interner Bericht: [AP1_RESULTS.md](AP1_RESULTS.md). AP2 wurde anschließend ausdrücklich zur Umsetzung freigegeben; siehe [AP2_EXPERIMENTS.md](AP2_EXPERIMENTS.md).

Einen verfügbaren historischen Datensatz importieren; Instrument, Quelle, Zeitzone, Handelszeiten, Preis-/Mengenraster, Kosten, Auflösung, Zeitraum und Nutzungsbedingungen festhalten. Rohdaten unverändert außerhalb Git lagern, normalisierte Version mit Kennung referenzieren. Trainings-, Validierungs- und finalen Testzeitraum festlegen. Vorhersageziel, Verfügbarkeit und geplante Ein-/Ausstiege abstimmen; die bisherige Close-Label-/Open-Exit-Abweichung bewusst entscheiden, nicht übersehen.

**Abnahme:** Ein wiederholbarer Import und Ridge-/Regel-Referenzlauf auf denselben Daten; fachlich überprüfte Beispiel-Labels, getestete Zeitgrenzen und dokumentierte Datensatzkennung. Keine großen Datensätze oder Secrets im Git-Repo. Ein finaler Testzeitraum wird nicht für laufendes Tuning genutzt.

### AP2 — Einheitliche Experiment-Konfiguration und Laufverwaltung

**Technisch abgeschlossen.** Konfigurierbare lokale CLI, überprüfte Snapshots/Parents/Replays, Status/Logs und explizite Vergleichsbedingungen sind implementiert. Abnahme mit echten Läufen, 786 Tests, unabhängigem Review und installiertem Wheel: [AP2_RESULTS.md](AP2_RESULTS.md). Keine Freigabe für AP3 oder spätere Pakete daraus ableiten.

Ein konfigurierbarer Ablauf für Daten -> Training -> eingefrorenes Modell -> Vorhersagen -> Backtest -> Bericht. Jeder Run erhält ID, optionalen Parent, Hypothese, unveränderlichen Konfigurationssnapshot, Datensatz-/Split-/Feature-/Zielversion, Code-/Dependency-Stand, Seed, Status, Logs und Artefaktpfade. Vorhersageartefakte machen Modell und Handelsregel getrennt auswertbar. Für den Start genügt ein einfacher lokaler Mechanismus statt zusätzlicher Plattform-Infrastruktur.

**Abnahme:** Ein Ridge-Experiment lässt sich mit einem dokumentierten Aufruf ausführen und aus seiner Konfiguration erneut starten. Alte Ergebnisse werden nicht überschrieben. Unterschiedliche Daten-/Kostenbedingungen werden erkennbar und nicht stillschweigend als fairer Modellvergleich behandelt. Reproduzierbarkeit wird geprüft; ein Seed allein ist keine plattformübergreifende Bitgleichheitsgarantie.

### AP3 — Erstes kleines tabellarisches neuronales Modell, Ende-zu-Ende

**Technisch abgeschlossen.** Das festgelegte MLP4→16→8→1 wurde tatsächlich trainiert, sicher gespeichert/geladen, im bestehenden Ledger bewertet und intern als Vergleich zu Ridge/Null/SMA veröffentlicht. Reales Replay,891Tests,885/6ohneTorch und unabhängige Reviews: [AP3_RESULTS.md](AP3_RESULTS.md). Negative Leistung ist dokumentiert;2025 bleibt reserviert. AP4 ist das nächste vorgeschlagene Paket, nicht automatisch freigegeben.

Separates, festgelegtes Trainingsenvironment erproben, das den vorhandenen Backtest nicht destabilisiert. Zunächst dieselben Merkmale, Zieldefinitionen und Zeitgrenzen wie die Ridge-Referenz verwenden. Kleines neuronales Modell trainieren, Training/Validierung protokollieren, Gewichte und Vorverarbeitung speichern, zeitgestempelte Vorhersagen exportieren und im vorhandenen Backtester bewerten.

**Abnahme:** Ein kleiner Trainingslauf ist wirklich ausführbar; Reload-Inferenz funktioniert; Bericht vergleicht NN, Ridge und Nullvorhersage unter gleichen Bedingungen. Der Meilenstein ist eine funktionierende Trainingskette, nicht ein versprochener Renditevorteil. Zunächst begrenzter Smoke-Test statt langer GPU-Jobs.

### AP4 — Webseite als Modell- und Experimentvergleich

Die HTML-Galerie um strukturierte Run-Ansichten erweitern: Datensatz/Zeitraum, Modell, Parameter, Laufstatus und Detailbericht. Mehrere kompatible Runs auswählen, Konfigurationsunterschiede hervorheben, Trainingskurven und Vorhersagegüte separat von Equity, Drawdown, Kosten und Tradeanzahl darstellen. Zunächst lesend; keine beliebigen Shell-Kommandos oder Pfade aus dem Browser.

**Abnahme:** Andreas kann zwei echte Experimentläufe im Browser vergleichen und sehen, was gleich blieb und was geändert wurde. Erfolgskriterien und Datenbasis sind verständlich, unpassende Vergleiche werden markiert oder abgewiesen.

### AP5 — Komfortables Variieren, Neu-Training und begrenztes Tuning

Aus einem bestehenden Run eine neue Konfiguration ableiten. Lernrate, Modellgröße, Regularisierung, Lookback, Trainingsdauer und Daten-/Featureversion bewusst als unterschiedliche Änderungskategorien behandeln. Einfache manuelle Varianten zuerst, danach kleine deklarative Parametersuchen. Modellparameter, Handelsregeln und Hyperparametersuche unterscheiden. Vor Browser-Start/Abbruch eine echte Job-Schicht mit Warteschlange, Zuständen, erlaubten Parametern, Ressourcen-/Zeitbudget, Zugriffsschutz und CSRF-Schutz bereitstellen; den heutigen lesenden Viewer nicht einfach zu einem Kommando-Endpunkt machen.

**Abnahme:** Ein neuer Run kann mit validierten Parametern gestartet und nachvollziehbar verfolgt werden. Parent, Änderungen, Fehler/Abbruch und Budget bleiben sichtbar. Keine unkontrollierten Trainingsschleifen; Auswahl anhand Validierung, nicht finalem Test.

### AP6 — Deterministische Bilddaten aus derselben Historie

Chartbilder aus denselben kausalen Fenstern und Zielen erzeugen. Renderer, Ausschnitt, Auflösung, Indikatoren, Achsenskalierung und Cache-Schlüssel versionieren. Zukünftige Kerzen, zukünftige Skalierungsstatistiken und sichtbare Ziel-/Ergebnisbeschriftungen ausschließen. Bilder bei Bedarf zwischenspeichern statt unkontrolliert Rohdaten zu vervielfachen. Echte Plattform-Screenshots sind ein zusätzlicher Daten-/Darstellungsfall; ihre Zeitzuordnung und Abweichung von synthetischen Charts separat prüfen.

**Abnahme:** Für jede Beispielprognose ist das tatsächliche Eingabebild mit historischen Kerzen und Zeitpunkt verknüpft. Tests prüfen Daten-/Cache-Trennung und unzulässige Zukunftsinformationen.

### AP7 — Bildmodell in denselben Experimentablauf integrieren

Ein kleines Bildnetz auf dem vereinbarten Ziel trainieren und als neuen Modelltyp in den vorhandenen Runner integrieren. Kein zweiter Backtester und keine separate Kontoführung. Numerische und visuelle Modelle auf gleichen eligible Entscheidungen, Zeiträumen und Kosten vergleichen; verbleibende unterschiedliche Informationsmengen explizit ausweisen.

**Abnahme:** Trainieren, Speichern, Inferenz, Backtest und Webseitenvergleich funktionieren für Zahlen- und Bildmodell. Die Webseite zeigt zu einem Trade das betrachtete Bild und die zugehörige Prognose.

### AP8 — Robustheitsprüfung und kontrollierte Ausweitung

Nach eingefrorener Modell-/Regelauswahl auf unberührten Zeiträumen prüfen; zeitlich rollierende Prüfungen und mehrere Seeds mit begrenztem Budget ergänzen. Ergebnisse nach Sitzung/Marktphase und Kostenannahmen aufschlüsseln. Erst dann Datenumfang, automatische Suchen, weitere Instrumente oder Trainingshardware nach tatsächlichem Engpass ausweiten.

**Abnahme:** Aussagekräftiger Abschlussbericht mit Grenzen und negativer Evidenz; kein Finaltest-Tuning. Ergebnisse sind Forschungsbefunde, keine Profitabilitätsgarantie.

## Priorisierung

Der erste nutzbare vertikale Durchstich ist AP1 -> AP2 -> AP3 -> AP4. Danach ist tabellarisches Training mit Browservergleich verfügbar, ohne auf den Bildzweig warten zu müssen. AP5 verbessert die Iteration, AP6/AP7 erweitern die Eingabemodalität. Zeitgrenzen, Datenleakage-Schutz und Testdisziplin gelten ab AP1, nicht erst in AP8. Keine Dauer- oder Hardwareversprechen vor Dateninventar und Smoke-Test.

## Effiziente Zusammenarbeit mit dem Agenten

- Pro Auftrag ein abgrenzbares Paket mit Ziel, Nicht-Zielen, Abnahmetest, Änderungsfreigabe und Ressourcenbudget. Andreas entscheidet Forschungsfrage, Priorität und Trade-offs; der Agent setzt um, testet und legt Belege vor.
- Projektziel/Roadmap und Entscheidungen im Repo pflegen; aktueller Arbeitsstand in einer kurzen Statusdatei mit fertigen/offenen Punkten, Testkommandos und nächstem Schritt. Geheimnisse gehören nicht in Kontextdateien. Keine Erwartung unbegrenzter Chat-Erinnerung.
- Ein aktiver Implementierungsstrang; unabhängige Subagenten für abgegrenzte Aufgaben/Reviews. Änderungen auf getrennten Dateien oder Worktrees. Nicht mehrere unkoordinierte Editoren auf denselben Branch loslassen.
- Umsetzung lokal mit Tests zuerst, Gesamttests, kurzem unabhängigem Review und nachvollziehbaren Artefakten. Lokale Arbeit, Commit, Push und Deployment haben getrennte Freigaben.
- Lange Trainingsläufe als dauerhafte Jobs mit Logs, Status, Stop und Ressourcenlimits ausführen, nicht an eine offene Chat- oder Subagent-Sitzung binden. Nur freigegebene Laufzeit/Kosten verbrauchen.
- Benötigte Voraussetzungen: ein zugänglicher Datenordner, Metadaten/Beispieldatei, erster Markt/Zeithorizont, Hardware-/Speicher-/Trainingsbudget und Rückmeldung anhand konkreter Berichte. Administrative Installationen bei tatsächlichem Bedarf; kein pauschaler Root-Zugriff erforderlich.
- Aktueller Komfort-Blocker: automatisierte Browserprüfung scheitert an fehlender libatk-1.0.so.0. Gezielte Browser-Abhängigkeiten durch einen Administrator könnten visuelle UI-Prüfung erleichtern; das ist keine Voraussetzung für Training oder HTTP-Hosting.

## Historische Daten: tatsächlich geprüfter Bestand

Im aktuellen Checkout gefunden: drei versionierte synthetische CSV-Fixtures unter tests/fixtures. Kein datasets/-Verzeichnis, keine echten historischen CSV-/Parquet-Datensätze und keine gespeicherten Modellgewichte in den geprüften Dateiformaten. Die Regeln ignorieren *.csv mit Ausnahme der Test-Fixtures sowie /datasets/.

Die bestehenden Dokumente beschreiben frühere lokale GER30-Daten und Ergebnisse; die referenzierten datasets/-Artefakte fehlen hier. Diese Ergebnisse sind nicht durch neu gelaufene Experimente auf diesem Server bestätigt.

In der Git-Historie existiert jedoch ein kleiner früherer Datenausschnitt:
- Commit e233b82 führt backtest/data/btc/5min/dev.csv ein.
- Commit 918b6b0 verschiebt ihn nach backtester/data/btc/5min/dev.csv.
- Commit 40b3bd6 entfernt ihn.
- Vorhandener Blob: 15.017 Bytes, 157 Datenzeilen; Zeitstempel laut Datei von 2017-08-17 04:00:00 bis 2017-08-17 17:00:00, ohne explizite Zeitzone. Instrument-/Quellzuordnung aus dem Pfad allein nicht als fachlich verifiziert behandeln.

Der kleine Ausschnitt ist aus Git wiederherstellbar, wurde aber nicht ungefragt in den Checkout zurückgeschrieben. Größere ursprüngliche Datensätze müssen separat bereitgestellt oder auf dem alten Speicher lokalisiert werden.
