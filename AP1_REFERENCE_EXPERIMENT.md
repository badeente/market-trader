# AP1 – Vorschlag für das erste Referenzexperiment

**Status: Von Andreas freigegeben, implementiert, ausgeführt und technisch abgenommen. Bericht intern veröffentlicht; visuelle Browserautomatik ist umgebungsbedingt blockiert. Ergebnisse und Reproduktion: [AP1_RESULTS.md](AP1_RESULTS.md).**

## Ziel und Nicht-Ziele

Ein reproduzierbarer Referenzdatensatz und ein überprüfbarer Baseline-Lauf sollen den späteren tabellarischen und visuellen Modellen dieselbe fachliche Grundlage geben. AP1 besteht nicht erst dann, wenn eine Strategie Gewinne erzielt. Nicht enthalten: neuronales Training, Parametersuche, Live-Trading, Hebel, eine neue Backtest-Engine oder eine große Webanwendung.

## 1. Datengrundlage

- Neue Serie: Dukascopy `deuidxeur`, vom Downloader als Germany 40 Index bezeichnet.
- Primär: die bereits heruntergeladenen 15-Minuten-Bid-Kerzen, 2021–2025, ISO UTC.
- Rohquelle: `var/market-trader/raw/dukascopy-deuidxeur-2021-2025-v1/` außerhalb des Checkouts. Hashes und Abrufparameter sind in deren Manifest enthalten.
- Die 5-Minuten-Datei bleibt Kontrollquelle. Sie ist kein unabhängiger Testdatensatz.
- Die alten Exporte unbekannter Herkunft bleiben separat und werden nicht beigemischt.
- Volumen wird gespeichert, zunächst aber nicht als Modellmerkmal verwendet; seine physikalische Bedeutung ist für dieses Produkt nicht unabhängig bestätigt.
- Quellbedingungen für die interne Forschungsnutzung dokumentieren; keine öffentliche Datenveröffentlichung.

## 2. Forschungsfenster und Datenqualität

**Vorschlag:** Montag bis Freitag, 09:00–17:30 `Europe/Berlin`; Speicherung/Verträge weiterhin in UTC. Die Umrechnung verwendet eine benannte Zeitzone, keinen festen ganzjährigen UTC-Offset.

Dieses Fenster ist eine bewusst gewählte Forschungseinschränkung, keine Behauptung über offizielle Dukascopy-Produkt- oder Börsenhandelszeiten. Der erste Referenzlauf handelt nicht über Nacht. Ganze fehlende Tage werden als ohne Daten erfasst und nicht automatisch mit erfundenen Feiertagsgründen versehen. Die Herkunft und der tatsächliche Produktkalender bleiben getrennt von unserem Forschungsfilter.

Für die erste abgeleitete Referenzversion werden nur beobachtete Tage mit allen 34 erwarteten 15-Minuten-Kerzen dieses Fensters zugelassen. Partielle Tage werden vollständig ausgeschlossen und mit Datum, fehlenden Starts und Grundcode protokolliert. Kein Forward-Fill und keine Auswahl anhand von Renditen. Die Beschränkung auf vollständige Tage kann Auswahlverzerrungen verursachen und wird im Bericht sichtbar gemacht; sie zertifiziert keine Anbieter-Vollständigkeit.

**Bereits ausgeführte Strukturprüfung an den CSVs, keine Modellbewertung:**
- 2021: 258 vollständige beobachtete Forschungstage.
- 2022: 258.
- 2023: 256.
- 2024: 258 vollständige Tage; 19. Juli 2024 ist im Fenster partiell (29 Kerzen).
- 2025: 257 vollständige Tage; 20. Januar 2025 ist im Fenster partiell (25 Kerzen).

Die zwei partiellen Tage werden nicht ohne weitere Quelle als fehlerhafte Exporte bezeichnet. Der Ausschluss ist eine transparente Entscheidung für diesen Pilot, keine universelle Marktkalenderregel. Auch vollständig ausbleibende Werktage werden im Qualitätsmanifest einzeln erfasst. Die außerhalb des Filters beobachteten 444 unvollständig vergleichbaren 5m/15m-Blöcke werden nicht pauschal zu Lücken im Referenzfenster erklärt.

## 3. Zeitliche Trennung

- Training: 1. Januar 2021 einschließlich bis 1. Januar 2024 ausschließlich.
- Validierung: 1. Januar 2024 einschließlich bis 1. Januar 2025 ausschließlich.
- Reservierter Projekt-Testabschnitt: 1. Januar 2025 einschließlich bis 1. Januar 2026 ausschließlich.

Jahresgrenzen sind explizite UTC-Grenzen; Forschungsfenster werden nach lokaler Berliner Zeit bestimmt. Keine Zufallsmischung über die Zeit. Zielhorizonte dürfen die Splitgrenzen nicht überschreiten; Vorverarbeitung wird nur am Training angepasst.

**AP1 führt Modell- und Handelsauswertung ausschließlich auf 2024 aus.** 2025 bleibt bis zur eingefrorenen Auswahl reserviert. Eine reine Datenqualitäts-/Zeitstempelprüfung des gesamten Rohbestands ist davon getrennt. Nach Einsicht in finale Testmetriken wird derselbe Zeitraum nicht weiter als unberührter Test für neue Entscheidungen verkauft. Gegebenenfalls später einen neuen Abschlusszeitraum beschaffen.

## 4. Vorhersageaufgabe

- Lookback: acht abgeschlossene 15-Minuten-Kerzen, also zwei Stunden, vollständig innerhalb derselben Forschungssitzung.
- Ziel: Bruttorendite über 30 Minuten vom Open der nächsten Einstiegskerze zum Open zwei 15-Minuten-Intervalle später.
- Beispiel: Entscheidung am Schluss um 11:00, Referenzeinstieg am nächsten Open um 11:00, Referenzausstieg am Open um 11:30. Schluss und folgende Eröffnung dürfen denselben Zeitstempel teilen.
- Zielwert: `(exit_reference_open - entry_reference_open) / entry_reference_open`.
- Merkmale bleiben die vorhandenen vier: letzte Close-Rendite, Close-Rendite über das Fenster, letzte High-Low-Spanne relativ zum Close und Close-Abstand zum Fenstermittel.

Damit wird die bisherige Abweichung zwischen Close-basiertem Label und Open-basiertem Zeitausstieg bewusst beseitigt. Der bestehende Close-Zielvertrag wird nicht stillschweigend umdefiniert: neuer Zielname und versionierte Artefakte, klare Ablehnung inkompatibler Modelle/Exporte, Rückwärtsverträglichkeit oder explizite Legacy-Abgrenzung durch Tests.

Der Zielaufbau benötigt zusätzlich die Kerze, deren Open den Ausstieg belegt. Deren `available_at` wird beim Purging konservativ berücksichtigt; eine erst später vollständig verfügbare Kerze darf nicht als bereits verfügbares Trainingslabel gelten. Ein Einstieg ist nur zulässig, wenn der geplante Ausstiegs-Open noch innerhalb der definierten Sitzung liegt. Kein erfundener Open am Sitzungsende.

Unter diesen vorgeschlagenen Regeln ergeben sich strukturell 24 mögliche beschriftete Entscheidungen pro vollständigem Tag: 18.528 im Training, 6.192 in der Validierung, 6.168 im reservierten Test. Das sind aus der Abdeckung berechnete potenzielle Beispiele, noch keine erzeugten/abgenommenen Trainingsartefakte und keine unabhängigen Marktbeobachtungen.

## 5. Baselines und Handelsabbildung

- Statistische Referenzen: Nullrendite-Vorhersage und vorhandene Ridge-Regression mit festem Alpha 0.1; Skalierung ausschließlich auf 2021–2023 fitten.
- Ridge-Handelsregel: Long bei prognostizierter Bruttorendite strikt größer null; sonst kein neuer Einstieg. Keine Schwellenoptimierung in AP1. Ein positives Bruttosignal ist ausdrücklich kein Nachweis positiver Nettorendite.
- Einfache nicht gelernte Kontrollregel: Long, wenn SMA über die letzten vier Closes über SMA der letzten acht Closes liegt; sonst kein Einstieg. Kein zusätzlicher Crossover-/Trendparameter und keine Suche nach besseren Perioden.
- Beide Handelswege: dieselben zulässigen Entscheidungszeitpunkte, dieselbe feste Menge, maximal eine offene Long-Position, Einstieg nächstes Open und Zeitausstieg nach 30 Minuten. Kein Short, Stop/Target oder Overnight-Halten in diesem ersten Referenzexperiment. Die Kontrollregel ist keine behauptete profitable Strategie.
- Die Nullprognose dient dem Prognosefehler-Vergleich; eine Nicht-Handeln-Kontrollkurve kann separat als flache Handelsreferenz erscheinen.

Die zeitgesteuerte Kontrollregel ist ein kleiner, noch zu implementierender Adapter im bestehenden Signal-/Ausführungspfad; keine zweite Buchhaltung. Frühere SMA-Crossover-Ergebnisse mit anderem Ausstieg werden nicht als gleichwertiger Modellvergleich ausgegeben.

## 6. Kosten als ausdrücklich hypothetisches Forschungsprofil

Noch keine bestätigten Konditionen eines konkreten Brokers. Für den ersten technischen Vergleich wird vorab eingefroren vorgeschlagen:

- Voller symmetrisch modellierter Spread: 2.0 Indexpunkte.
- Slippage: 0.1 Punkt je Ausführung; separate Kommission: 0.
- Eine Einheit, Kontraktmultiplikator 1 als Forschungsannahme; keine Wiederanlage oder risikobasierte Größenoptimierung.
- Preisraster 0.001 als Rechen-/Datenraster, nicht als bestätigter handelbarer Produkttick.
- Interne Startmittel 100000 Q, damit eine Einheit im vorliegenden nominalbesicherten Simulator finanzierbar ist. Q sind synthetische Buchungseinheiten, keine Empfehlung für Kontogröße oder Euroeinsatz.

Vor weiterer Rasterrundung sind damit 2.2 Punkte Reibung pro vollständigem Trade modelliert. Die Bid-Quelle plus symmetrische Offsets ist ein synthetisches Kostenexperiment, keine Rekonstruktion historischer Ask-Fills. Ein-/Ausstiege werden im vorhandenen Ledger berechnet, niemals nachträglich nur im Bericht korrigiert.

Bericht: Bruttobewegung, modellierte Kosten und Nettoergebnis getrennt; Punkte je Einheit im Vordergrund. Keine Aussage über tatsächlich handelbare Euro-Gewinne. Ein kostenfreier Kontrolllauf kann die Buchhaltung prüfen, ist aber kein Beleg wirtschaftlicher Nutzbarkeit. Ein bestätigtes Produktprofil ersetzt diese Annahmen später als eigene Version.

## 7. Konkrete AP1-Abnahme

1. Derselbe Rohbestand + dokumentierte Konfiguration erzeugen dieselbe kanonische Datensatzkennung.
2. UTC-Konvertierung, Berliner Sommer-/Winterzeit, Sitzungsgrenzen und Ausschlussmanifest sind mit kleinen Beispielen getestet.
3. Trainings-/Validierungsgrenzen und Labelverfügbarkeit sind getestet; einige Open-to-Open-Labels sind separat nachgerechnet.
4. Rohdaten und bestehende Exporte bleiben unverändert; Features/Modelle/Ergebnisse referenzieren eindeutige Versionen.
5. Ridge, Nullprognose und Kontrollregel sind mit den eingefrorenen Parametern tatsächlich ausgeführt, ohne Tuning.
6. Modellfehler (MAE/RMSE) und Handelsauswertung (Punkte/P&L, Kosten, Drawdown, Trades, Ergebnisse nach Jahr/Monat der Validierung) sind getrennt dargestellt und rechnerisch nachvollziehbar.
7. Ein Bericht des Referenzlaufs 2024 kann über die bestehende interne Webseite geöffnet werden. Veröffentlichung erst nach erfolgreicher Validierung; Rohdaten bleiben privat.
8. Die reservierten Testmetriken 2025 werden in AP1 nicht berechnet oder angezeigt.
9. Alle bestehenden und neuen Pflichtprüfungen bestehen; Graphify und Projektdokumentation sind aktualisiert. Ein dokumentierter Offline-Aufruf reproduziert den Referenzlauf.

**AP1 gilt auch bei enttäuschender Ridge-/Regelperformance als erfolgreich, wenn diese technische/fachliche Abnahme erfüllt ist. Danach folgt AP2 für Run-IDs, Konfigurationssnapshots und komfortable Wiederholungen.**

## Aktueller Änderungsumfang

Die oben beschriebenen Forschungsparameter wurden freigegeben und als feste AP1-Version implementiert. Import, versionierter Zielvertrag, Ridge/Nullreferenz und SMA-Zeitkontrolle sind mit echten Trainings-/Validierungsdaten ausgeführt; 2025-Modellmetriken bleiben gesperrt. Der aktuelle Prüf-, Reproduktions- und Veröffentlichungsstand steht in [AP1_RESULTS.md](AP1_RESULTS.md). Keine Freigabe für AP2, neuronale Modelle, Live-Trading oder öffentliche Veröffentlichung daraus ableiten.
