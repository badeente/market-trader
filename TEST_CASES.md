# Testfälle vor der Implementierung

Stand 2026-11-12 · 69 fachliche Fälle und 420 bestandene Prüfungen

## 1. Zweck und Reihenfolge

Diese Testfälle konkretisieren die [ROADMAP.md](ROADMAP.md). Sie beschreiben den
angestrebten Vertrag und nicht das Verhalten des Legacy-Codes. Einige Fälle
richten sich ausdrücklich gegen die dort dokumentierten Fehler.

Reihenfolge: Roadmap → fachliche Testfälle → Schnittstellendetails und ausführbare
Tests für das jeweilige Arbeitspaket → Produktionsimplementierung. Nach der
fachlichen Erstfassung wurde das erste Testpaket mit Datentypen und aufrufbaren
Funktionshüllen angelegt, rot ausgeführt und anschließend implementiert. Der
aktuelle, weiterhin begrenzte Umfang steht in Abschnitt 15.

Die folgenden Tabellen definieren das Sollverhalten. Der aktuelle
Automatisierungsstand steht in Abschnitt 15. Ein Fall wird erst nach
tatsächlicher erfolgreicher Ausführung als bestanden markiert. Offene
Profilentscheidungen und nicht unterstützte Fähigkeiten dürfen nicht durch
pauschales Überspringen als erledigt erscheinen.

Für die spätere Automatisierung werden Fall-ID, Testdatei/-funktion und Ergebnis
zugeordnet. Konkrete öffentliche API-Namen werden vor dem jeweiligen Arbeitspaket
festgelegt; die fachlichen Erwartungen sind unabhängig von einer Bibliothek.

## 2. Gemeinsame synthetische Testgrundlage

### Referenzprofil R0 für D-03

Der [Backtest-Vertrag](BACKTEST_CONTRACT.md) legt das synthetische Profil R0 und
seine Engine verbindlich fest. Die folgenden Beispiele definieren die Sollwerte
für M3. D-01 bleibt für die Wahl eines realen Markts offen.

- Instrument `TEST`, Abrechnungswährung `Q`, Kontraktmultiplikator 1; Preise und
  Mengen werden ausdrücklich angegeben. Startkapital: 1.000 Q.
- Grundsätzlich Menge 1, Gebühren und Spread/Slippage 0, sofern ein Fall nichts
  anderes angibt. Synthetische Short-Fälle erhalten ausdrücklich ein Profil mit
  Short-Unterstützung; daraus folgt keine Unterstützung eines realen Produkts.
- Daten haben `open_time`, `close_time`, `available_at` in UTC sowie OHLCV,
  Instrument und Intervall. Verfügbarkeit ist frühestens am Kerzenende.
- Ein Signal nach Schluss von Kerze t darf frühestens an der nächsten
  Kerzeneröffnung ausgeführt werden. Ohne nächste Kerze gibt es keinen Fill.
- Stop-Market-Orders einer schon bestehenden Position werden bei Berührung zum
  Stop ausgeführt; bei einer Eröffnung jenseits des Stops zum schlechteren
  Eröffnungspreis. Konfigurierte Ausführungskosten kommen zusätzlich hinzu.
- Treffen Stop und Ziel innerhalb derselben Kerze einer bereits offenen Position,
  gilt bei unbekannter Intrabar-Reihenfolge der Stop zuerst. Die Mehrdeutigkeit
  wird im Ergebnis ausgewiesen. OHLC-Daten belegen keine Intrabar-Reihenfolge.
- Standard am Laufende: offene Position zum letzten verfügbaren Schlusskurs
  bewerten, nicht automatisch schließen. Eine ausdrücklich konfigurierte
  Endschließung ist ein separater Fall.
- `HOLD` bedeutet unverändert; `EXIT` schließt eine bestehende Position. Die Namen
  sind fachliche Bezeichner, noch keine festgelegten Python-Enum-Namen.
- Numerische Vergleiche erlauben höchstens 1e-8 Q absolute Rechentoleranz für die
  hier angegebenen Beispiele. Marktbezogene Rundung erfolgt nach explizitem Raster;
  auf Anzeigeformat gerundete Werte sind keine Berechnungsgrundlage.
- Risikobudgets beziehen sich auf den geplanten Verlust bei angenommener
  Stop-Ausführung. Sie garantieren keine Verlustobergrenze bei Kurslücken.

### Geplante Fixtures

| Fixture | Inhalt und Zweck |
| --- | --- |
| F-ACCOUNT | Synthetische Ausführungen mit bekannten Mengen, Ein-/Ausstiegspreisen und Kosten für unabhängige Kontorechnungen. |
| F-BARS | Kleine UTC-Kerzenfolgen mit explizitem Intervall und Verfügbarkeit; bei Bedarf inklusive Lücke oder Konflikt. |
| F-RESAMPLE | Fünf 1-Minuten-Kerzen ab 00:00 UTC: Open 100, 101, 102, 103, 104; High 101, 102, 103, 104, 105; Low 99, 100, 101, 102, 103; Close 100,5 bis 104,5 in Schritten von 1; Volumen 1,5 / 2 / 0 / 3 / 0,5. |
| F-CAUSAL | Historischer Präfix mit zwei unterschiedlichen Fortsetzungen, um Einfluss zukünftiger Daten zu erkennen. |
| F-PROVIDER | Lokaler Anbieterersatz mit Seiten, Überlappungen und vorgegebenen Fehlern; kein externer Netzwerkzugriff. |
| F-PREDICTOR | Deterministischer Modellersatz mit vorgegebenen Vorhersagen und nachvollziehbaren Aufrufen; keine Gewichte herunterladen. |

## 3. Einrichtung und Modulgrenzen — M2

| ID | Gegeben / Aktion | Erwartetes Ergebnis |
| --- | --- | --- |
| SET-001 | Frischer Checkout; dokumentierte Einrichtung mit gewählter Python-Version und Lockfile. | Installation gelingt; Versionsstand ist eindeutig; keine privaten Dateien oder vorhandenen Datensätze erforderlich. |
| SET-002 | Nur Kernabhängigkeiten installiert; Import von Domain, Dateiimport und späterem CLI-Grundgerüst. | Kein Import von PyTorch, torchvision oder Desktop-Automatisierung; kein Netzwerk, Training, Backtest oder GUI als Import-Nebeneffekt. |
| SET-003 | Testlauf mit synthetischen Fixtures und ohne Netzwerk. | Die jeweils implementierten Kernfälle laufen ohne API-Schlüssel, GPU, echten Anbieter oder Modell-Download. |
| SET-004 | Dieselbe Beispielkonfiguration von einem anderen Arbeitsverzeichnis starten. | Pfade werden nach dokumentierter Konfigurations-/Projektregel aufgelöst; Ergebnis ist fachlich identisch. Gesamter Beispielablauf ist erst in M3 abnehmbar. |

## 4. Import, Datenqualität und Verarbeitung — M2

| ID | Gegeben / Aktion | Erwartetes Ergebnis |
| --- | --- | --- |
| DAT-001 | Kerzen mit explizitem Schema und Volumen 0,125 importieren. | Werte bleiben erhalten; keine Umwandlung des Volumens in eine ganze Zahl; Instrument und Intervall bleiben zugeordnet. |
| DAT-002 | Dieselbe Zeit einmal als 2026-01-01T01:00:00+01:00, einmal als 2026-01-01T00:00:00Z normalisieren. | Identischer UTC-Zeitpunkt. Ein Zeitstempel ohne Zeitzone wird ohne ausdrücklich konfigurierte Zeitzone abgewiesen. |
| DAT-003 | Fehlender Schlusskurs, NaN/Infinity, negatives Volumen oder High unter max(Open, Close) bzw. Low über min(Open, Close). | Verständlicher Datenfehler mit betroffenem Datensatz/Datenschlüssel; kein als erfolgreich gemeldeter Import. |
| DAT-004 | Identische Kerze zweimal; danach gleicher Schlüssel mit abweichendem Schlusskurs. | Identische Dublette wird nachvollziehbar dedupliziert; widersprüchliche Dublette wird abgewiesen. Schlüssel umfasst Instrument, Intervall und Kerzenbeginn. |
| DAT-005 | Unsortierte Kerzen und eine fehlende Kerze während der aktiven Zeit des synthetischen Kalenders. | Stabile chronologische Normalisierung; Lücke wird gemeldet. Strikter Backtest lehnt sie ab; keine stillschweigend erfundene Kerze. |
| DAT-006 | Eine Lücke liegt ausschließlich außerhalb einer ausdrücklich konfigurierten Handelssitzung. | Keine falsche Meldung einer fehlenden aktiven Kerze; verwendeter Kalender steht in den Metadaten. |
| DAT-007 | F-RESAMPLE zu einem 5-Minuten-Intervall [00:00, 00:05) zusammenfassen. | Open 100, High 105, Low 99, Close 104,5, Volumen 7; Ende und früheste Verfügbarkeit 00:05 UTC. |
| DAT-008 | Zusätzlich eine Kerze mit Beginn 00:05; andere Folge endet vor Abschluss des 5-Minuten-Intervalls. | 00:05 gehört zum nächsten Intervall. Unvollständige Aggregate werden nicht als vollständige Kerzen ausgegeben; Behandlung wird protokolliert. |
| DAT-009 | Dieselben Rohdaten und dieselbe Verarbeitung zweimal ausführen; danach einen Kurs oder die Resampling-Konfiguration ändern. | Wiederholung liefert gleiche normalisierte Datenkennung. Geänderte Inhalte/Verarbeitung sind unterscheidbar; Rohdaten bleiben unverändert. |
| DAT-010 | Ein Datensatz enthält mehrere Instrumente oder Intervalle. | Auswahl und Schlüssel bleiben getrennt; keine Vermischung innerhalb eines Strategie-Datenfensters. |

## 5. Historische Verfügbarkeit und Fenster — M2/M3

| ID | Gegeben / Aktion | Erwartetes Ergebnis |
| --- | --- | --- |
| CAU-001 | Strategie benötigt drei Kerzen; es stehen zunächst zwei, danach drei, danach vier abgeschlossene Kerzen bereit. | Kein vorzeitiger Strategieaufruf; anschließend Fenster 1–3 und 2–4 in korrekter Reihenfolge. Ende der Daten ist ein eigener, normaler Zustand. |
| CAU-002 | Entscheidung um 00:05; eine Kerze endet um 00:05, ist laut Quelle aber erst um 00:06 verfügbar. | Die Kerze ist um 00:05 nicht sichtbar und darf erst ab 00:06 in Entscheidungen eingehen. |
| CAU-003 | F-CAUSAL einmal mit günstiger, einmal mit ungünstiger zukünftiger Fortsetzung auswerten. | Merkmale, Bilder und Signale des gemeinsamen Präfixes sind unverändert. Gemessen wird in derselben festgelegten Umgebung. |
| CAU-004 | Quelle oder Merkmalsberechnung liefert einen technischen Fehler. | Fehler wird als fehlgeschlagener Lauf/Verarbeitungsschritt ausgewiesen; nicht als reguläres Datenende oder neutrales Signal verborgen. |
| CAU-005 | Strategie erhält ein historisches Fenster und versucht seine Werte zu verändern. | Verbindlicher Quelldatensatz und spätere Fenster bleiben unverändert; der Schutz ist durch Kopie oder andere definierte Isolation umgesetzt. |

## 6. Handelssignale und Erweiterbarkeit — M3/M4

| ID | Gegeben / Aktion | Erwartetes Ergebnis |
| --- | --- | --- |
| SIG-001 | HOLD einmal ohne Position und einmal mit offener Long-Position; anschließend EXIT mit Position. | HOLD erzeugt keine Order und schließt nichts. EXIT erzeugt genau eine Schließorder. Kein widersprüchliches numerisches Aktionsmapping zwischen Modulen. |
| SIG-002 | Strategie verlangt Einstieg Short, Instrumentprofil unterstützt nur Long. | Nachvollziehbare Ablehnung ohne Fill; Short wird nicht als Long interpretiert. |
| SIG-003 | Zwei registrierte Teststrategien mit unterschiedlichen gültigen Parametern per Konfiguration auswählen. | Richtige Strategie/Parameter werden verwendet; unbekannter Name oder ungültige Parameter scheitern vor dem Lauf mit verständlicher Meldung. |
| SIG-004 | Neue Teststrategie registrieren und über dieselbe CLI/Engine ausführen. | Integration gelingt ohne Änderung an Risiko, Kontoberechnung oder Engine; Signalvertrag wird eingehalten. |
| SIG-005 | Zustandshaltende Strategie auf Datensatz A, anschließend B; B zusätzlich separat starten. | B liefert im kombinierten Startablauf dieselben fachlichen Ergebnisse wie separat; keine Historie oder Position aus A wird übernommen. |
| SIG-006 | Regel-, Zahlen- und Bildstrategie verwenden vorgegebene identische Signale sowie gleiche Daten und Kosten. | Identische Ausführungen und Kontoverläufe. Unterschiedliche Eingabedarstellung verändert keine Ausführungsregeln. |

## 7. Kontoführung und Renditen — M3

Diese Zahlen sind unabhängig von einer Engine herzuleiten. Falls nicht anders
angegeben, ist jede Position am Ende des Falls geschlossen.

| ID | Gegeben / Aktion | Erwartetes Ergebnis |
| --- | --- | --- |
| ACC-001 | Keine Orders bei wechselnden Kursen. | Kapital und Equity bleiben 1.000 Q; keine Ausführungen oder Trades, Gesamtrendite 0. |
| ACC-002 | Long: 1 Einheit zu 100 kaufen, zu 110 schließen. | Realisierter Gewinn +10 Q, Endkapital 1.010 Q, Kontorendite +1 %. |
| ACC-003 | Long: 1 Einheit zu 100 kaufen, am Stop zu 98 schließen. | Verlust −2 Q, Endkapital 998 Q, Kontorendite −0,2 %. Kursrendite −2 % ist davon zu unterscheiden. |
| ACC-004 | Short-fähiges Profil: 1 Einheit zu 100 eröffnen und zu 90 schließen; zweiter Lauf zu 102 schließen. | Erster Lauf +10 Q / 1.010 Q; zweiter Lauf −2 Q / 998 Q. Stop-Verlust wird niemals allein durch Vorzeichenumkehr zum Gewinn. |
| ACC-005 | Long: 2 Einheiten zu 100, Ausstieg 110; Gebühr pro Seite 0,001 des jeweiligen Nominals. | Bruttogewinn 20 Q, Gebühren 0,20 + 0,22 = 0,42 Q, Nettogewinn 19,58 Q, Endkapital 1.019,58 Q. |
| ACC-006 | Long: 1 Einheit zu 100, letztes verfügbares Close 105; Standardprofil lässt Position offen. | Equity 1.005 Q, realisierter P&L 0, unrealisierter P&L +5 Q, eine offene Position; keine erfundene Exit-Ausführung. |
| ACC-007 | Derselbe Verlauf mit ausdrücklich aktivierter Endschließung zum letzten Close und Kosten 0. | Genau eine Exit-Ausführung zu 105; realisierter P&L +5 Q, unrealisierter P&L 0, Equity 1.005 Q. |
| ACC-008 | Equity 1.000 → 1.100 → 990 → 1.089. | Gesamtrendite +8,9 %; maximaler Drawdown −10 %. Die Renditen der Abschnitte werden nicht einfach addiert. |
| ACC-009 | Zwei vollständige Läufe auf frischen Instanzen mit denselben Ausführungen. | Identische Kontowerte; keine globalen oder klassenweiten Reste des ersten Laufs. |

## 8. Ausführungen und Kostenannahmen — M3

| ID | Gegeben / Aktion | Erwartetes Ergebnis |
| --- | --- | --- |
| EXE-001 | Einstiegssignal nach Close 100; nächste Kerze öffnet bei 103. | Marktorder wird zu 103 ausgeführt, nicht rückwirkend zu 100; Kosten 0. |
| EXE-002 | Einstiegssignal erst nach der letzten verfügbaren Kerze. | Kein Fill; nicht ausgeführte Order wird ausgewiesen. Kein Zugriff auf eine nicht vorhandene nächste Kerze. |
| EXE-003 | Long bereits zu 100 offen, Stop 98; nächste Kerze O=100, H=101, L=97, C=99. | Stop-Market-Fill zu 98, Nettoverlust −2 Q. Schließung wird nicht auf die übernächste Kerze verschoben. |
| EXE-004 | Short bereits zu 100 offen, Stop 102; nächste Kerze O=100, H=103, L=99, C=101. | Stop-Market-Fill zu 102, Nettoverlust −2 Q. Long- und Short-Bedingungen sind richtungsgerecht. |
| EXE-005 | Long bereits zu 100 offen, Stop 98; nächste Kerze O=95, H=97, L=94, C=96. | Fill zu 95, Verlust −5 Q; kein garantierter Fill zu 98. Abweichung zum geplanten Stop-Risiko ist sichtbar. |
| EXE-006 | Long bereits zu 100 offen, Stop 98, Ziel 104; nächste Kerze O=100, H=105, L=97, C=101. | Unter R0 Stop-Fill zu 98, Verlust −2 Q und ausgewiesene Intrabar-Mehrdeutigkeit; kein frei erfundener Gewinn. |
| EXE-007 | Long bereits zu 100 offen, Ziel 104; O=100, H=105, L=99, C=103. Short-Variante mit Ziel 96 und O=100, H=101, L=95, C=97. | Long-Fill zu 104 bzw. Short-Fill zu 96; jeweils Gewinn +4 Q bei Menge 1 und Kosten 0. |
| EXE-008 | Referenzpreise Kauf 100, Verkauf 110; konfigurierte feste ungünstige Slippage 0,5 Q je Seite, Menge 1. | Tatsächliche Preise 100,5 und 109,5; Brutto-P&L +9 Q; Slippage nicht zusätzlich ein zweites Mal vom Konto abziehen. |
| EXE-009 | Synthetisches Kostenprofil: voller Spread 0,2 Q um den Referenzpreis, Menge 1; Kaufreferenz 100, Verkaufsreferenz 110. | Kauf zum Ask 100,1, Verkauf zum Bid 109,9; P&L +9,8 Q ohne weitere Kosten. Das Profil ist eine ausdrückliche Modellannahme. |
| EXE-010 | Marktbezogene Konfiguration verlangt nicht unterstützte Kosten oder Fähigkeiten, etwa Funding/Margin-Regeln. | Lauf wird vor einer als gültig ausgegebenen Auswertung abgewiesen. Fehlende Kosten werden nicht stillschweigend als 0 behandelt. |

Der R0-Vertrag umfasst die synthetischen Kostenprofile aus EXE-008 und EXE-009.
Ihre Implementierung gilt nach bestandenem Ausführungsfall als unterstützt.

## 9. Risikoprüfung und Einheiten — M3

| ID | Gegeben / Aktion | Erwartetes Ergebnis |
| --- | --- | --- |
| RSK-001 | Equity 1.000 Q, Risikobudget 0,01, geplanter Long-Einstieg 100, Stop 98; Kosten 0, Mengenraster 1, ausreichende Mittel. | Geplantes Budget 10 Q; Menge 5; geplantes Stop-Risiko 10 Q. Positionsgröße wird nicht pauschal auf 1 gesetzt. |
| RSK-002 | Wie RSK-001, Mengenraster 0,3 und Mindestmenge 0,3. | Menge auf 4,8 abrunden; geplantes Stop-Risiko 9,6 Q. Keine Aufrundung auf 5,1. |
| RSK-003 | Gültiges Signal, aber unzureichendes Kapital oder Menge nach Rundung unter Mindestmenge. | Nachvollziehbare Ablehnung ohne Fill und ohne Kontoveränderung; keine implizite Hebelfinanzierung. |
| RSK-004 | Stop fehlt bei einer Stop-basierten Größenregel, Stop entspricht Einstieg oder liegt für Long darüber bzw. für Short darunter. | Regel meldet ungültigen Risikovertrag; keine Division durch 0, NaN-Menge oder verdeckter Standardstop. |
| RSK-005 | Long-Einstieg 100; explizites Feld für relativen Stop-Abstand einmal 0,005, einmal 0,02. | Stoppreise 99,5 bzw. 98. Eine explizite Umrechnung von 2 Prozentpunkten liefert 0,02; keine Größenordnungsheuristik. |
| RSK-006 | Größenberechnung bei gleichem Budget mit zusätzlichen nichtnegativen Gebühren/Slippage-Annahmen. | Geplantes Gesamtrisiko einschließlich Kosten bleibt innerhalb des Budgets oder Order wird abgelehnt. Annahmen werden ausgewiesen; Gap-Verluste bleiben eine separate mögliche Abweichung. |

## 10. Ergebnisberichte und Reproduzierbarkeit — M3/M4

| ID | Gegeben / Aktion | Erwartetes Ergebnis |
| --- | --- | --- |
| REP-001 | Vollständiger synthetischer Lauf mit Gewinn, Verlust, Gebühren und offener Endposition. | Bericht lässt sich aus Ausführungen, realisiertem/unrealisiertem P&L und Kontoverlauf unabhängig rekonstruieren; keine doppelte Kosten- oder Gewinnzählung. |
| REP-002 | Gleiches Experiment zweimal in festgelegter Softwareumgebung ausführen. | Fachliche Ergebnisse identisch. Lauf-ID, Dateipfad und Erstellzeit dürfen abweichen; diese organisatorischen Felder werden getrennt verglichen. |
| REP-003 | Datensatz, Strategieparameter, Kostenprofil oder Modellversion ändern. | Änderung ist in den Laufmetadaten erkennbar; Ergebnisse können nicht versehentlich als derselbe fachliche Lauf gelten. |
| REP-004 | Ein Lauf scheitert an Datenfehler oder nicht unterstützter Konfiguration. | Verständlicher Fehlerstatus und CLI-Fehlercode; kein erfolgreicher Nullrendite-Bericht. Diagnose benennt den Verarbeitungsschritt. |
| REP-005 | Beispiel-CLI mit mitgelieferten Daten; danach eine andere registrierte Strategie auswählen. | Beide Läufe liefern denselben Ergebnisvertrag und vollständige Metadaten. Der Beispielbefehl funktioniert nach dokumentierter Einrichtung offline. |

## 11. Externe Datenanbieter — M4 nach D-01

| ID | Gegeben / Aktion | Erwartetes Ergebnis |
| --- | --- | --- |
| SRC-001 | F-PROVIDER liefert Seite A mit Kerzen 1–3 und Seite B mit 3–5, Kerze 3 identisch. | Normalisierter Bestand enthält Kerzen 1–5 genau einmal; Pagination endet korrekt. |
| SRC-002 | Download wird nach Seite A unterbrochen und fortgesetzt; anschließend denselben Zeitraum erneut importieren. | Vollständiger Bestand ohne neue Dubletten; bestätigte Rohdaten bleiben nachvollziehbar; kein stiller Datenverlust. |
| SRC-003 | Anbieterersatz liefert vorübergehende Fehler bzw. dauerhaft einen Fehler. | Begrenzte Wiederholungen mit testbarem Warteplan; Erfolg oder eindeutiger Fehlerstatus nach dem Limit. Test wartet nicht auf reale Netzwerk-Zeitlimits. |
| SRC-004 | Anbieter liefert für einen bereits gespeicherten Schlüssel einen anderen Kurs oder eine laufende Kerze. | Konflikt bzw. Unvollständigkeit wird dokumentiert und nach festgelegter Regel behandelt; vorhandene abgeschlossene Daten werden nicht unbemerkt überschrieben. |

## 12. Zahlenmodelle und zeitliche Auswertung — M5

| ID | Gegeben / Aktion | Erwartetes Ergebnis |
| --- | --- | --- |
| ML-001 | 14 Minutenkerzen mit drei Kerzen Lookback und zwei Kerzen Zielhorizont. Validierung beginnt bei Minute 7, Test bei Minute 10. | Entscheidungen bei Minuten 3 und 4 landen im Training, Minute 7 in der Validierung und Minuten 10 bis 12 im Test. Zielwerte an oder nach der nächsten Grenze sowie unvollständige Endfenster entfallen. |
| ML-002 | Trainingsmerkmale [1, 3], Validierungsmerkmal [100]; Zentrierung um Trainingsmittel. | Mittelwert bleibt 2; Validierungswert wird zu 98. Kein Fit auf Validierung/Test. |
| ML-003 | Der Entscheidungsschluss liegt bei 102, das folgende Open bei 103 und das Close der zweiten Zielkerze bei 104. | Das Ziel ist 1/103. Eine Änderung am späteren Kurs verändert das Ziel und lässt die Merkmale der Entscheidung unverändert. |
| ML-004 | Gespeichertes Modell mit festgelegten Merkmalen, Trainingsgrenzen und Versionen laden; danach inkompatible Konfiguration verwenden. | Kompatibles Artefakt ist reproduzierbar verwendbar; inkompatible Merkmale oder Training im Auswertungszeitraum führen zu verständlichem Fehler. |
| ML-005 | F-PREDICTOR während eines Backtests mehrfach aufrufen. | Keine Trainings-/Optimiereraufrufe oder Gewichtsänderungen. Vorhersagegüte und Handelsergebnis bleiben getrennte Auswertungen. |
| ML-006 | Parameter-/Modellauswahl auf Training/Validierung; abschließender Testabschnitt ist vorhanden. | Auswahl verändert sich nicht durch Testdaten; abschließende Testauswertung erfolgt erst mit festgelegtem Modell und Parametern. |

## 13. Visuelle Darstellung und Modellvergleich — M6

| ID | Gegeben / Aktion | Erwartetes Ergebnis |
| --- | --- | --- |
| VIS-001 | Gleiche historischen Kerzen, gleiche Renderer-Konfiguration und festgelegte Software-/Schriftumgebung zweimal rendern. | Identische Bildpixel; keine Uhrzeit oder zukünftige Daten im Bild. Bytes dürfen nur bei ausdrücklich zugelassenen Metadatenabweichungen differieren. |
| VIS-002 | Training und Validierung haben gleiche Zeilenindizes und Form, aber unterschiedliche Kursinhalte. Zusätzlich Renderer-Parameter oder Fensterlänge ändern. | Kein falscher Cache-Treffer. Dateninhalt/-identität und darstellungsrelevante Konfiguration sind Teil der Cache-Identität. |
| VIS-003 | Zahlen- und Bildansatz für dieselben Beispielzeitpunkte erzeugen. | Identische Informationsgrenze, Zielzuordnung und Datensplit; keine Bilder mit zukünftigen Kerzen oder Zielmarkierungen. |
| VIS-004 | Bildmodell mit F-PREDICTOR und anschließend gespeichertem kompatiblem Artefakt integrieren. | Gemeinsamer Signal- und Backtest-Vertrag bleibt gültig; Modellartefakt und Renderer-Version stehen im Ergebnis. |

## 14. Zuordnung, Priorität und Abnahme

| Meilenstein | Zugeordnete Fälle | Abnahmehinweis |
| --- | --- | --- |
| M2 | SET-001–003; DAT-001–010; CAU-001–005 | Pflichtgrundlage vor Vertrauen in Daten-/Zeitverträge. SET-004 wird in M2 vorbereitet und in M3 vollständig geprüft. |
| M3 | SET-004; SIG-001–002; ACC-001–009; EXE-001–010; RSK-001–006; REP-001–004; erster Durchlauf REP-005 | Geld-, Zeit- und Ausführungsfehler haben Vorrang. Profilabhängige Unterstützung oder Ablehnung wird ausdrücklich getestet. |
| M4 | SIG-003–005; REP-005; SRC-001–004 | Reale Anbieterfälle erst nach D-01; weiterhin mit lokalen Anbieterersatztests absichern. |
| M5 | ML-001–006; SIG-006 mit Regel-/Zahlenmodell | Kein Lernlauf vor festgelegtem Ziel und zeitlicher Trennung. |
| M6 | VIS-001–004; SIG-006 vollständig | Darstellung und Caches ergänzen, gemeinsame Ausführung erneut prüfen. |

Empfohlene Reihenfolge der ersten ausführbaren Fälle: DAT-001/003/007,
CAU-001/003, ACC-001/002/003/004/005 und EXE-001/005/006. Sie sichern die
wichtigsten Daten-, Zeit- und Rechenannahmen ab. Die restlichen Pflichtfälle des
Meilensteins folgen vor dessen Abschluss; diese Auswahl ersetzt die Abnahme nicht.

Ein Meilenstein erfordert bestandene Verhaltensprüfungen und eine dokumentierte
Zuordnung von IDs zu ausführbaren Tests. Profitabilität, möglichst viele Trades
oder ein bestimmter Modell-Loss sind keine Abnahmekriterien der Infrastruktur.
Echte Datendownloads, Training und Backtests gelten nicht durch diese
Testfallspezifikation als ausgeführt oder erfolgreich.

## 15. Datenfunktionen, Normalisierung und Kalender — Ergebnis am 2026-11-02

Schnittstellen und Einheiten sind in [DATA_CONTRACT.md](DATA_CONTRACT.md)
festgelegt; Einrichtung, Funktionszuordnung und Ausführungsstand stehen in
[TESTING.md](TESTING.md). Nach der ursprünglichen Rotphase wurden CSV-Import,
Validierung, Resampling, historische Fenster und das numerische Referenzmerkmal
implementiert. Das zweite Paket ergänzt Normalisierung und Qualitätsberichte;
das dritte ergänzt explizite UTC-Sitzungskalender. Das vierte prüft angefragte
Zeiträume mit Randlücken und vollständig fehlenden Serien.
Das fünfte ergänzt lokale Artefakte und Herkunftsmanifeste.
Die folgenden Zahlen zählen parametrisierte Prüfungen; eine grüne
Teilprüfung nimmt keine noch fehlende Integration einer fachlichen ID vorweg.

| Fall-ID | Ausführbare Varianten | Tatsächlicher Stand / verbleibender Umfang |
| --- | --- | --- |
| DAT-001 | 1 | Bestanden; kanonischer Import. |
| DAT-002 | 7 | Bestanden; UTC-Normalisierung und explizite Zeitzonen. |
| DAT-003 | 54 | Bestanden; numerische, Metadaten- und CSV-Strukturprüfungen sowie Eingangsprüfung vor Normalisierung. |
| DAT-004 | 8 | Bestanden; Dubletten-Audit und ausdrückliche Konfliktabweisung. |
| DAT-005 | 44 | Bestanden für Sortierung, aktive Lücken/Randlücken, fehlende angefragte Serien und strikte Normalisierung; Backtest-Integration offen. |
| DAT-006 | 44 | Bestanden für explizite UTC-Sitzungen, geschlossene Zeiten und aktive Auftragsgrenzen/-anzahlen; reale Kalenderadapter offen. |
| DAT-007 | 6 | Bestanden; Aggregate, Grenzen, Intervallparameter und Präzision. |
| DAT-008 | 3 | Bestanden; unvollständige Zielblöcke scheitern ausdrücklich. |
| DAT-009 | 78 | Bestanden; Inhalts-/Auftragskennungen, lokale Artefakte, Herkunfts-/Quellprüfung, vollständige Wiederherstellung und Ablehnung inkonsistenter Dateien. Bestehende Datenkennungen bleiben kompatibel. |
| DAT-010 | 10 | Bestanden; getrennte Gruppen, Auswahl, fehlende Instrumente/Intervalle und Abweisung nicht angefragter Eingaben. |
| CAU-001 | 12 | Bestanden für Fenster und Eingabegrenzen; Strategieaufrufe im Runner offen. |
| CAU-002 | 1 | Bestanden für Verfügbarkeitsmetadaten; zeitgesteuerte Freigabe im Runner offen. |
| CAU-003 | 6 | Bestanden für Zahlenmerkmale; Bild- und Signalanteile offen. |
| CAU-004 | 5 | Bestanden für fehlende Dateien, bestehende Ziele und Schreibabbrüche; weitere Anbieter-/Merkmalsfehlerwege offen. |
| CAU-005 | 1 | Bestanden für Fenster-Eingabesnapshot; spätere Strategieintegration offen. |

Insgesamt wurden 280 Prüfungen ausgeführt und bestanden, ohne Fehler oder
übersprungene Fälle. Die ursprünglichen 48 und die vor Umsetzung ergänzten
42 Randprüfungen wurden zuerst rot ausgeführt. Ein zusätzlicher Regressionstest
für fehlerhafte Header-Zitate wurde vor seiner Korrektur ebenfalls rot ausgeführt.
Die 39 Normalisierungsprüfungen wurden vor der zugehörigen Fachlogik rot
ausgeführt und anschließend gemeinsam mit den bisherigen 91 Fällen grün.
Die 46 Kalenderprüfungen wurden ebenfalls zuerst rot ausgeführt und bestehen
nach Implementierung zusammen mit den bisherigen 130 Fällen.
Die 47 Zeitraum-/Serienprüfungen wurden zuerst rot ausgeführt und bestehen
nach Implementierung zusammen mit den bisherigen 176 Fällen.
55 Speicherprüfungen wurden vor ihrer Fachlogik rot ausgeführt; zwei zusätzliche
Unicode-Regressionsprüfungen wurden vor der Korrektur rot ausgeführt. Die
ausdrückliche Verarbeitungsversion wurde im bestehenden Roundtrip-Fall zuerst
rot geprüft. Zusammen bestehen die 57 neuen Varianten mit den bisherigen 223.
Die übrigen fachlichen Fall-IDs bleiben spezifiziert. M2 und die gesamte
Testfallspezifikation sind damit noch nicht vollständig abgenommen.

## 16. Konkreter synthetischer Kalenderfall — DAT-005/006

Das dritte Arbeitspaket verwendet endliche, ausdrücklich konfigurierte
UTC-Sitzungen mit Gültigkeitsbereich `[2026-01-01, 2026-01-06)`; es setzt keinen
realen Markt voraus. Ein handgerechneter Kernfall steht in
`tests/test_session_calendars.py::test_dat_005_only_active_parts_of_a_gap_are_reported`:

- Aktiv am 2026-01-02: `[09:00, 09:03)` und `[09:05, 09:08)` UTC.
- Vorhanden: Minutenkerzen mit Beginn 09:00 und 09:07, zusätzlich eine identische
  Dublette und unsortierte Eingaben.
- Erwartet: genau zwei eindeutige Kerzen; Lücken `[09:01, 09:03)` und
  `[09:05, 09:07)` mit jeweils zwei fehlenden Kerzen. Die Pause zählt nicht mit.

Weitere Prüfungen unterscheiden fehlende aktive Sitzungen von geschlossenen
Nächten/Tagen. Sitzungen dürfen eigene Raster haben; Kerzen außerhalb aktiver
Sitzungen, auf falschem Raster oder über deren Ende hinweg werden zurückgewiesen.
Nicht unterstützte Teilkerzen werden als Konfigurationsfehler abgewiesen.
Der vollständige Kalender bleibt in den Metadaten und in der Datenkennung.
Randlücken und vollständig leere angefragte Serien werden durch das anschließende
Paket mit ausdrücklichem Datenzeitraum geprüft; siehe Abschnitt 17.

## 17. Angefragte Datenzeiträume — DAT-005/006/009/010

Das vierte Arbeitspaket verwendet `DatasetRequest` mit Instrument, Intervall
und UTC-Zeitraum `[start_time, end_time)`. Die ausführbaren Sollwerte stehen in
`tests/test_requested_ranges.py` und wurden vor der Fachlogik rot ausgeführt.

- Durchgehender Zeitraum `[00:00, 00:05)` mit Minutenkerzen 00:02 und 00:03:
  fünf erwartet; Randlücken `[00:00, 00:02)` mit zwei und `[00:04, 00:05)`
  mit einer fehlenden Kerze. Eine zusätzliche Dublette zählt nicht doppelt.
- Derselbe Auftrag ohne Eingabedaten: leere auswählbare Serie, fünf erwartet,
  eine Lücke über den gesamten Zeitraum; strikte Verarbeitung scheitert mit
  `row_number=None`. Eine andere vorhandene Serie füllt diesen Auftrag nicht.
- Aktive Sitzungen `[00:00, 00:02)` und `[00:04, 00:06)`, Auftrag über
  `[00:00, 00:06)`: vier erwartet. Ohne Daten gibt es zwei aktive Lücken mit
  jeweils zwei fehlenden Kerzen. Die Pause wird nicht mitgezählt.
- Auftrag nur über die Pause: null erwartet, keine Lücke; leere Daten sind
  auch im strikten Modus vollständig.
- Daten außerhalb des Auftrags, zusätzliche Serien, doppelte Aufträge,
  unbekannte Kalenderabdeckung und angeschnittene aktive Kerzen werden abgewiesen.
- Auftrag und erwartete Anzahl bleiben im Ergebnis; geänderte Soll-Zeiträume
  verändern die Datenkennung. Reihenfolge und identische Dubletten tun dies nicht.

Ohne Auftrag bleiben die bisherigen Historienregeln und Datenkennungen erhalten.
Ein tatsächlicher Backtest-Runner ist weiterhin ein späterer Meilenstein.

## 18. Lokale Datensätze und Herkunft — DAT-009/CAU-004

Die Sollwerte des fünften Pakets stehen in `tests/test_dataset_storage.py`;
der Schnittstellen- und Formatvertrag in [STORAGE_CONTRACT.md](STORAGE_CONTRACT.md).

- Datei mit den exakten Bytes `abc`: Quellgröße 3 und SHA-256
  `ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad`.
  Die Quelle bleibt unverändert.
- Normalisieren eines gemischten Batches, Speichern und Laden: identische Kerzen,
  Dezimalpräzision, Kalender, Aufträge, Lücken und Originalzeilen. Globale
  Zeilennummern dürfen die Eingabezahl einer ausgewählten Gruppe übersteigen.
- Gleicher Datensatz und gleiche Herkunft: identische Dateien und Manifest-
  Kennungen. Ein anderer Audit oder eine andere Herkunft ändert die Artefaktkennung,
  auch wenn die normalisierte Datenkennung gleich bleibt.
- Leere ausdrücklich angefragte Serien bleiben speicherbar, einschließlich
  vollständig geschlossener Zeiträume. Abgeleitete Daten referenzieren ihre
  Vorgängerkennung; Quellen lassen sich ausdrücklich erneut prüfen.
- Veränderte Daten oder Herkunft werden über die Artefaktprüfsumme erkannt.
  Selbst mit neu berechneter Prüfsumme bleiben widersprüchliche Datenkennungen,
  Qualitätsberichte und ungültige Schemata unzulässig.
- Ein bestehendes Ziel bleibt bytegleich. Simulierte Fehler bei Datei-`fsync`
  und Veröffentlichung hinterlassen weder ein Zielartefakt noch temporäre Dateien.
  Fehlende Dateien bleiben technische Fehler; ein umgezogenes Artefakt ist ohne
  seine ursprünglichen Quellen lesbar.

In der Speicher-API gibt der Aufrufer die Herkunft an. Die CLI ermittelt sie beim
Import automatisch.

## 19. Offline-Datenablauf über CLI — SET-004/DAT-009/CAU-004

| ID | Gegeben / Aktion | Erwartetes Ergebnis |
| --- | --- | --- |
| SET-004 | `market-trader data import --config configs/data-example.toml --output <Ziel>` aus dem Repo und mit absolutem Konfigurationspfad aus einem anderen Arbeitsverzeichnis starten. | Die CSV wird relativ zur Konfigurationsdatei gefunden. Beide Läufe erzeugen bytegleiche Artefakte und dieselbe JSON-Zusammenfassung mit Datensatz- und Artefaktkennung. |
| DAT-009 | Synthetische CSV mit einer Kerze und Volumen 0,125 importieren. Anschließend `market-trader data verify --artifact <Ziel>` ausführen. | Die Kerze bleibt exakt erhalten. Die Herkunft enthält den logischen Namen, die Größe und SHA-256 der importierten CSV-Bytes. Die Prüfung liefert dieselbe Zusammenfassung. |
| DAT-005 | Das angefragte Ende um ein Kerzenintervall verschieben. | Bei `report` weist die Zusammenfassung eine fehlende Kerze aus. Bei `reject` scheitert die Normalisierung. |
| CAU-004 | Ungültige Konfiguration, fehlende oder ungültige CSV, vorhandenes Ziel oder verändertes Artefakt. | Der Befehl endet mit einem Fehlercode und einer Meldung zum betroffenen Schritt. Ein fehlerhafter Import veröffentlicht kein Artefakt und ersetzt kein vorhandenes Ziel. |

Alle 17 Varianten in `tests/test_data_cli.py` schlugen vor der Implementierung
fehl und bestanden nach der Umsetzung.

## 20. Erster R0-Backtest

`tests/test_backtesting.py` prüft ACC-001–006, EXE-001–007, DAT-005 und die
Verfügbarkeit von Strategie-Fenstern aus CAU-001/002. Die Tests für SIG-002 und
RSK-003 prüfen fehlende Short-Freigabe und unzureichendes Kapital. Alle 17
Varianten wurden vor und nach der Umsetzung ausgeführt.

## 21. R0-Kosten, Risiko und Berichte

Die ergänzten Fälle in `tests/test_backtesting.py` und `tests/test_data_cli.py`
prüfen ACC-007–009, EXE-008–010, RSK-001–006 und REP-001–005 für den ersten
synthetischen Ablauf. Das CLI-Beispiel produziert bei Startkapital 1.000 Q
einen Gewinn nach Gebühren von 13,185 Q. Zwei Arbeitsverzeichnisse liefern
denselben Bericht.

## 22. Strategiewechsel und Zustandsisolation

Die Prüfungen zu SIG-003–005 und REP-005 wählen `single_long` und
`sma_crossover` über TOML mit denselben Daten und Kosten. Die zweite Strategie
benötigt `slow_period + 1` Kerzen und erzeugt Ein- und Ausstiege an den
vorab berechneten Kreuzungen. Der CLI-Test vergleicht die Ergebnisstruktur
und die Laufmetadaten beider Konfigurationen. Wiederholte Aufrufe der
`single_long`-Factory beginnen mit frischem Zustand. Unbekannte Namen sowie
falsche, fehlende oder zusätzliche Parameter scheitern vor dem Datenzugriff.

## 23. Portierter zweibeiniger Pullback

Das synthetische Beispiel setzt das erste Tief bei 100 Q, das Erholungshoch
bei 102 Q und das zweite Tief bei 99 Q mit einem Schlusskurs von 100 Q.
Beim R0-Standardraster 0,1 Q erzeugt der Schluss der vierten Kerze einen
Stop bei 98,8 Q und ein Ziel bei 102,4 Q. Der nächste Open bei 100,2 Q
führt zum Kauf von acht Einheiten. Ein Zielausstieg liefert 17,6 Q Gewinn,
eine Stop-Lücke mit Eröffnung bei 98 Q liefert 17,6 Q Verlust.
Das CLI-Beispiel prüft zusätzlich das Raster 0,25 Q. Die Grenzfälle prüfen
eine zu schwache Erholung, ein neues Tief in der Erholungskerze, gleiche
erste und zweite Tiefs, einen fehlenden Schlusskursrückgang und verspätete
Daten. Die Erholung muss unter dem Starthoch bleiben.

## 24. Backtest eines Datensatzartefakts

Die CLI lädt ein Artefakt mit zwei Sitzungen und einer geschlossenen Periode
dazwischen. Die gespeicherte Herkunft und Artefaktkennung stehen im Bericht.
Der Aufruf aus zwei Arbeitsverzeichnissen liefert dieselben Ergebnisse.
Fehlende oder veränderte Artefakte, aktive Lücken und mehrdeutige Eingaben
scheitern mit einer Diagnose und ohne Ergebnisbericht. Die bestehenden
CSV-Konfigurationen werden ebenfalls geprüft.

## 25. Überwachter Datensatz für das Zahlenmodell

`tests/test_training_data.py` prüft die Ausschlussregeln an zeitlichen
Grenzen und bei verspäteten Daten aus ML-001 sowie das handgerechnete Ziel
und seine Trennung von historischen Merkmalen aus ML-003. Weitere Varianten
prüfen Sitzungswechsel, Eingabefehler und den reproduzierbaren CLI-Export.

## 26. Erstes CPU-Modell

`tests/test_baseline.py` prüft die Skalierung auf Trainingsdaten (ML-002),
handgerechnete Ridge-Koeffizienten bei konstanten und korrelierten Merkmalen,
Validierungsfehler und Ausschluss des Tests aus Fit und Validierung (ML-006).
Der CLI-Fall prüft reproduzierbare Modellparameter im Bericht, Prüfsummen
und die Veröffentlichung.

## 27. Auswertung eines gespeicherten Modells

Die acht zusätzlichen Fälle in `tests/test_baseline.py` prüfen die
Kompatibilität von Modell und Datensatz (ML-004), die Grenzen des
Testabschnitts (ML-001) und die Berechnung der Testfehler mit gespeicherten
Parametern (ML-006). Der CLI-Fall prüft den Bericht, die Prüfsumme und
unveränderte Eingabeartefakte.

## 28. Eingefrorenes Modell im Backtest

`tests/test_model_strategy.py` prüft neun Varianten. Der synthetische
CLI-Fall verwendet gespeicherte Modellparameter und erzeugt einen Long-Einstieg
zu 103,20 und einen Ausstieg zu 105,80 für zwei Einheiten. Nach 0,418 Gebühren
beträgt das Netto-P&L 4,782. Die übrigen Fälle prüfen Verfügbarkeit,
Sitzungsgrenzen, die Signalschwelle, Kompatibilität und wiederholte
Inferenz ohne Training (ML-004/005).

## 29. Feste SMA-Menge

Die vier zusätzlichen Varianten zu SIG-004 prüfen eine feste SMA-Menge,
deren Ausführungen und die Ablehnung von null, booleschen Mengen und
gemischten Mengenkonfigurationen.

## 30. Wiederverwendbarer HTML-Bericht

| ID | Gegeben / Aktion | Erwartetes Ergebnis |
| --- | --- | --- |
| REP-006 | Gespeicherten R0-Lauf zweimal von unterschiedlichen Arbeitsverzeichnissen aus darstellen. | Identische, eigenständige HTML-Dateien mit Equity, Drawdown und auswählbaren Trades. Bestehende Ausgaben und Eingaben bleiben erhalten. |
| REP-007 | Zweiten Lauf auf denselben Daten und mit demselben Ausführungsprofil hinzufügen. | Beide Strategien erscheinen auf derselben Zeitachse. Abweichende Daten, Kosten oder Zeitpunkte werden abgelehnt. |
| REP-008 | Eingefrorenes Ridge-Modell, passende Auswertung, Trainingsdaten und Backtest-Datensatz verwenden. | Prognosepunkte und Referenzfehler stimmen mit der Testauswertung überein. Die Einzeltrade-Ansicht rekonstruiert die Prognose und ihre Beiträge aus den bis zur Entscheidung verfügbaren Kerzen. |
| REP-009 | Unbekanntes Berichtsformat, fremdes Modell oder beschädigte Eingabe übergeben. | Der Befehl meldet den Fehler ohne Teilausgabe. Eingabetext kann keinen HTML-Code in die Ausgabe einschleusen. |
