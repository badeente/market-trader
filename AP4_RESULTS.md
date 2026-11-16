# AP4 – technische Abnahme und interne Veröffentlichung

Stand: 16. November 2026. **Implementiert, technisch abgenommen und intern veröffentlicht.** Andreas hat Commit, Push und Veröffentlichung ausdrücklich freigegeben. Der bestehende Dienst unter `http://10.10.1.136:8086/` liefert jetzt die AP4-Workbench aus dem akzeptierten r2-Wheel. Keine automatische Freigabe für AP5.

## Gelieferter Umfang

- Nur lesende Run-Übersicht mit Suche, Modell-/Status-/Sammlungsfilter, Datenidentität und tatsächlicher Validierungsperiode.
- Auswahl von 2–4 abgeschlossenen, gültigen Läufen, einschließlich erkennbar ausgeblendeter Auswahl. Fehlerhafte/ungültige Läufe bleiben sichtbar, sind aber nicht vergleichbar.
- Detailansichten für gespeicherte Konfiguration, Modellwahl, Herkunft/Code/Runtime, Prognosebaselines, Kosten/Trades/Netto/Drawdown und MLP-Epochen.
- Vergleich anhand gespeicherter Daten-/Kohorten-/Ziel-/Kosten-/Runtime-Signaturen. Unpassende Bedingungen werden vor den Metriken ausdrücklich markiert. Name/Hypothese sind Run-Metadaten, keine Modellparameter.
- Trainings-/Validierungs-MSE getrennt von Konto-Equity und Drawdown. Prognosefehler in bp mit bis zu vier Nachkommastellen; Handelswerte in Punkten je fester Einheit, nicht EUR. Keine Rangliste oder Profitabilitätszusage.
- Selbstständige deutsche HTML/CSS/JS/SVG-Oberfläche ohne CDN. Lange Konfigurationstabellen einklappbar; maßgebliche Bedingungswarnungen bleiben sichtbar.
- Explizite private Registry-Zuordnungen statt Browser-Dateipfaden. GET/HEAD, Host-/Client-Prüfung, bestehender HTML-Verzeichnisschutz, bounded JSON/Report-Lesen, kein CORS und nur gleich-originige API-Aufrufe.
- Kein Training, keine Inferenz, kein Backtest, keine Jobs/Abbruch-/Upload-/Shell-Endpunkte aus dem Browser. Rohdaten, Modellgewichte, private Logs und Quellpfade werden nicht ausgeliefert.

Scope/API: [AP4_WORKBENCH.md](AP4_WORKBENCH.md). Betrieb: [REPORT_VIEWER.md](REPORT_VIEWER.md). Tests: [TESTING.md](TESTING.md).

## Tatsächlich ausgeführte Prüfungen

- Vollsuite mit dem isolierten Trainingsstack: **1.028 bestanden in 203,23 s**.
- Vollsuite ohne Torch/NumPy: **1.022 bestanden, 6 reine Trainingstests übersprungen in 240,78 s**.
- Abschließender fokussierter Daten-/Server-/Frontendstand: **137 bestanden**; unabhängiger Daten-/Server-Nachtest **123 bestanden**; finales UI-Review **14 bestanden** plus eigene Koordinaten-/DOM-Proben.
- Statischer Scan der hinzugefügten Produktionsquellen: keine Treffer für eingebettete Geheimnisse, Shell-Ausführung, eval/exec, unsichere Deserialisierung oder SQL-Interpolation. `git diff --check` nach Dokumentkorrektur erneut prüfen; Graphify wurde code-only aktualisiert und abgefragt. Inline-JavaScript/HTML liegt außerhalb seiner AST-Abdeckung.
- Separat installiertes r2-Wheel außerhalb des Checkouts, ohne Torch/NumPy; `pip check` grün. Alle **33 Paketdateien** im Wheel bytegleich zu den aktuellen Quellen; keine externen Python-Laufzeitabhängigkeiten.

### Originaldaten und HTTP

Explizite Sammlungen `ap2-acceptance-v2` und `ap3-acceptance-v2`: **7 Läufe, davon 6 completed und 1 failed**. Es wurde kein neuer Marktmodelllauf gestartet.

Alle Primär- und Baseline-Prognosefehler sowie sämtliche Handels-Summarys wurden gegen die gespeicherten Originale abgeglichen. Die Primärkurven wurden aus vollständigen Kontoverläufen separat nachgerechnet: Endpunkte, globale Extrema und alle angezeigten Stichproben stimmen. MLP-Historie/Best-Epoche/Abbruchgrund und sämtliche angebotenen Reportbytes stimmen exakt. Kompatibler MLP-/Ridge-/Replay-Vergleich und inkompatible Kosten-/Runtime-Vergleiche wurden wirklich angefragt.

Inventar und SHA-256 sämtlicher **125 Registry-Dateien** sind gegenüber dem Vorher-Snapshot unverändert. Alle sechs bestehenden veröffentlichten HTML-Berichte bleiben bytegleich. Falsche Hosts, Mutation, ungültige/mehrdeutige Queries, fehlgeschlagene Vergleichsläufe und fremde Datei-/Gewichtszugriffe werden abgewiesen. Die Prüfung lief auf einer temporären Loopback-Testinstanz, nicht als Live-Deployment oder LAN/VPN-Abnahme der neuen Version.

### Echter Browser

Chromium 145.0.7632.6 mit separat entpackten Bibliotheken, ohne sudo oder Hermes-/Systembibliotheksänderung. Reale Desktop-, 390px- und 320px-Abläufe: Filter/Leersuche, Auswahlgrenzen, versteckte Auswahl, fehlgeschlagener Import, kompatibler und inkompatibler Vergleich, Deep-Link-Reload, Fehler/Zurück, native Tastaturbedienung, lesbare getrennte Diagramme und kein horizontaler Gesamtseiten-Overflow. PNGs wurden betrachtet. Keine unbehandelten App-Fehler oder externen App-Netzwerkaufrufe im aufgezeichneten Lauf.

Eine ausdrücklich adversariale Namens-Fixture wurde nur im QA-Browser eingesetzt: HTML-förmiger Text blieb Text und wurde nicht ausgeführt. Sie verändert keine Forschungsdateien und ist kein erfundenes Marktergebnis.

## Unabhängige Reviews und Korrekturen

HTTP-/Frontend-Erstreview grün. Datenreview fand drei Korrektheitsfehler und eine ergänzende Speicherhärtung:

1. Vergleich mischte separat geladene Detail-/Originalversionen. Jetzt stammt alles aus denselben erfassten Einträgen; sämtliche Fingerprints werden nach Sammlung und Berechnung erneut geprüft. Statuswechsel und kohärente Neuschreibungen führen zum sicheren Abbruch statt gemischter Ergebnisse.
2. Nicht endliche Konto-/Kapitalwerte. Jetzt endliches positives Startkapital und endliche Equity, auch beim initialen Snapshot ohne Zeitstempel.
3. Fehlende Quelle erschien als leer. Jetzt alias-only Nichtverfügbarkeitsmeldung; tatsächlich leere lesbare Quelle bleibt fehlerfrei.
4. Zu große HTML-Datei wurde vor der Grenze komplett verifiziert. Jetzt Größen-/Dateityp-Preflight vor kaltem Registry-Hashing; bestehende Hash-/No-follow-Prüfung bleibt erhalten.

Alle vier Punkte mit beobachteten RED→GREEN-Regressionen korrigiert. Unabhängiger Nachreview ohne offene Sicherheits-/Logikbefunde, einschließlich 16 zusätzlicher synthetischer Mutationsfälle über alle vier Vergleichspositionen/kalt/warm. Separates finales UI-Review bestätigt die nachträglichen mobilen Viewbox-, bp-Präzisions- und Tabellenänderungen. Keine Reviewer-Schreibzugriffe auf Originaldaten.

## Release und private Nachweise

- Wheel: `var/market-trader/releases/ap4-workbench-20261115-r2/market_trader-0.0.0-py3-none-any.whl`, **109.708 Bytes**.
- SHA-256: `a8160aa42957d5e2c118887b09b8217262716dfd341207e736c4896a21568f42`.
- Separater installierter Test: `/home/hermes/workspace/.venvs/market-trader-ap4-release`.
- Akzeptierte reale HTTP-Prüfung: `var/market-trader/acceptance/ap4-wheel-http-v3.json` und `exercise_ap4.py`.
- Akzeptierter Browserlauf: `var/market-trader/qa/ap4-wheel-v2/browser-verification.json`, PNGs im selben Verzeichnis, Treiber `acceptance/ap4_browser.mjs` und `qa/cdp.mjs`.
- Originalzustand: `acceptance/ap4-before.json`. Finale Reviews: `ap4-review-http-ui.json`, `ap4-review-ui-final.json`, `ap4-review-data-final.json`.
- r1 und frühe Zwischenbelege bleiben historisch; sie sind nicht der freigegebene Releasekandidat.

Private Artefakte, Daten, QA-Bilder und Belege bleiben außerhalb Git und HTTP-Verzeichnis. Der Paketversionsstring `0.0.0` allein identifiziert den Stand nicht; Wheelhash und Quellenabgleich verwenden.

## Erneute Abnahme am 16. November 2026

- Vollsuite erneut ausgeführt: **1.028 bestanden in 187,10 s**; `git diff --check` erfolgreich.
- Akzeptiertes r2-Wheel erneut mit dem Checkout abgeglichen: alle **33 Paketdateien bytegleich**, unveränderter Wheelhash.
- Frische isolierte Loopback-Instanz auf Port 18087: Originalmetriken, Kurven, kompatible/inkompatible Vergleiche, sämtliche Reportbytes und Sicherheitsgrenzen erneut erfolgreich geprüft. Alle **125 Registry-Dateien** bleiben unverändert; **7 echte Läufe**, davon **6 completed / 1 failed**.
- Echte Chromium-Desktop-/390px-/320px-Abläufe erneut erfolgreich; keine App-Fehler oder externen App-Aufrufe. Neue Belege: `acceptance/ap4-recheck-20261116-http.json` und `qa/ap4-recheck-20261116/browser-verification.json`.
- Graphify mit der tatsächlich vorhandenen isolierten Hermes-Installation aktualisiert und abgefragt: 1.522 Knoten / 5.045 Kanten. `GRAPHIFY.md` berichtigt den veralteten Installationspfad.
- Kein Commit, Push, Live-Neustart oder Deployment im Zuge dieser Nachprüfung.

## Live-Veröffentlichung am 16. November 2026

- Akzeptiertes r2-Wheel im getrennten Serving-Environment installiert; `pip check` grün und alle 33 installierten Paketdateien bytegleich zum akzeptierten Wheel. Kein Torch/NumPy/Node im Serving erforderlich.
- Bestehende Unit, Releasepointer und altes Wheel gesichert unter `releases/ap4-deployment-backup-20261116T053126Z`; bisheriges installiertes Paket vorab gegen das Rollback-Wheel bytegeprüft. `rollback.sh` liegt bereit; ein tatsächlicher Rückwechsel war nicht nötig und wurde nicht als Live-Test ausgeführt.
- Bind `10.10.1.136:8086`, bestehende LAN- und VPN-Allowlist sowie alle übrigen Unit-Einstellungen unverändert; ausschließlich Beschreibung und explizite AP2/AP3-Registry-Argumente ergänzt. Dienst neu gestartet, active/enabled; Linger vorhanden.
- Live-HTTP-Abnahme erfolgreich: sieben echte Läufe, alle Originalmetriken/Kurven/Reportbytes, kompatible/inkompatible Vergleiche und Sicherheitsgrenzen geprüft. Alle 125 Registry-Dateien unverändert. Nachweis `acceptance/ap4-live-20261116-http.json`.
- Zweiter LAN-Rechner `10.10.1.20`: Startseite, Run-API, tatsächlicher MLP-/Ridge-Vergleich und bestehender AP3-Bericht HTTP 200 und bytegleich. Nachweis `acceptance/ap4-live-20261116-lan.json`. VPN-Allowlist erhalten; kein separater neuer VPN-Clienttest behauptet.
- Echter Browser auf der Live-Adresse: Desktop-/390px-/320px-Abläufe erneut bestanden, Screenshots geprüft, keine App-Fehler oder externen App-Aufrufe. Nachweise `qa/ap4-live-20261116/`.
- Startseite `/` und `/experiments` sind jetzt die Workbench; die ursprüngliche Galerie liegt unter `/reports`. Bisherige `/reports/<name>.html`-URLs bleiben bytegleich erreichbar.

## Grenzen

- 2024 bleibt Entwicklungs-/Selektionsdatenbasis, kein unabhängiger Abschlusstest. 2025 wurde nicht neu modelliert/bewertet. AP4 zeigt gespeicherte Ergebnisse; keine wirtschaftliche Verbesserung durch die UI behaupten.
- Nur interner MVP, ohne Login/TLS und nicht öffentlich betreiben. Bestehende LAN-/VPN-Freigaben nicht erweitern. Cache-Fingerprints/Hashes sichern Konsistenz, nicht Authentizität gegen den vertrauenswürdigen Registry-Eigentümer.
- Kurven mit bis zu 800 angezeigten Punkten sind als ausgedünnt markiert; Summen stammen aus vollständigen Daten. Keine Jobsteuerung, große Datenbankindizierung oder beliebige Artefaktdownloads.
- Nicht blockierende UI-Hinweise aus dem Review: schnell überlappende Aktualisierungen können die Auswahl leeren; nach Entfernen ausgeblendeter Auswahlchips kann der Tastaturfokus zurückspringen. Das erzeugt keine veralteten Finanzmetriken; Auswahl bei Bedarf erneut setzen.
- Die gesonderte Git-/Deployment-Freigabe wurde am 16. November 2026 erteilt; die obigen Live-Prüfungen ersetzen die frühere lokale-only Einschränkung. AP5, neue Trainingsläufe und öffentliche Freigabe sind weiterhin nicht umfasst.
