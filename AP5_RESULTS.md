# AP5 — Abnahme und interne Veröffentlichung

Stand: 16. November 2026. **Lokaler AP5-Funktionsumfang technisch abgeschlossen und real abgenommen; unabhängige Schlussreviews ohne offene Befunde.** Der freigegebene Umfang ist lokale CLI-Iteration mit begrenzten Trainingsjobs und deklarativen Kleinvergleichen. Andreas hat Commit, Push und interne Veröffentlichung separat freigegeben. Die Website zeigt jetzt zusätzlich die AP5-Sammlung und bleibt ausschließlich lesend. Keine automatische Browser-Steuerungs-, AP6- oder weitere Trainingsfreigabe.

Vertrag, Grenzen und Bedienbeispiele: [AP5_EXPERIMENT_CONTROL.md](AP5_EXPERIMENT_CONTROL.md).

## Geliefert

- Verifizierte Parent-Ableitung mit neuer Hypothese, kanonischem Snapshot und expliziten Änderungen/Kategorien. Neue Dateien werden nicht überschrieben; unveränderte Konfigurationen können als begrenzter Replay-Job gestartet werden.
- Konfigurationsversion 3 und sicheres MLP-JSON v2 für zwei variable Hidden-Layer sowie Lernrate, Weight Decay, Batchgröße, Epochenlimit, Patience und min_delta. Feste CPU/float64/Ein-Thread-/Train-only-Skalierungsregeln bleiben erhalten. Alte Konfigurationen v1/v2 und Modell-JSON v1 bleiben nach alter Semantik lesbar.
- Private lokale Warteschlange mit Status, Parent, Run-ID, Budgets, unveränderlicher Submission, Ereignisprotokoll und tatsächlicher CPU-Terminierungs-Evidenz. Ein Worker pro Queue; feste Python-Modulaufrufe statt beliebiger Kommandos.
- Sofortiger Abbruch vorgemerkter Jobs; kontrollierter TERM/KILL-Abbruch laufender eigener Prozesse, Wall-/CPU-/Adressraumgrenzen, Elternprozess-Todesschutz und Recovery ohne automatische Wiederholungen.
- Deklarative Suchpläne mit höchstens vier eindeutigen Varianten, vollständigem Preflight und atomarer Kapazitätsprüfung. Auswahl nach Validierungs-RMSE, Handelsergebnis separat; inkompatible Bedingungen erhalten keine Bestenauswahl.
- V3-Signaturen erfassen zusätzlich die SMA-Referenzperioden. Geänderte Modellarchitekturen werden mit tatsächlicher Parameterzahl dargestellt, nicht mit dem festen AP3-Text.
- Kein Browser-Start/Stop, keine Upload-/Shell-/Job-HTTP-Endpunkte, keine öffentliche Freigabe und keine neuen 2025-Modellmetriken.

## Tatsächlich ausgeführte Prüfungen

- Stabile Vollsuite mit Torch: **1.143 bestanden in 280,05 s**.
- Stabile Vollsuite ohne Torch/NumPy: **1.135 bestanden, 8 reine Trainingstests übersprungen in 256,10 s**.
- Fokussierte Modell-/Integrationsnachprüfung: unabhängiger Review ohne offene Befunde; 22 Tests und zusätzlicher bytegenauer Legacy-v2-Reportvergleich.
- Job-Regressionen mit echten synthetischen Subprozessen: laufender/queued Abbruch, Wall-Timeout, nativer CPU-Hardkill, gewöhnlicher SIGKILL, Controller-Tod, TERM→KILL, einzelne Worker, Queuekapazität/Batch-Rollback und Crashfenster vor/nach Registry-Publikation. Originaldaten werden dafür nicht benutzt.
- Graphify code-only aktualisiert/abgefragt: 1.692 Knoten, 5.639 Kanten. Inline-Frontend ist nicht AST-abgedeckt; echte Browserprüfung zusätzlich erfolgt.

## Echter begrenzter Variantenvergleich

Quelle und Zeitvertrag unverändert: Dukascopy DEUIDXEUR m15, dieselben vier kausalen Merkmale, Lookback 8, Horizont 2, Training 2021–2023 und Entwicklung/Validierung 2024. **18.528 Trainings- und 6.192 Validierungsbeispiele**. 2025 vor Feature-Erstellung ausgeschlossen.

In einer frischen privaten Registry wurde der akzeptierte AP3-Parent bytegleich kopiert, nicht verändert oder als neu trainierter Lauf ausgegeben. Daraus wurden zwei deklarierte Lernratenvarianten abgeleitet. Beide: Seed 0, Architektur 4→16→8→1, höchstens 20 Epochen, alle übrigen Einstellungen/Kosten gleich.

- Lernrate **0.001**: 20 Epochen, beste Epoche 20; MAE **10,0951522017 bp**, RMSE **13,8632037293 bp**. **1.312 Trades**, Kosten **2.886,400 Punkte**, netto **−2.473,892 Punkte**.
- Lernrate **0.0005**: 20 Epochen, beste Epoche 20; MAE **10,0934172000 bp**, RMSE **13,8679949211 bp**. **1.723 Trades**, Kosten **3.790,600 Punkte**, netto **−3.684,999 Punkte**.

Die niedrigere Lernrate verbessert den MAE minimal, verschlechtert aber das erklärte Auswahlmaß RMSE und das simulierte Nettoergebnis. Die Auswahl nennt deshalb 0.001 als niedrigsten RMSE **innerhalb dieser zwei Varianten**, nicht als Optimum oder profitable Strategie. 2024 bleibt Entwicklungs-/Selektionsdatenbasis, kein unabhängiger Abschlusstest. Punkte je fester Einheit sind keine EUR.

- Registry: `var/market-trader/experiments/ap5-acceptance-v1`.
- Queue: `var/market-trader/jobs/ap5-acceptance-v1`.
- Kopierter AP3-Parent: `run-bb6a039cf4434971938091be691b8fe1`.
- Job 0.001: `job-7f54ec3deed94f6b85660f1ad94774d7`, Run `run-ff5344830dd5430a8bb42472d88f3df1`.
- Job 0.0005: `job-b809b8a8e7434a8183a73f148d994d5d`, Run `run-da62617bf38d4416b171ef09544b5824`.
- Ein zusätzlicher Replay der zweiten Variante: **alle 22 Backend-Artefakte bytegleich**. Exakte Replay-ID und vollständige Zahlen stehen im privaten Nachweis `acceptance/ap5-real-v1/verification.json`.
- Alle **6.192** gespeicherten Prognosen exakt durch Reload ohne Torch/NumPy reproduziert. AP4-Projektion der zwei neuen v3-Läufe tatsächlich kompatibel.
- Insgesamt genau **vier neue echte Fits**: zwei Varianten, ein Repo-Replay, ein installiertes Wheel-Replay. Jeder mit max20 Epochen und Wall300s/CPU180s/Adressraum4096MiB. Summe der vier Wall-Jobobergrenzen **1.200 s**, kein unbeschränkter Suchlauf.

## Separat installiertes Paket und Browser

- Wheel `releases/ap5-control-20261116-r1/market_trader-0.0.0-py3-none-any.whl`, **124.967 Bytes**, SHA-256 `aaa452c4ebb3ca094d47a73ba245ff8810832883f4072dbf2d9810517517db26`.
- Alle **37 Paketdateien** bytegleich zum aktuellen Checkout. Source-ID `sha256:c618fa8671120451b99d87f18427ce46a8c8f49cc0fb132b5e8a7ec2bec6b582` stimmt mit den echten Läufen überein.
- Neue Umgebung `.venvs/market-trader-ap5-release`, installiertes Wheel, außerhalb Checkout geprüft. Zuerst ohne Torch/NumPy: alle 6.192 Prognosen und kompatibler Read-only-Vergleich erfolgreich. Danach bekannter CPU-Stack ausschließlich aus bestehendem hashgesperrtem Offline-Wheelhouse installiert; `pip check` grün.
- Vierter realer Fit aus diesem installierten Wheel liefert **identische Modellbytes, Prognosebytes und fachliche Summarys** zur 0.0005-Variante. Andere Git-/Lock-Metadaten im installierten Kontext werden nicht als vollständige Kontext-Bytegleichheit ausgegeben.
- Temporäre Loopback-Instanz mit neuen AP5-Läufen: echte Chromium-Desktop-/320px-Vergleichs-/Detailabläufe, sichtbare Lernratendifferenz, echte negative Ergebnisse und Diagramme. Screenshots betrachtet, kein Gesamtseiten-Overflow, keine App-Fehler oder externen App-Aufrufe. Breite Tabellen sind intern horizontal scrollbar. Dies war **kein Live-Deployment**.

## Unabhängige Findings und Korrekturen

1. Nativer CPU-Code konnte das harte CPU-Limit erreichen, bevor Python SIGXCPU behandelt; SIGKILL wurde fälschlich `failed`. Jetzt exaktes Child-`wait4`-Accounting plus beobachtete Kernelgrenzen; nur bestätigte CPU-Erschöpfung wird `timed_out`, andere SIGKILL bleiben Fehler. Echte native und gewöhnliche Kill-Proben korrigiert.
2. Worker-Tod zwischen Registry-Anlage und Executor konnte einen laufenden Run ohne Job-Link hinterlassen. Optionaler `on_allocated`-Hook persistiert jetzt die exakte Zuordnung unmittelbar nach exklusiver Verzeichnisanlage und **vor** erster Config-/Run-Publikation. Recovery betrifft nur den zugeordneten Run; noch unveröffentlichte leere Anlagen werden nicht zu Ergebnissen erfunden.
3. Bericht nannte bei variabler Architektur weiterhin 225 Parameter. Jetzt tatsächliche V3-Architektur/-Parameterzahl; alte v2-Berichtbytes bleiben gleich.
4. Geänderte SMA-Fastperiode war als Bedingung klassifiziert, aber nicht verglichen. V3-Signatur enthält jetzt beide Referenzperioden; der Test verhindert fälschliche Auswahl bei abweichenden Bedingungen.
5. Zusätzliche Härtung: Plan-Provenienzvergleich verwendet kanonische JSON-Bytes und weist bool/int-Aliase ab.

Alle Korrekturen mit beobachteten RED→GREEN-Regressionen. Modell-/Integrationsnachreview und unabhängiger finaler Lifecycle-Nachreview grün, ohne offene Sicherheits-/Logikbefunde. Der Lifecycle-Nachreview führte 142 fokussierte Tests ohne Torch aus (182,93 s), bestätigte echte native CPU-Hardkill-/gewöhnliche SIGKILL-/Abbruchproben und die Recovery-Isolation samt unveröffentlichten Anlagen.

## Nachweise und unveränderte Grenzen

- `acceptance/ap5-real-v1/verification.json`, `worked.json`, `selection.json`, `replay-worked.json`; Treiber `exercise_ap5.py`.
- `acceptance/ap5-installed-core.json`, `ap5-installed-wheel-verification.json`; Treiber `exercise_ap5_installed.py`.
- `qa/ap5-installed-v2/verification.json` und PNGs; Treiber `acceptance/ap5_browser.mjs`. Frühere QA-v1 ohne zusätzliche scrolled mobile Bilder bleibt historisch.
- `acceptance/ap5-final-immutability.json`: sämtliche **125** Original-Registry-Dateien und **sechs** veröffentlichte Reports unverändert, ursprünglicher Releasepointer/Unit unverändert; Live-AP4 weiterhin sieben alte Läufe/read-only/HTTP200.
- `acceptance/ap5-review-model-followup.json` und `acceptance/ap5-review-lifecycle-final.json`: eigenständige grüne Modell-/Integrations- und Lifecycle-Schlussreviews.

Linux/CPython-spezifischer lokaler MVP, keine Sandbox gegen den eigenen vertrauenswürdigen Unix-Benutzer. `Popen`-Wait-Lock ist eine private CPython-Schnittstelle. CPU-Accounting erlaubt einen Kernel-Tick; ein externer/OOM-Kill exakt an derselben CPU-Grenze ist nicht vollständig vom Limit-Kill unterscheidbar. Batch-Validierung und Ausnahme-Rollback sind keine Cross-file-Power-loss-/SIGKILL-Transaktionsgarantie. Keine automatische Wiederholung oder Flächensuche. Browser-Trainingsstart braucht weiterhin ausdrücklich entworfenen Zugriffsschutz/CSRF und eine separate sichere Bedienoberfläche.

## Interne Veröffentlichung am 16. November 2026

Nach ausdrücklicher Git-/Veröffentlichungsfreigabe wurde das akzeptierte r1-Wheel im separaten Serving-Environment installiert, `pip check` erfolgreich; alle 37 installierten Paketdateien stimmen exakt. Serving weiterhin ohne Torch/NumPy. Das CLI-Trainingsenvironment und dessen Jobs bleiben davon getrennt.

- Vorheriger AP4-Release, aktive Unit und Releasepointer gesichert unter `releases/ap5-deployment-backup-20261116T123756Z`; altes installiertes Paket vorab bytegleich mit dem Rollback-Wheel geprüft. `rollback.sh` liegt bereit; tatsächlicher Rückwechsel war nicht nötig.
- Nur `--registry ap5=.../ap5-acceptance-v1` ergänzt; beide alten Quellen, Bind/Port und bisherige LAN-/VPN-Allowlist samt übrigen Dienstlimits unverändert. Dienst active/enabled, Linger vorhanden. Kein neuer VPN-Clienttest behauptet.
- Live-HTTP-Abnahme: **11 Einträge, 10 completed / 1 failed**; darin der bytegleich kopierte Referenzparent, nicht elf neue Fits. Alle Originalmetriken, Kurven, Historien und Reportbytes abgeglichen, AP5-Varianten/Replay kompatibel, unterschiedliche alte/neue Laufbedingungen ausdrücklich inkompatibel. Während der HTTP-Prüfung alle **225 Registry-Dateien unverändert**.
- Startseite, Run-API, echter AP5-Vergleich, AP5-Detailbericht und bisheriger AP3-Bericht von zweitem LAN-Rechner `10.10.1.20` HTTP200 und bytegleich geprüft.
- Echter Browser auf Live-Adresse: Desktop-/320px-Vergleich, Parameter, Detailmetriken und Diagramme geprüft; PNGs betrachtet, keine App-Fehler oder externen App-Aufrufe. Bisherige sechs statische Report-URLs und Galerie bleiben unverändert.
- Keine Job-/Training-HTTP-Schnittstelle: `/api/jobs` bleibt404, POST auf Trainings-/Mutationsziele wird abgewiesen. Keine neuen Fits während Veröffentlichung; 2025 bleibt unbewertet.

Nachweise: `acceptance/ap5-live-20261116-http.json`, `ap5-live-20261116-lan.json`, `qa/ap5-live-20261116/verification.json` und Screenshots. Die vorherigen Aussagen zur unveränderten Live-Unit beschreiben den Abnahmestand **vor** dieser autorisierten Veröffentlichung. Keine automatische Freigabe für AP6/Bildmodelle oder weitere reale Fits.
