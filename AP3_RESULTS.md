# AP3 – ausgeführte neuronale Referenz

**AP3 ist technisch abgeschlossen und der Bericht intern veröffentlicht.** Reale Abnahme, zwei unabhängige Modell-/Integrationsreviews und die gesonderte Layout-Nachprüfung sind erfolgreich. Automatisierte visuelle Browserprüfung bleibt umgebungsbedingt blockiert; HTTP-/Byte-Prüfung lokal und von einem zweiten LAN-Rechner ist bestätigt. Vertrag: [AP3_NEURAL_REFERENCE.md](AP3_NEURAL_REFERENCE.md), Environment/Hashes: [AP3_ENVIRONMENT.md](AP3_ENVIRONMENT.md).

## Tatsächlich trainiertes Modell

- MLP 4→16→8→1, ReLU, linearer Ausgang, 225 trainierbare Parameter.
- AdamW, Lernrate0.001, WeightDecay0.0001, Batch128, maximal100Epochen.
- Seed0, CPU, ein Thread, float64, deterministische Operationen.
- Eingabe- und Zielskalierung nur aus Trainingsdaten; Rücktransformation in ursprüngliche Renditeeinheiten vor Prognosefehlern und Handel.
- EarlyStopping: Geduld10, relevante Verbesserung1e-6 in standardisiertem Validierungs-MSE. Tatsächlich **30Epochen**, ausgewählt und wiederhergestellt: **Epoche20**.
- Keine Architektur-, Parameter- oder Seed-Suche. Wiederholungen dienen Reproduzierbarkeit und Paket-/Darstellungsprüfung, nicht der Auswahl eines besseren Ergebnisses.

Zeitliche Grundlage bleibt unverändert: **18.528 Trainingsbeispiele 2021–2023**, **6.192 Validierungsbeispiele 2024**, testsplit leer. 2024 dient auch Checkpoint-Auswahl/EarlyStopping und ist deshalb Entwicklungs-/Selektionsdatenbestand, kein unabhängiger Abschlusstest. **2025 wurde nicht modelliert oder bewertet**; der reine Quellen-/Qualitätsimport darf es enthalten.

## Ergebnisse 2024

Deutsches Dezimalkomma; Prognosefehler in Basispunkten der Bruttorendite:

- MLP: MAE **10,0952bp**, RMSE **13,8632bp**.
- Ridge: MAE **10,0878bp**, RMSE **13,8698bp**.
- Nullprognose: MAE **10,0914bp**, RMSE **13,8603bp**.

Das Netz zeigt damit keinen überzeugenden Vorteil gegenüber der Nullprognose: sein MAE und RMSE sind beide etwas höher. Es hat einen etwas niedrigeren RMSE als Ridge, aber einen höheren MAE. Kein Nachweis statistischer Signifikanz oder Profitabilität.

Handelsabbildung unter identischem vorab festgelegtem hypothetischem Kostenprofil:

- **MLP:**1.312Trades; Referenzbewegung+412,508; Kosten2.886,400; netto**−2.473,892 Punkte**; maximaler Drawdown2.731,112.
- **Ridge:**1.784Trades; Referenzbewegung−395,140; Kosten3.924,800; netto**−4.319,940 Punkte**; maximaler Drawdown4.593,783.
- **SMA:**1.297Trades; Referenzbewegung+1.062,135; Kosten2.853,400; netto**−1.791,265 Punkte**; maximaler Drawdown1.890,890.

Punkte je fester Einheit, keine Eurobeträge. Voller Spread2.0, Slippage0.1jeFill, separate Kommission0, Raster0.001, Menge1, synthetische Startmittel100000Q. Keine bestätigten Brokerkonditionen und keine historische Ask-Fill-Rekonstruktion aus den Bid-Daten. Alle Wege verlieren nach Kosten. Ein geringerer Verlust als Ridge ist kein Beleg für ein wirtschaftlich brauchbares Netz.

## Echte Läufe und unabhängige Nachrechnungen

Maßgebliche Registry: `/home/hermes/workspace/var/market-trader/experiments/ap3-acceptance-v2`.

- Referenz: `run-bb6a039cf4434971938091be691b8fe1`.
- Snapshot-Replay: `run-294d566d9b7a4efca5f68cb1cd08dcb8`, neuer Parent-verknüpfter Lauf.
- Separat neu gerechneter Ridge-Lauf: `run-bb3169c4473f45fcb59e611bc606f2ae`.
- Referenz/Replay: **alle22Backenddateien bytegleich**, einschließlich Gewichten, Historie, Prognosen, Backtests, HTML und Manifest. Registry-IDs/Zeitstempel unterscheiden sich absichtlich.
- MLP und separat gerechneter Ridge werden von der Registry als kompatible Daten-/Kosten-/Laufzeitbedingungen erkannt; Modellwahl ist expliziter Konfigurationsunterschied.
- Alle train-only Mittelwerte/Standardabweichungen für Eingaben/Ziel wurden aus den exportierten Trainingszeilen separat nachgerechnet und exakt mit dem Modell verglichen.
- Alle6.192 gespeicherten Validierungsprognosen stimmen exakt mit JSON-Reload-Inferenz überein. Dasselbe wurde im alten Environment ohne installierteTorch-/NumPy-Pakete und ohne entsprechende Imports bestätigt.
- Die aus den gespeicherten Gewichten nachgerechnete standardisierte Validierungs-MSE ist **0.5057045305406469** und stimmt mit dem tatsächlichen besten Checkpoint überein; Auswahl- und Stoppangaben wurden geprüft.
- Sämtliche1.312MLP-,1.784Ridge- und1.297SMA-Trades wurden zusätzlich gegen die unveränderten Quell-Opens, zulässige Entscheidungskohorte,30Minuten-Ausgang, gleiche lokale Sitzung und2.2Kostenpunkte proTrade reconciled. Monats-/Kontosummen werden über den bestehenden Ledger geprüft.
- Rohquelle unverändert: `sha256:d7c694f295a863076b4b00ceb0a2adfbeed6f1b0458a2098a52cd171e24cef2f`.

Die vorherige `ap3-acceptance-v1` ist eine erhaltene Vorversion vor reinem Berichtslayout-Fix. Architektur, Training, gewählte Epoche, Modellkennung und fachliche Ergebnisse blieben unverändert; veröffentlicht werden soll nur der finalev2-Bericht. Die zusätzliche Regression prüft Panel-Platzierung im bestehenden Hauptlayout und horizontal scrollbare Tabellen, keine neue Modelloptimierung.

Private Prüfskripte/Nachweise:
- `var/market-trader/acceptance/verify_ap3.py`
- `var/market-trader/acceptance/ap3-first-run-v2.json`
- `var/market-trader/acceptance/ap3-real-verification-v2.json`
- `var/market-trader/acceptance/ap3-wheel-run-v2.json`

## Interner Bericht und Review

Veröffentlicht: `http://10.10.1.136:8086/reports/ap3-neural-2024-reference.html` (auch in der bisherigen Übersicht). Publiziert wurde exakt die validierte HTML-Datei des finalen v2-Referenzlaufs, ohne Rohdaten/Modell-JSON freizugeben.

- HTTP 200 und vollständige Bytegleichheit lokal und über den zweiten LAN-Rechner `10.10.1.20` bestätigt.
- HTML: 3.912.328 Bytes; SHA-256 `cb5310b583b4eb382cb735ee4bc5a6e62b2e284e818bff0f2613431817ce5bd9`.
- JSON/Traversal mit 404, fremder Host mit 403, POST mit 405 abgewiesen. Dienst blieb aktiv/enabled; kein Neustart, Wheel-Austausch, ACL- oder Infrastrukturwechsel des Webdienstes.
- Nachweis: `var/market-trader/acceptance/ap3-http-verification.json`.
- Unabhängiges Kernreview: keine Befunde, 104 fokussierte Tests mit Torch sowie 98/6 ohne Torch und zusätzliche synthetische Prüfungen.
- Unabhängiges Integrationsreview: keine Befunde; Modellwahl, gemeinsame Kohorte/Kosten, Zeitgrenzen, Reporteinheiten und nichtlineare Attribution geprüft.
- Separate Nachprüfung des Präsentationsfixes: keine Befunde; 29 Layout-/Integrations-/Reporttests und 16 Server-/CLI-Tests plus Escaping-/Fragmentprüfungen bestanden. Keine visuelle Pixelprüfung behauptet.

## Tests, Build und Grenzen

- AP3-Environment: **891 passed in 147,64 s**.
- Ursprüngliches Core-Environment ohne Torch: **885 passed, 6 reine Trainingstests übersprungen; 107,33 s**. PurePython-Modellladen, Inferenz, Strategie und Bericht bleiben geprüft; übersprungen wird nur optionales Torch-Training.
- `pip check` erfolgreich; Runtime-Lock mit11 tatsächlich heruntergeladenen CPU-Wheels und Hashes, ohne Veränderung des stdlib-only Kerns.
- Separates installiertes Wheel: `var/market-trader/releases/ap3-neural-20261115-r2/market_trader-0.0.0-py3-none-any.whl`.
- Wheel-SHA256: `6efa0e0bf6b9f6570145150a6b156f4357fb874516f5733ea67786fb832e008b`.
- Frische Installation aus hashgeprüfter privater Wheelhouse in `/home/hermes/workspace/.venvs/market-trader-ap3-release`. Echter Lauf außerhalb des Checkouts: `run-60fb40c0f4d043e186f5e4feaf36dd25` in `experiments/ap3-wheel-smoke-v2`; Modell-ID, Prognosen und Handelswerte stimmen exakt mit der Entwicklungsausführung überein.
- Modellbericht: `sha256:f6f982b991dd671aeb4a2946c5e92a85aac4cc730a01f0dcbd6b23b04a844e17`.
- Python-Quellkennung: `sha256:3554a8d6cab24192241c06bb1219006f8e2894d5638e3771cb47af8ace56ea24`.

Das installierte Wheel hat erwartungsgemäß keinen Checkout-Git-Kontext; die im Repository liegende Runtime-Lockdatei wird dort nicht automatisch als Projektdatei gefunden. Die Runtime wurde ausdrücklich vor Installation mit dieser Hash-Lockdatei eingerichtet und geprüft. Keine Bytegleichheit über verschiedene Metadatenkontexte/Software/Hardware behauptet; im selben getesteten Kontext ist Replay geprüft.

Automatisierte visuelle Browserprüfung aktuell blockiert: Chromium startet nicht wegen fehlender `libatk-1.0.so.0`. Keine privilegierten Systempakete installiert. HTML-Struktur, Escaping und Darstellungsvoraussetzungen sind getestet, aber keine Screenshot-/Pixelprüfung behauptet. Die erfolgreiche HTTP-/Byte-Prüfung ersetzt keine visuelle Kontrolle auf dem Client.

## Bedienung

Nach Einrichtung gemäß `AP3_ENVIRONMENT.md`, aus dem Checkout:

```sh
/home/hermes/workspace/.venvs/market-trader-ap3/bin/market-trader experiment run \
  --config configs/ap3-neural-reference.toml \
  --registry /home/hermes/workspace/var/market-trader/experiments/my-neural-runs
```

NeueKonfigurationv2 benötigt `[model] kind = "mlp"` oder `"ridge"`. V1-Ridge-Snapshots bleiben unverändert unterstützt. JSON-Modellablage statt pickle/torch.load; endliche Gewichte, Dimensionen, Ziel-/Feature-/Zeitverträge und Prüfsummen werden geprüft. Nichtlineare Prognosen bekommen keine erfundenen additiven Ridge-Merkmalsbeiträge.

AP3 liefert keine Bildmodelle, Parametersuche, Browser-Trainingsendpunkte oder Live-Trading. AP4 ist die nächste vorgeschlagene Vergleichsoberfläche, nicht automatisch freigegeben.
