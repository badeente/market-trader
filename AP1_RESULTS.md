# AP1 – ausgeführter Referenzlauf

Stand: 15. November 2026. Der Experimentvertrag wurde von Andreas freigegeben.
**AP1 technisch abgeschlossen: Offline-Abnahme, unabhängiges Review nach Korrektur und interne Veröffentlichung erfolgreich. Die automatisierte visuelle Browserprüfung bleibt als unten dokumentierte Umgebungsgrenze offen.**

## Was tatsächlich ausgeführt wurde

- Protokoll `ap1-deuidxeur-open30-v1`, keine Parametersuche.
- Dukascopy `deuidxeur`, 15-Minuten-Bid-Daten; die alte unbekannte GER30-Serie wurde nicht beigemischt.
- Forschungsfilter: vollständige beobachtete Werktage 09:00–17:30 Europe/Berlin; kein offizieller Produktkalender und keine Vollständigkeitszertifizierung.
- Training 2021–2023: **18.528 Beispiele**; Validierung 2024: **6.192 Beispiele**.
- Acht abgeschlossene Kerzen, vier bestehende kausale Merkmale, neues versioniertes 30-Minuten-Open-to-Open-Ziel.
- Ridge mit festem Alpha 0.1 und ausschließlich am Training gefitteter Skalierung; Nullprognose als statistische Referenz.
- Ridge-Long-Regel und SMA-4/8-Zeitregel im selben Ausführungsledger; maximal eine Einheit, Ausstieg nach 30 Minuten, keine Übernachtpositionen.
- **2025: keine Features für Modelle, keine exportierten Trainings-Testzeilen, keine Modell-/Handelsbewertung.** Der Rohimport und das strukturelle Qualitätsmanifest enthalten das Jahr bewusst; das ist keine Modellbewertung.

## Ergebnisse – nur Validierung 2024

Vorhersagefehler in Basispunkten der Bruttorendite, gerundet:
- Ridge: MAE **10.088 bp**, RMSE **13.870 bp**.
- Nullprognose: MAE **10.091 bp**, RMSE **13.860 bp**.

Ridge hat somit praktisch keinen MAE-Vorsprung; beim RMSE ist die Nullprognose etwas besser. Daraus wird kein belastbarer Prognosevorteil abgeleitet.

Handelsrechnung in Punkten je fester Einheit:
- Ridge: **1.784 Trades**; Referenzbewegung **−395,140**; modellierte Kosten **3.924,800**; netto **−4.319,940**; maximaler Drawdown **4.593,783 Punkte**.
- SMA-Zeitregel: **1.297 Trades**; Referenzbewegung **+1.062,135**; modellierte Kosten **2.853,400**; netto **−1.791,265**; maximaler Drawdown **1.890,890 Punkte**.

Diese Handelswerte sind deutsch formatiert (Dezimalkomma). Vollpräzise maschinenlesbare Werte mit Dezimalpunkt stehen in `summary.json`.

Beide Regeln verlieren unter dem vorab festgelegten Kostenprofil. Das ist ein Forschungsergebnis, kein technischer AP1-Fehlschlag. Keine nachträgliche Schwellen-/Periodenoptimierung; 2025 bleibt reserviert.

Kosten sind **hypothetisch**, keine bestätigten Brokerkonditionen: voller Spread 2.0, Slippage 0.1 je Fill, Kommission 0, Raster 0.001, Menge 1, synthetische Startmittel 100000 Q. Bid-Quelle plus symmetrische Offsets rekonstruiert keine historischen Ask-Fills. Keine Euro-Gewinn-, Kontogrößen- oder Handelsfähigkeitsempfehlung.

## Datenqualitäts- und Korrektheitsnachweis

- Rohdatei m15: 114.324 Zeilen, SHA-256 `d7c694f295a863076b4b00ceb0a2adfbeed6f1b0458a2098a52cd171e24cef2f`.
- Zugelassen: 1.287 vollständige Forschungstage / 43.758 Kerzen über 2021–2025.
- Zwei partielle Tage ausgeschlossen: 19. Juli 2024 (5 fehlende Starts), 20. Januar 2025 (9 fehlende Starts); insgesamt 54 beobachtete Kerzen ausgeschlossen.
- 15 ganz fehlende Werktagsfenster separat protokolliert; keine erfundenen Feiertagsnamen oder künstlichen Kerzen.
- 70.512 Quellzeilen liegen außerhalb des Forschungsfensters.
- Eine im Abschlusscheck gefundene Audit-Randabweichung wurde mit RED→GREEN korrigiert: Ein lokales Datum, dessen gesamtes Forschungsfenster außerhalb der UTC-Abdeckung liegt, wird nicht als fehlender Quelltag deklariert. Das verhindert insbesondere einen falschen Eintrag für 1. Januar 2026. Modell und Validierungszahlen blieben unverändert.
- Vier Labels direkt gegen die unveränderten CSV-Open-Werte rational nachgerechnet.
- Alle 1.784 Ridge- und 1.297 SMA-Trades zusätzlich gegen Quell-Opens, zulässige Entscheidungszeitpunkte, 30-Minuten-Ausstieg, gleiche lokale Sitzung und exakt 2.2 Kostenpunkte je Trade geprüft.
- Keine offenen Positionen, ausstehenden Orders oder Rejections akzeptiert; Trade-/Monats-/Equity-Summen reconciled.
- Vollsuite nach Korrektur: **653 passed**, 35.70 s; `git diff --check` und `pip check` erfolgreich.

Die Einträge vollständiger/fehlender Tage sind strukturelle Angaben. Sie belegen keine Datenlizenz, tatsächliche Börsenöffnungszeiten, Anbieter-Vollständigkeit oder wirtschaftliche Nutzbarkeit. Quellrecht für Weiterverteilung ist nicht zertifiziert; keine öffentliche Rohdatenveröffentlichung.

## Reproduzierbarkeit und Artefakte

Privater Laufstamm: `/home/hermes/workspace/var/market-trader/runs/`.
- `ap1-deuidxeur-open30-v1-c` und `ap1-deuidxeur-open30-v1-d`: **alle 15 Dateien bytegleich**, einschließlich Manifest und HTML.
- Vorläufige Läufe `-a` und `-b` entstanden vor der Audit-Randkorrektur und sind keine abgenommenen Veröffentlichungsquellen.
- `ap1-deuidxeur-open30-v1-wheel`: separater realer Lauf aus einem installierten Wheel außerhalb des Checkouts. Identische Datensatz-/Modellkennungen, Modellparameter, Prognosefehler und Handelswerte. Git-Metadaten unterscheiden sich erwartungsgemäß vom editierbaren Checkout; deshalb sind die laufzeitbeschrifteten Berichte nicht bytegleich mit `-c`.

Wichtige private Artefakte je Lauf:
- `research-dataset.json`, `development-dataset.json`, `validation-dataset.json`
- `import-audit.json`, `reference-policy.json`, `label-checks.json`
- `training-data.json`, `ridge-model.json`
- `validation-evaluation.json`, `validation-predictions.json`
- `ridge-backtest.json`, `sma-backtest.json`, `summary.json`, `report.html`, `manifest.json`

Prüfsummen im Manifest beziehen sich auf die tatsächlich geschriebenen Bytes. Das vollständige Manifest wird zuletzt veröffentlicht; vorhandene Zielverzeichnisse werden nie überschrieben.

Kennungen:
- Validierungsdatensatz: `sha256:34eac837246a4f22e99b68e311aeb4e6b69ea9ddfdd0702a126511bf177b0522`
- Modellbericht: `sha256:7ffe642a97731ec5bbff7d7238ac96113ab5ad6a5ba958d02bb23d40238badf6`
- Python-Quellkennung: `sha256:a2d9102ea91b51c581ada594109243bb7b8a81abda18a44a7282a6a21e656e9b`

Abnahmenachweis: `var/market-trader/acceptance/ap1-verification.json`; überprüfendes Skript daneben `verify_ap1.py`. Diese liegen außerhalb von Git und Webverzeichnis. Das Skript liest und prüft die abgenommenen Läufe; es trainiert oder bewertet 2025 nicht.

Wheel: `var/market-trader/releases/ap1-reference-20261115/market_trader-0.0.0-py3-none-any.whl`.
SHA-256: `3c4b98d1b9b52a22a9ed4f946117d43878c55d6c471023df11cebbe39d6f130b`.
Separat installiert unter `/home/hermes/workspace/.venvs/market-trader-ap1` (kein Austausch des laufenden Webdienst-Wheels).

## Offline wiederholen

Nach Einrichtung sind keine Netzwerkzugriffe, APIs, GPU oder Fremdpakete für den Lauf erforderlich. CPython 3.14 wird vorausgesetzt; geprüft mit 3.14.9. Für dieselben Bytes müssen Quellbestand, Code, Laufzeit/Zeitzonendaten und Metadatenkontext gleich bleiben. Keine Gleichheit über beliebig veränderte Umgebungen behauptet.

Vom Checkout aus, mit neuem Zielnamen:

```sh
/home/hermes/workspace/.venvs/market-trader-ap1/bin/market-trader reference \
  --config configs/ap1-reference.toml \
  --output /home/hermes/workspace/var/market-trader/runs/ap1-my-reproduction
```

Die vier TOML-Felder sind `version`, `experiment`, `source_csv`, `source_sha256`; relative Eingabepfade gelten relativ zur TOML-Datei. Keine frei änderbaren Parameter oder Testzeiträume in AP1. Das erzeugte `reference-policy.json` enthält die eingefrorenen Forschungsannahmen. Ein vorhandenes Ziel oder falscher Quellhash führt zum Fehler, nicht zum Überschreiben oder Ausweichen auf andere Daten.

## Bericht und Restgrenzen

Veröffentlicht unter `http://10.10.1.136:8086/reports/ap1-deuidxeur-2024-reference.html`; auch in der Startseitenübersicht verlinkt. Exakt das validierte `report.html` aus Lauf `-c` wurde neu ins bestehende flache Reportverzeichnis geschrieben. Rohdaten, Modell-JSON und Trainingsdaten bleiben privat. Webdienst-Wheel, Netzwerkfreigaben und laufender Service wurden nicht geändert oder neu gestartet.

- Unabhängiges Erstreview fand ausschließlich die oben beschriebene Audit-Randabweichung; nachfolgendes unabhängiges Review bestätigte den Fix, 55 fokussierte Tests und fünf zusätzliche synthetische Grenzprüfungen, ohne neue Befunde.
- HTTP 200, Übersichtseintrag und Bytegleichheit mit dem privaten Original sind geprüft.
- SHA-256 des veröffentlichten HTML: `0360bcb58c1d66e4ac23e7bb2b3ea4c4c305332f4e2d80e661b61ea6f052dba0`.
- Derselbe Hash wurde über HTTP vom zweiten LAN-Rechner `10.10.1.20` bestätigt. Der Dienst blieb aktiv; er ist enabled.
- JSON und Traversal werden mit 404, fremder Host mit 403, POST mit 405 abgewiesen. Lokaler Nachweis: `var/market-trader/acceptance/ap1-http-verification.json`.
- Beim ersten Probeaufruf ohne das notwendige `/reports/`-Präfix wurde erwartungsgemäß 404 geliefert. Der oben angegebene tatsächliche Link wurde danach erfolgreich geprüft.

Automatisierte visuelle Prüfung ist blockiert: Chromium fehlt `libatk-1.0.so.0`. Keine privilegierten Systempakete installiert. HTTP-/Byteprüfung ist möglich und wird getrennt von einer echten Browser-Sichtprüfung ausgewiesen.

Noch keine AP2-Runverwaltung, Parameteroberfläche, neuronalen Netze oder Live-Trading implementiert. Kein Commit/Push freigegeben oder durchgeführt.
