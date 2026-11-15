# AP2 – ausgeführte Abnahme

Stand: 15. November 2026. **AP2 ist technisch abgeschlossen:** Implementierung, echte CLI-Läufe, Wiederholung, installierter Build und unabhängiges Review einschließlich der abschließenden Fehlerbehandlung sind erfolgreich geprüft.

## Implementiert

- Strikte versionierte TOML-Konfiguration, aufgelöste JSON-Snapshots mit Quellhash.
- Echte Verarbeitung über vorhandenen Import, kausale Trainingsdaten, Ridge-Fit, eingefrorene Validierung, Backtests und überprüften HTML-Bericht.
- Private UUID-Läufe mit Parent, Name, Hypothese, Seed, Zeitstempeln, Status, Lebenszyklus-/Fehlerlog und Artefaktpfaden.
- `experiment run`, `list`, `show`, `replay`, `compare` als lokale CLI; keine HTTP-Trainingsendpunkte.
- Neue Läufe statt Überschreiben. Snapshots und fertige Artefakte werden bei Lesen/Wiederholung/Vergleich über Prüfsummen verifiziert.
- Vergleich meldet abweichende Bedingungen separat von Parametern und beschreibenden Metadaten.

## Tatsächlich ausgeführte Läufe

Private Registry: `/home/hermes/workspace/var/market-trader/experiments/ap2-acceptance-v2`.

- Referenz: `run-21821708675245e18ffea5be42092fa7`, completed.
- Replay: `run-6258c9e67af34888b205058b99524e14`, completed; Parent ist die Referenz.
- Bewusste Kostenabweichung: `run-c880cc2213044175bb36074d5927e99f`, completed; Parent ist die Referenz.
- Bewusster Quellhash-Fehler: `run-d9731eff51f8459a8321f7ae8289123c`, failed; Fehler und Status bleiben nachvollziehbar gespeichert.

Referenz und Replay sind in **allen 15 Backend-Artefakten bytegleich**. Ihre Run-IDs, Statuszeitstempel und Registry-Logs unterscheiden sich absichtlich. Replay liest den gespeicherten Snapshot, nicht die inzwischen eventuell geänderte Original-TOML.

Bei der Kostenprobe wurde ausschließlich der volle hypothetische Spread von 2.0 auf 3.0 geändert (plus Beschreibung). `compare` liefert `compatible: false` und nennt exakt `execution.spread`; keine versteckte Gleichsetzung der Handelsbedingungen. Das ist ein Funktionsnachweis, keine Strategieoptimierung.

Beim absichtlich falschen Quellhash endet die CLI mit Exitcode 1 und `market-trader: experiment run: sha256 source hash mismatch`, ohne rohen Traceback. Der Lauf ist failed, hat ein Fehlerlog und keine vorgetäuschten fertigen Artefakte.

Alle vier Records wurden über `list` und `show` erneut gelesen. Fertige Manifeste und Daten werden dabei verifiziert.

## Forschungsisolation und Ergebnisse

- 18.528 Trainingsbeispiele 2021–2023, 6.192 Validierungsbeispiele 2024.
- Exportierter Testsplit leer. Trainings-/Labelgrenzen geprüft; keine 2025-Modellmetriken.
- Quelle unverändert: SHA-256 `d7c694f295a863076b4b00ceb0a2adfbeed6f1b0458a2098a52cd171e24cef2f`.
- Standard-AP2-Prognosefehler und reconciled Handels-/Monatswerte stimmen exakt mit der akzeptierten AP1-Referenz überein.
- Validierungsdatensatz: `sha256:34eac837246a4f22e99b68e311aeb4e6b69ea9ddfdd0702a126511bf177b0522`.
- AP2-Modellbericht: `sha256:1de2f74b1779162fcb3f0aeb258ac32bf23ab6720fe89be4bfd36c6f5fd74632`.

AP2-Provenienzbeschreibungen sind eigene Artefaktmetadaten. Deshalb muss die Modellartefakt-ID nicht mit AP1 übereinstimmen, obwohl die fachlichen Ergebnisse gleich sind. Es wird keine Verbesserung der Strategie behauptet.

## Tests und installierbarer Build

Das unabhängige Erstreview fand keine blockierenden Befunde. Anschließend wurde ein zusätzlicher Randfall mit RED→GREEN gehärtet: syntaktisch gültiges JSON ohne Objektstruktur in `run.json` wird mit verständlichem Fehler statt AttributeError/Traceback abgewiesen. Die separate unabhängige Nachprüfung bestand 82 fokussierte Tests ohne Befund; die Parent-Gesamtsuite und echten Läufe wurden danach erneut ausgeführt. Die früheren `*-v1`-Abnahmen bleiben als historische Vorversion erhalten; maßgeblich ist hier `*-v2` mit dem korrigierten Wheel `-r2`.

- Parent-Gesamtsuite nach zusätzlicher Fehlerbehandlung: **786 passed in 114.49s**; `git diff --check` und `pip check` bestanden.
- Neue CLI-Ende-zu-Ende-Prüfung verwendet echte Pipeline/Registry mit kleinen synthetischen Kerzen, keine Attrappen für den Modellpfad.
- Config-/Pipeline-Prüfungen decken unter anderem variable Horizonte/Kohorten/Kosten, 2025-Isolation, Artefaktidentität und HTML-Escaping ab; Registry-Prüfungen Lebenszyklus, Fehler, Checksummen, IDs, Parent, Replay und Vergleich.
- Separates Wheel unter `var/market-trader/releases/ap2-experiments-20261115-r2/market_trader-0.0.0-py3-none-any.whl`.
- Wheel SHA-256: `31d73a63edb4c8f92a82d7f5f9591771b88b9afaf51c151f88ab1a5fda8670f2`.
- Installiert in `/home/hermes/workspace/.venvs/market-trader-ap2`; Kern ohne externe Laufzeitpakete. `pip check` erfolgreich. AP1-Forschungsenvironment und Webdienst-Wheel bleiben unverändert.
- Tatsächlicher Aufruf außerhalb des Checkouts erzeugte `run-2380823316ad4f03a66e6ad10309586e` in `var/market-trader/experiments/ap2-wheel-smoke-v2`. Geladen wurde das Modul aus site-packages, nicht das editierbare Repository. Modell-ID, Python-Quellkennung und Handelswerte stimmen mit dem Entwicklungslauf überein.
- Wheel enthält Registry, Pipeline und HTML-Template. Git-Kontext im installierten Wheel ist erwartungsgemäß anders als im Checkout; kein plattformübergreifender Bitgleichheitsanspruch.

Private Nachweise:
- `var/market-trader/acceptance/exercise_ap2.py`
- `var/market-trader/acceptance/ap2-real-execution-v2.json`
- `var/market-trader/acceptance/ap2-wheel-run-v2.json`

Diese Pfade liegen nicht im Webverzeichnis. Der akzeptierte AP1-Bericht und der laufende Webdienst wurden durch AP2 nicht verändert. AP2 ist noch keine neue Browser-Vergleichsoberfläche; diese folgt in AP4. Neuronale Modelle sind AP3, Job-Steuerung/automatische Suche spätere Pakete.

## Einrichtungs- und Bedienhinweise

Siehe [AP2_EXPERIMENTS.md](AP2_EXPERIMENTS.md) für Vertrag und CLI. Vom Repository-Stamm aus:

```sh
/home/hermes/workspace/.venvs/market-trader-ap2/bin/market-trader experiment run \
  --config configs/ap2-reference.toml \
  --registry /home/hermes/workspace/var/market-trader/experiments/my-research
```

Run-ID aus der JSON-Ausgabe für `show`, `replay` und `compare` übernehmen. Die jeweils neue Registry-Run-ID ist exakt zu erhalten, nicht zu normalisieren. Unterschiedlicher Code/Runtime-Kontext kann als abweichende Vergleichsbedingung erscheinen; ein Replay garantiert keinen unveränderten externen Datenbestand oder unveränderte Software.
