# Datenvertrag: kanonische Kerzen und Normalisierung

Stand: 2026-11-02 · Status: Datenfunktionen, Zeiträume und lokale Persistenz geprüft

Dieser Vertrag konkretisiert DAT-001, DAT-003, DAT-007, CAU-001 und den numerischen
Teil von CAU-003 aus [TEST_CASES.md](TEST_CASES.md). Datentypen und Verarbeitung
liegen unter `src/market_trader/`. Import, Validierung, Resampling, Fenster und
numerischer Referenzmittelwert sind umgesetzt. Das zweite Arbeitspaket ergänzt
Sortierung, Deduplizierung, Qualitätsberichte und Datenkennungen für
DAT-004/005/009/010. Das dritte Paket ergänzt ausdrücklich konfigurierte
UTC-Sitzungen für DAT-006. Das vierte ergänzt angefragte Datenzeiträume und
vollständig fehlende Serien für DAT-005/006/009/010. Ergänzende Randprüfungen decken auch Teile von DAT-002/008
und CAU-002/004/005 ab. Der tatsächliche Teststand steht in [TESTING.md](TESTING.md).
Das fünfte Paket ergänzt lokales Speichern/Laden mit Herkunftsmanifest und
Prüfung von Daten- und Artefaktkennungen; siehe [STORAGE_CONTRACT.md](STORAGE_CONTRACT.md).

## Datentyp Candle

`Candle` ist ein unveränderlicher Datenträger. Der Konstruktor führt keine
fachliche Validierung durch. Validierung und Dateiimport bilden die Daten-Grenze.

| Feld | Python-Typ | Bedeutung |
| --- | --- | --- |
| `instrument` | `str` | Explizite Instrumentkennung; `TEST` bezeichnet nur ein synthetisches Beispiel. |
| `interval` | `datetime.timedelta` | Dauer der Kerze; kein implizites festes Handelsintervall. |
| `open_time` | `datetime.datetime` | Zeitzonenbehafteter UTC-Beginn des halboffenen Intervalls. |
| `close_time` | `datetime.datetime` | UTC-Ende; `close_time = open_time + interval`. |
| `available_at` | `datetime.datetime` | Frühester Informationszugriff; mindestens `close_time`, gegebenenfalls später. |
| `open`, `high`, `low`, `close` | `decimal.Decimal` | Preise in der dokumentierten Notierung des Instruments. |
| `volume` | `decimal.Decimal` | Nichtnegatives Volumen in der dokumentierten Einheit; Nachkommastellen bleiben erhalten. |

Kerzenintervalle sind `[open_time, close_time)`. Preise und Volumen werden aus
Dezimaltext eingelesen, ohne einen Zwischenweg über binäre Gleitkommazahlen.
Felder haben endliche Werte. `high >= max(open, close)` und
`low <= min(open, close)` müssen gelten; eine allgemeine Regel gegen negative
Preise wird ohne festgelegten Markt nicht eingeführt.

`CandleValidationError` erbt von `ValueError` und enthält `row_number` sowie
`field`. Zeilen zählen ab 1 unterhalb der CSV-Kopfzeile bzw. ab 1 in einer
übergebenen Sequenz. Ein Fehler bezeichnet das betroffene Feld oder bei
`field="series"` eine nicht angefragte Instrument-/Intervallkombination. Tests
vergleichen diese Angaben und nicht den exakten Wortlaut einer Meldung.
Fehler der CSV-Kopfzeile verwenden `row_number=0`; Datenzeilen beginnen bei 1.
Bei einer fehlenden Serie oder einer Lücke nach der letzten vorhandenen Kerze
gibt es keine nachfolgende Quellzeile: `DataGapError.row_number` ist dann `None`,
und die Fehlermeldung bezeichnet den Datensatz.

## Kanonisches CSV für den ersten Import

UTF-8, Komma als Trennzeichen, Punkt als Dezimalzeichen, eine Kopfzeile:

```text
instrument,interval_seconds,open_time,close_time,available_at,open,high,low,close,volume
```

`interval_seconds` ist eine positive ganze Zahl und wird zu `timedelta`.
Zeitstempel enthalten eine ausdrückliche Zeitzone, etwa `2026-01-01T00:00:00Z`.
Zonenbehaftete Eingaben werden nach UTC normalisiert; eine fehlende Zeitzone
wird nicht geraten. Quellen mit anderen Schemata erhalten später eigene Adapter.
Fehlende, doppelte oder unbekannte Kopfspalten und überzählige Zeilenwerte
werden als Datenfehler zurückgewiesen. Ein gültiger Header ohne Datenzeilen
liefert eine leere Folge; eine Datei ohne Header ist ungültig.
Der Import erhält die Zeilenreihenfolge. Sortierung und Deduplizierung übernimmt
die getrennte Normalisierung; ausdrückliche UTC-Sitzungen können dabei geprüft
werden. Anbieter für konkrete Börsenkalender folgen später. Für Fenster und
Resampling wird eine streng geordnete, überlappungsfreie Folge desselben
Instruments und Intervalls vorausgesetzt; ungültige Folgen werden abgewiesen.

Die erste versionierte CSV-Fixture liegt in
[`tests/fixtures/fractional_volume.csv`](tests/fixtures/fractional_volume.csv).
Historische Marktdatensätze bleiben weiterhin von Git ausgeschlossen.

## Aufrufbare Schnittstellen

Die folgenden Schnittstellen sind implementiert. Ihre Verarbeitung benötigt
ausschließlich die Python-Standardbibliothek.

| Schnittstelle | Rückgabe / erwarteter Vertrag |
| --- | --- |
| `market_trader.data.read_candles(path: Path)` | `tuple[Candle, ...]`; kanonisches CSV lesen und Eingaben validieren. Fehlende oder ungültige Werte erzeugen `CandleValidationError`. |
| `market_trader.data.validate_candles(candles: Sequence[Candle])` | `None` bei gültigen Eingaben, andernfalls `CandleValidationError`. Sequenz wird nicht verändert. |
| `market_trader.normalization.normalize_candles(candles, *, config: NormalizationConfig, requests: Sequence[DatasetRequest] \| None = None)` | `NormalizationResult` mit getrennten, unveränderlichen `NormalizedDataset`-Objekten; Validierung, Sortierung, Deduplizierung, Lückenprüfung und Inhaltskennung. Mit Aufträgen zusätzlich erwartete Serien und vollständige Zeiträume prüfen. |
| `NormalizationResult.select(*, instrument: str, interval: timedelta)` | Exakter Datensatz dieser Kombination; `KeyError`, falls nicht vorhanden. |
| `market_trader.data.resample_candles(candles, *, interval: timedelta)` | `tuple[Candle, ...]`; vollständige, ausgerichtete Intervalle eines Instruments aggregieren. Open zuerst, High Maximum, Low Minimum, Close zuletzt, Volumen Summe. |
| `market_trader.data.rolling_windows(candles, *, size: int)` | Iterator über vollständige `tuple[Candle, ...]`-Fenster; ein Schritt pro Kerze, keine vorgefüllten Warmup-Fenster. |
| `market_trader.representations.mean_close_features(candles, *, lookback: int)` | `tuple[MeanCloseFeature, ...]`; nach vollständigem Warmup ein arithmetischer Mittelwert der letzten `lookback` Schlusskurse je Kerze. |

Die Fenster-/Merkmalsfunktionen arbeiten auf einer validierten, chronologisch
geordneten Folge desselben Instruments und Intervalls; Fenstergröße und Lookback
sind positive ganze Zahlen. Für jede Ausgabe werden ausschließlich Kerzen bis
einschließlich ihrer Endkerze verwendet.
Der Fensteriterator sichert beim Aufruf eine unveränderliche Momentaufnahme der
Eingabefolge. Zielintervalle beim Resampling müssen positive ganzzahlige Vielfache
des Quellintervalls sein und werden am UTC-Epochenbeginn ausgerichtet. Ein
unvollständiger Zielblock erzeugt einen expliziten `CandleValidationError`, keine
stillschweigend ausgefüllte oder verworfene Kerze. Ganze positive Sekundenzahlen
im CSV sind das Austauschformat; interne `timedelta`-Werte behalten ihre Präzision.

`MeanCloseFeature` trägt Instrument, Intervall, `close_time` der Endkerze,
`available_at` als spätesten Verfügbarkeitszeitpunkt der beitragenden Kerzen und
den Dezimalwert `value`. Der kleine Referenz-Mittelwert ist ein prüfbarer
Darstellungsvertrag, keine ausgewählte Handelsstrategie.
Volumen wird ohne Verlust kleiner Summanden addiert. Mittelwerte verwenden
28 signifikante Dezimalstellen mit `ROUND_HALF_EVEN`, unabhängig vom globalen
Dezimalkontext des Aufrufers. Dies ist eine Regel der Merkmalsdarstellung, keine
vorgezogene Festlegung marktbezogener Preis- oder Mengenraster.

Ein Fensteriterator allein ist kein zeitgesteuerter Strategie-Runner. Die spätere
Ausführung muss Merkmale und Kerzen anhand von `available_at` freigeben.
Die Metadaten verzögerter Verfügbarkeit werden erhalten; die Freigabe im
Strategie-Runner steht noch aus. Adapter für marktbezogene Kalenderregeln bleiben
ein späteres Arbeitspaket. Unvollständige Zielintervalle werden bereits geprüft
und ausdrücklich abgewiesen.

## Normalisierung und Qualitätsbericht

Die Normalisierung ist eine reine Verarbeitung im Arbeitsspeicher. Sie schreibt
keine Dateien und verändert weder die Eingabefolge noch ihre Kerzen. Beispiel für
eine ausdrücklich gewählte durchgehende Zeitachse mit strikter Lückenbehandlung:

```python
from datetime import timedelta
from pathlib import Path

from market_trader.data import read_candles
from market_trader.normalization import NormalizationConfig, normalize_candles

result = normalize_candles(
    read_candles(Path("synthetic.csv")),
    config=NormalizationConfig(calendar="continuous", gap_policy="reject"),
)
dataset = result.select(instrument="TEST", interval=timedelta(minutes=1))
history = dataset.candles
report = dataset.quality
```

`calendar` muss ausdrücklich angegeben werden: `"continuous"` erwartet zwischen
der ersten und letzten vorhandenen Kerze durchgehende Aktivität; alternativ
beschreibt ein `SessionCalendar` die ausdrücklich aktiven Zeiten. Beide sind
konfigurierbare Prüfprofile, keine Festlegung auf einen Markt. Es werden keine
Feiertage, Wochenenden oder Sitzungen erraten. Unbekannte Kalender und
Lückenregeln werden auch bei leeren Eingaben abgewiesen.

Nach Validierung aller Eingabezeilen gelten folgende Regeln:

- Jede Kombination `(instrument, interval)` bildet einen eigenen Datensatz.
  Datensätze sind nach Instrument und dann Intervall geordnet; ihre Kerzen nach
  `open_time`. Ein leeres Eingangspaket liefert `datasets=()`.
- Der Dublettenschlüssel ist `(instrument, interval, open_time)`. Numerisch
  identische Dezimalwerte gelten auch mit anderen Nachkommastellen als gleich.
  Die erste identische Kerze bleibt erhalten; entfernte Zeilen werden protokolliert.
- Abweichungen bei OHLCV, Kerzenende oder Verfügbarkeit sind Konflikte:
  `DuplicateConflictError` enthält `key`, `first_row`, `row_number` und
  `differing_fields`. Der Aufruf liefert dann kein Teilergebnis.
- Bei `continuous` liegen Kerzenanfänge ohne Auftrag auf dem Raster der ersten
  Kerze ihrer Gruppe, mit Auftrag auf dem Raster von dessen `start_time`;
  bei Sitzungskalendern auf dem Raster des jeweiligen Sitzungsbeginns.
  Überlappungen und nicht passende Intervallabstände sind Datenfehler. Die
  Normalisierung verschiebt keine Zeitstempel auf ein geratenes Börsenraster.
- Eine `CandleGap` beschreibt fehlende Kerzenanfänge in
  `[start_time, end_time)` und ihre Anzahl `missing_count`. Es werden keine
  Ersatzkerzen erzeugt. Vor der ersten und nach der letzten vorhandenen Kerze
  werden ohne angefragten Zeitraum keine Randlücken behauptet.
- `gap_policy="report"` liefert Daten samt Bericht; `"reject"` wirft bei einer
  Lücke `DataGapError`. Dessen `dataset` enthält die Diagnose der abgewiesenen
  Gruppe; `row_number` bezeichnet die Originalzeile unmittelbar nach der ersten
  Lücke. Für reine Endlücken und vollständig fehlende Serien ist es `None`.
  Eine Einbindung dieser Regel in den Backtest-Runner folgt erst in M3.

`DataQualityReport` enthält je Gruppe `input_count`, `output_count`,
`source_rows`, `was_reordered`, `duplicates`, `gaps` und `expected_count`. Mit
einem Auftrag zählt `expected_count` die aktiven, angefragten Kerzen; ohne Auftrag
ist es `None`, weil kein vollständiger Soll-Zeitraum festgelegt ist. `source_rows` ordnet
jeder normalisierten Kerze ihre ursprüngliche Zeile in der gesamten Eingabefolge
zu. `DuplicateRemoval` nennt Schlüssel, `kept_row` und `removed_row`; die Einträge
folgen der ursprünglichen Zeilenreihenfolge. Zeilennummern beginnen bei 1.
`was_reordered` bezieht sich auf die Reihenfolge der eindeutigen Kerzen vor dem
Sortieren. Ergebnisobjekte, Berichte und enthaltene Folgen sind unveränderlich.

## Explizite Sitzungskalender

`market_trader.calendars.TradingSession` enthält `open_time` und `close_time`
für eine halboffene aktive Sitzung `[open_time, close_time)`. Ein unveränderlicher
`SessionCalendar` enthält:

| Feld | Bedeutung |
| --- | --- |
| `name` | Nichtleere, ungepolsterte Kennung des gewählten Zeitplans, beispielsweise `synthetic-v1`. |
| `coverage_start`, `coverage_end` | Positiver UTC-Gültigkeitsbereich `[coverage_start, coverage_end)`, für den die aufgeführten Sitzungen als vollständig konfiguriert gelten. |
| `sessions` | Unveränderliches Tupel chronologisch geordneter, positiver `TradingSession`-Intervalle innerhalb dieses Bereichs, ohne Überschneidungen. Aneinandergrenzende Sitzungen sind erlaubt. |

Alle Zeitstempel sind zeitzonenbehaftete UTC-Werte. Der Kalender wird an der
Normalisierungsgrenze geprüft, auch ohne Kerzen. Ungültige Konfigurationen
erzeugen `ValueError`; ein noch nicht unterstütztes Schema wird nicht geraten.
Ein leeres Sitzungstupel bezeichnet einen vollständig geschlossenen Bereich.

Beispiel mit ausdrücklich angegebener Pause, ohne Bezug zu einer echten Börse:

```python
from datetime import UTC, datetime

from market_trader.calendars import SessionCalendar, TradingSession

calendar = SessionCalendar(
    name="synthetic-v1",
    coverage_start=datetime(2026, 1, 2, tzinfo=UTC),
    coverage_end=datetime(2026, 1, 3, tzinfo=UTC),
    sessions=(
        TradingSession(datetime(2026, 1, 2, 9, tzinfo=UTC),
                       datetime(2026, 1, 2, 9, 3, tzinfo=UTC)),
        TradingSession(datetime(2026, 1, 2, 9, 5, tzinfo=UTC),
                       datetime(2026, 1, 2, 9, 8, tzinfo=UTC)),
    ),
)
config = NormalizationConfig(calendar=calendar, gap_policy="reject")
```

Jede Sitzung muss eine ganze Anzahl der jeweiligen Quellintervalle enthalten;
andernfalls wird die Konfiguration für diese Gruppe abgewiesen. Kerzen müssen
vollständig innerhalb genau einer Sitzung und auf deren Intervallraster liegen.
Eine Kerze darf genau am Sitzungsende schließen; eine dort beginnende Kerze
gehört nur zu einer ausdrücklich anschließenden Sitzung. Kerzen in Pausen,
außerhalb des Gültigkeitsbereichs oder über ein Sitzungsende hinweg erzeugen
`CandleValidationError` mit ihrer Originalzeile und dem betroffenen Zeitfeld.
Es werden keine Eingabekerzen wegen des Kalenders stillschweigend verworfen.
Eine erst nach Sitzungsende verfügbare Kerze bleibt erlaubt, sofern
`available_at >= close_time` gilt.

Lücken zwischen beobachteten Kerzen werden mit den aktiven Sitzungen geschnitten.
Ein über eine Pause reichender Ausfall erzeugt getrennte aktive `CandleGap`-
Einträge; die Pause zählt nicht als fehlende Kerze. Eine vollständig fehlende
Sitzung zwischen zwei beobachteten Kerzen wird ebenfalls erkannt. Die strikte
Regel weist nur fehlende aktive Kerzen ab. Der Gültigkeitsbereich des Kalenders
ist kein angefragter Datenzeitraum: Randlücken vor der ersten oder nach der
letzten Kerze werden mit einem ausdrücklichen `DatasetRequest` geprüft.

Der vollständige Kalender steht in `dataset.config.calendar`. Eine Konfiguration
gilt für alle Instrument-/Intervallgruppen ihres Aufrufs; unterschiedliche
Handelskalender werden in getrennten Aufrufen verarbeitet. Wiederkehrende lokale
Sitzungen, Sommerzeit, automatische Feiertagsregeln und reale Börsenadapter sind
noch nicht implementiert. Fenster und Resampling behalten ihre bisherigen
Verträge; das Resampling richtet Zielblöcke weiterhin an der UTC-Epoche aus und
verwendet noch keine sitzungsbezogene Aggregation.

## Angefragte Datenzeiträume und vollständig fehlende Serien

`market_trader.coverage.DatasetRequest` ist ein unveränderlicher Auftrag mit
`instrument`, positivem `interval` und UTC-Zeitraum `[start_time, end_time)`.
Beispiel für fünf erwartete Minutenkerzen, auch wenn die Eingabe leer ist:

```python
from datetime import UTC, datetime, timedelta

from market_trader.coverage import DatasetRequest
from market_trader.normalization import NormalizationConfig, normalize_candles

request = DatasetRequest(
    instrument="TEST",
    interval=timedelta(minutes=1),
    start_time=datetime(2026, 1, 1, tzinfo=UTC),
    end_time=datetime(2026, 1, 1, 0, 5, tzinfo=UTC),
)
result = normalize_candles(
    (), config=NormalizationConfig(calendar="continuous", gap_policy="report"),
    requests=(request,),
)
dataset = result.select(instrument="TEST", interval=timedelta(minutes=1))
assert dataset.candles == ()
assert dataset.quality.expected_count == 5
assert dataset.quality.gaps[0].missing_count == 5
```

`requests=None` verwendet weiterhin die vorhandene Historie ohne angenommene
Randlücken. Mit Aufträgen gilt:

- Die Folge enthält mindestens einen `DatasetRequest` und höchstens einen pro
  Instrument-/Intervallkombination. Jede Kombination darf ihren eigenen Zeitraum
  haben. Leere Folgen, doppelte Kombinationen und ungültige Angaben erzeugen
  `ValueError`, auch ohne Kerzen.
- Alle angefragten Serien erscheinen im Ergebnis, auch ohne Eingabezeilen.
  Eine vollständig fehlende aktive Serie hat leere Kerzen und Audit-Folgen,
  `input_count=output_count=0` und Lücken für alle angefragten aktiven Intervalle.
  `dataset.request` enthält den Auftrag.
- Eingabekerzen müssen einer angefragten Serie angehören und vollständig im
  angefragten Zeitraum liegen. Beginn am Start ist erlaubt, Beginn am Ende ist
  ausgeschlossen, Schluss genau am Ende erlaubt. Verstöße erzeugen einen
  `CandleValidationError` mit Originalzeile. Ein Auftrag filtert keine Rohdaten;
  eine gewünschte Teilmenge muss ausdrücklich vor dem Aufruf ausgewählt werden.
- Bei `continuous` muss die positive Zeitspanne eine ganze Anzahl von Intervallen
  enthalten; ihr Beginn legt das Kerzenraster fest. Fehlende erste Kerzen führen
  dadurch nicht zu einem verschobenen Raster.
- Bei Sitzungskalendern liegt der gesamte Auftrag innerhalb des deklarierten
  Gültigkeitsbereichs. Grenzen innerhalb aktiver Sitzungen müssen auf deren
  Kerzenraster liegen. Grenzen in geschlossenen Zeiten dürfen frei liegen.
  Sitzungsraster werden weiterhin je Sitzung neu begonnen.
- Lücken am Anfang, zwischen Kerzen und am Ende werden genau einmal gemeldet;
  nur aktive Intervalle zählen. Es gilt
  `expected_count = output_count + sum(gap.missing_count for gap in gaps)`.
  Ein Auftrag ausschließlich über geschlossene Zeit hat `expected_count=0` und
  besteht auch im strikten Modus mit leeren Daten.
- `gap_policy="reject"` weist alle fehlenden aktiven Kerzen ab. Die Diagnose
  bleibt in `DataGapError.dataset`, einschließlich Auftrag und erwarteter Anzahl.
  Existiert nach der ersten Lücke eine Quellzeile, bleibt ihre Originalnummer
  erhalten; andernfalls wird `row_number=None` verwendet.

Die Berechnung verwendet ganze `timedelta`-Intervalle und schneidet Lücken mit
aktiven Sitzungen. Sie erzeugt weder Ersatzkerzen noch eine große Liste aller
erwarteten Kerzenanfänge. Instrumente und Intervalle werden unabhängig geprüft.
Die Eingabefolgen und ihre ursprünglichen Zeilenbezüge bleiben erhalten.

## Reproduzierbare Datenkennung

`NormalizedDataset.dataset_id` hat die Form `sha256:<64 Hexzeichen>`. Sie umfasst
normalisierte Kerzen, Instrument, Intervall in ganzen Mikrosekunden, Kalender,
Lückenregel und die Verarbeitungsversion `market-trader.normalized-candles.v1`.
Eine andere Regel oder ein anderes Resampling-Ergebnis ist unterscheidbar.
Sortierung der Rohzeilen, zusätzliche identische Dubletten und andere numerisch
gleiche Dezimalschreibweisen ändern die Kennung nicht. Quelldateipfad, Zeilen-Audit
und andere Instrumentgruppen gehören nicht zur Inhaltskennung.

Der Hash verwendet UTF-8-JSON-Zeilen mit sortierten Objektschlüsseln, kompakten
Trennzeichen und abschließendem LF je Zeile. Die erste Zeile enthält `format`,
`calendar`, `gap_policy`, `instrument` und `interval_us`; danach folgt je Kerze
eine Liste aus UTC-Zeitstempeln für Beginn, Ende und Verfügbarkeit mit sechs
Nachkommastellen sowie Open, High, Low, Close und Volumen. Dezimalwerte werden
ohne Rundung als vorzeichenbehafteter Koeffizient mit `e` und Exponent kodiert,
ohne nachgestellte Koeffizientennullen; jede Null wird `"0"`. Dadurch sind die
Kennungen unabhängig vom globalen Dezimalkontext und von Float-Rundung.

`calendar` bleibt für das durchgehende Profil der String `"continuous"`. Aufrufe
ohne Auftrag behalten ihre bisherigen Kennungen. Für Sitzungskalender ist es ein
Objekt mit `format="market-trader.session-calendar.v1"`, `name`,
`coverage_start`, `coverage_end` und der geordneten Liste `sessions` aus jeweils
zwei UTC-Zeitstempeln. Name, Gültigkeitsbereich und sämtliche Sitzungen sind Teil
der Verarbeitungsidentität, auch wenn die beobachteten Kerzen gleich bleiben.

Mit Auftrag enthält der Header zusätzlich `request` mit `start_time` und
`end_time` im selben UTC-Format. Dadurch unterscheiden sich andere angefragte
Zeiträume oder eine reine Historie von einer ausdrücklich geprüften Zeitspanne,
auch bei identischem Kerzeninhalt. Eine vollständig fehlende angefragte Serie
hat ebenfalls eine eindeutige Kennung aus Serie, Auftrag und Verarbeitungsregeln.
Die Reihenfolge der Aufträge und andere angefragte Serien ändern sie nicht.

Diese Kennung ist eine Kennung des normalisierten Inhalts und seiner aktuellen
Verarbeitungsregeln. Lokale Artefakte mit vollständigem Datensatz und Herkunftsmanifest
sind gemäß [Speichervertrag](STORAGE_CONTRACT.md) implementiert. Deren zusätzliche
`artifact_id` umfasst auch Qualitätsbericht und Herkunft; die vorhandenen
Datenkennungen bleiben erhalten. Reale Kalenderadapter und die vollständige
Konfiguration späterer Backtest-Läufe folgen in weiteren Arbeitspaketen.

## Umfang der ersten automatisierten Fälle

- DAT-001: CSV-Werte einschließlich Volumen 0,125 und Zeit-/Instrumentzuordnung.
- DAT-003: fehlender Schlusskurs, NaN/±Infinity in jedem OHLCV-Feld, negatives
  Volumen und beide OHLC-Grenzen; Prüfung im Import und in der Validierung.
- DAT-007: der handgerechnete vollständige 5-Minuten-Block aus F-RESAMPLE.
- CAU-001: leere Historie, Warmup, erstes vollständiges Fenster und Fortschaltung.
- CAU-003: numerische Präfix-Invarianz; zusätzlich handgerechnete Werte, damit
  konstante oder leere Ausgaben den Test nicht bestehen können.

Die Strategieaufrufe aus CAU-001 sowie die Bild-/Signalanteile von CAU-003 bleiben
bis zu den zugehörigen Meilensteinen offen. Anleitung und tatsächlicher
Ausführungsstand stehen in [TESTING.md](TESTING.md).

Das zweite Testpaket `tests/test_normalization.py` ergänzt 39 Prüfungen zu
DAT-003/004/005/009/010. Das dritte Paket `tests/test_session_calendars.py` ergänzt
46 Prüfungen zu DAT-005/006/009/010. DAT-005 umfasst nun auch aktive Lücken eines
ausdrücklichen Sitzungskalenders. Die Ablehnung durch einen tatsächlichen
Backtest bleibt bis M3 offen.
Das vierte Paket `tests/test_requested_ranges.py` ergänzt 47 Prüfungen zu
DAT-005/006/009/010: Randlücken, fehlende Serien, Erwartungszahlen, strikte
Ablehnung, Auftragsvalidierung, Grenzen und Kennungen.
Das fünfte Paket `tests/test_dataset_storage.py` ergänzt 57 Prüfungen zu DAT-009
und CAU-004: verlustfreie Speicherung, Herkunfts-/Integritätsprüfung, ungültige
Artefakte, Schutz bestehender Ziele und Fehler während des Schreibens.
