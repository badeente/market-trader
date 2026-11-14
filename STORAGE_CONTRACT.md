# Speichervertrag v1: lokale Datensätze und Herkunft

Stand: 2026-11-02 · Status: implementiert und mit synthetischen Daten geprüft

Dieses Paket ergänzt DAT-009 und die Speicherfehlerwege von CAU-004. Es speichert
genau einen `NormalizedDataset` pro JSON-Datei. Kerzen, Qualitätsbericht,
Kalender, Auftrag und Herkunft werden gemeinsam veröffentlicht. Laufzeitbedarf:
nur die Python-Standardbibliothek; kein Netzwerk, Anbieter oder Modell.

## Öffentliche Schnittstellen

Die folgenden Namen werden aus `market_trader.storage` importiert:

| Schnittstelle | Vertrag |
| --- | --- |
| `fingerprint_source(path, *, name)` | `SourceFingerprint(name, sha256, size_bytes)` der unveränderten Quelldatei; SHA-256 über ihre exakten Bytes. |
| `save_dataset(path, dataset, *, provenance)` | Validiert Daten und Herkunft und liefert `DatasetManifest`. Erst vollständige Bytes schreiben, dann am Ziel sichtbar machen; vorhandene Ziele werden abgewiesen. |
| `load_dataset(path)` | Liefert `StoredDataset(dataset, manifest)` nach Format-, Prüfsummen-, Datenkennungs- und Qualitätsprüfung. |
| `verify_sources(manifest, *, paths)` | `None`, wenn alle zugeordneten Quelldateien mit den gespeicherten Fingerabdrücken übereinstimmen; sonst Fehler. |

`DatasetProvenance` enthält eine ausdrückliche, nichtleere `description`, ein
unveränderliches Tupel `sources` und ein Tupel `parent_dataset_ids`.
Quellnamen sind eindeutig, Größen nichtnegativ und SHA-256-Kennungen bestehen aus
`sha256:` und 64 kleinen Hexzeichen. Elternkennungen sind eindeutig. Synthetische
Datensätze dürfen ohne Quelldatei auskommen, müssen aber beschrieben werden.
Abgeleitete Datensätze können ihre normalisierten Vorgänger referenzieren.

`DatasetManifest` enthält `dataset_id`, `artifact_id`, `provenance`, das
Speicherformat `market-trader.dataset-artifact.v1` und die Verarbeitungsversion
`market-trader.normalized-candles.v1`. Ergebnis, Manifest und Herkunftsobjekte
sind unveränderlich. Datentypen stehen in `artifacts.py`, das Dateiverhalten in
`storage.py`, die strikte Kodierung in `_dataset_codec.py`.

## Beispiel: synthetische CSV importieren und wieder laden

Vom Repository-Stamm nach Einrichtung gemäß [TESTING.md](TESTING.md):

```python
from datetime import timedelta
from pathlib import Path

from market_trader.data import read_candles
from market_trader.normalization import NormalizationConfig, normalize_candles
from market_trader.storage import (
    DatasetProvenance, fingerprint_source, load_dataset, save_dataset, verify_sources,
)

raw_path = Path("tests/fixtures/fractional_volume.csv")
dataset = normalize_candles(
    read_candles(raw_path),
    config=NormalizationConfig(calendar="continuous", gap_policy="reject"),
).select(instrument="TEST", interval=timedelta(minutes=1))
provenance = DatasetProvenance(
    description="canonical CSV import of the synthetic fixture",
    sources=(fingerprint_source(raw_path, name="fractional-volume"),),
)
target = Path("datasets/fractional-volume.json")
target.parent.mkdir(parents=True, exist_ok=True)
manifest = save_dataset(target, dataset, provenance=provenance)
restored = load_dataset(target)
assert restored.dataset == dataset
assert restored.manifest == manifest
verify_sources(restored.manifest, paths={"fractional-volume": raw_path})
```

Ein zweites Speichern am selben Ziel erzeugt `FileExistsError`. Für einen weiteren
Snapshot wird ein neuer Zielpfad verwendet. Das Speicherverzeichnis wird vom
Aufrufer angelegt. Relative Pfade beziehen sich auf sein Arbeitsverzeichnis.
`datasets/` ist von Git und Graphify ausgeschlossen; synthetische Test-Fixtures
bleiben versionierbar.

## Dateiformat und Identitäten

Die UTF-8-JSON-Datei hat genau fünf Felder auf oberster Ebene:

- `format`: Speicherformat v1.
- `normalization_format`: ausdrücklich unterstützte Verarbeitungsversion v1.
- `dataset`: der vollständige Datensatz einschließlich `dataset_id`, Kerzen,
  Regeln, Auftrag und ursprünglichem Qualitätsbericht.
- `provenance`: Beschreibung, Quellfingerabdrücke und Vorgängerkennungen.
- `artifact_id`: SHA-256 über die übrigen vier Felder in kanonischer Kodierung.

Datensätze verwenden die Feldnamen ihrer Datenträger aus [DATA_CONTRACT.md](DATA_CONTRACT.md).
Zeitstempel sind UTC-ISO-Strings mit sechs Nachkommastellen; `interval` enthält
ganze Mikrosekunden. OHLCV-Werte sind Dezimalstrings, keine JSON-Floats. Auch
Dezimalschreibweise, kleine Bruchteile und verspätete Verfügbarkeit bleiben
erhalten. Tupel werden JSON-Arrays, optionale Werte `null`. Kalender enthalten
ihre vollständigen Sitzungsobjekte und den Gültigkeitsbereich.

Die kanonische Kodierung verwendet sortierte Objektschlüssel, kompakte
Trennzeichen und UTF-8 ohne ASCII-Ersatz. Der Hash enthält kein abschließendes LF;
die gespeicherte Datei erhält ein LF. Es gibt keine automatisch erzeugte Uhrzeit
oder automatisch eingebetteten lokalen Pfade. Dieselben Daten und dieselbe
Herkunft ergeben identische Bytes.

`dataset_id` bezeichnet weiterhin den normalisierten Inhalt samt Regeln und
Auftrag. `artifact_id` umfasst zusätzlich die konkrete gespeicherte Darstellung,
den Zeilen-Audit und die Herkunft. Zwei Importläufe können denselben normalisierten
Inhalt, aber unterschiedliche Artefaktkennungen besitzen. Die Prüfsummen prüfen
Konsistenz; sie sind keine digitalen Signaturen.

## Validierung und Veröffentlichung

Vor Schreiben und nach Lesen werden die Kerzen erneut nach dem gespeicherten
Datenvertrag normalisiert. Instrument, Intervall, Reihenfolge und Datenkennung
müssen übereinstimmen. Lücken und erwartete Anzahl werden nachgerechnet. Leere
Serien bleiben mit ausdrücklichem Auftrag speicherbar; ein im strikten Profil
wegen Lücken abgewiesener Datensatz wird nicht als gültiges Artefakt gespeichert.

Der ursprüngliche Qualitätsbericht wird erhalten und auf Konsistenz geprüft:
Zählwerte, eindeutige positive Originalzeilen, Sortierungsflag und Dublettenaudit.
Behaltene Dublettenzeilen müssen zur passenden normalisierten Kerze gehören;
entfernte Zeilen sind eindeutig und folgen der ursprünglichen Reihenfolge.
Originalzeilen sind globale Batch-Nummern und dürfen deshalb größer als die
Eingabezahl einer einzelnen Instrumentgruppe sein.

Unbekannte Versionen, fehlende/zusätzliche Felder, doppelte JSON-Schlüssel,
JSON-Floats/NaN/Infinity, falsche Typen und ungültiger Text werden zurückgewiesen.
Auch bei neu berechneter Artefaktprüfsumme werden widersprüchliche Inhalte oder
Qualitätsberichte abgewiesen. Wiederherstellung verwendet nur ausdrücklich
unterstützte Datentypen und keine dynamischen Importe oder Pickle-Ausführung.

Die Datei entsteht zunächst temporär im Zielverzeichnis. Nach vollständigem
Schreiben, Flush und Datei-`fsync` wird sie über einen Hardlink veröffentlicht,
der ein bestehendes Ziel nicht ersetzt. Temporäre Dateien werden bei Erfolg und
behandelten Schreibfehlern entfernt. Dieser Vertrag setzt ein lokales Dateisystem
mit Hardlink-Unterstützung voraus; Datenbank, Cloud-Speicher und Überschreibmodus
sind kein Bestandteil dieses Pakets.

## Herkunft prüfen und Fehler behandeln

`load_dataset` benötigt die Quelldateien nicht. Das Artefakt bleibt nach einem
Umzug lesbar; Quellnamen werden niemals automatisch als Dateipfade geöffnet.
`verify_sources` erhält ausdrücklich die Zuordnung aller logischen Namen zu
lokalen Pfaden und prüft sowohl Bytezahl als auch SHA-256. Geänderte Quellen und
unvollständige/zusätzliche Zuordnungen werden zurückgewiesen. Quelldateien werden
bei diesen Operationen nicht verändert.

Quellfingerabdrücke und Elternkennungen dokumentieren die vom Aufrufer genannte
Herkunft. Sie ersetzen keine erneute Durchführung oder Prüfung der gesamten
Verarbeitungskette. Automatischer Herkunftsaufbau gehört zum nächsten
integrierten Datenablauf; ein Backtest-Runner folgt in M3.

`DatasetStorageError` erbt von `ValueError` und bezeichnet ungültige Artefakte,
Metadaten oder nicht passende Quelldateien. Technische Dateifehler wie
`FileNotFoundError`, `FileExistsError` und `OSError` bleiben als solche erhalten.
Ausführbare Fälle und tatsächlicher Stand stehen in [TESTING.md](TESTING.md).
