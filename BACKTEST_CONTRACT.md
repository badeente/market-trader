# Synthetischer Backtest R0

Stand 2026-11-11 · Synthetischer R0-Backtest mit CLI geprüft

## D-02 · Engine

M3 verwendet einen lokalen, deterministischen Kerzensimulator mit genau einer
Kontoführung. Der Umfang ist eine normalisierte Serie, eine Position und
vollständige Ausführungen. Signale und Risikoprüfung übergeben Orders an den
Simulator. Der Bericht verwendet dessen Ausführungen und Kontoverlauf.
Die Beträge und Mengen folgen den bestehenden `Decimal`-Datenverträgen.

Geprüft wurde Backtesting.py 0.6.6 in einer isolierten CPython-3.14.8-Umgebung.
Ein minimales Skript mit synthetischen UTC-Kerzen und vorgegebenen Orders ergab
folgende Ergebnisse. Die Tabellennamen beziehen sich auf [TEST_CASES.md](TEST_CASES.md).

| Fall | Soll in R0 | Beobachtung in Backtesting.py 0.6.6 |
| --- | --- | --- |
| ACC-005 | Gebühren 0,42 Q und Nettogewinn 19,58 Q. | Beide Werte stimmen innerhalb der Testtoleranz. |
| ACC-007 | Bei ausdrücklich aktivierter Endschließung erfolgt der Exit zum letzten Close von 105. | `finalize_trades=True` schließt zum letzten Open von 101 bei `trade_on_close=False`. |
| EXE-005 | Long-Stop 98 bei Open 95 wird zu 95 ausgeführt. | Exit zu 95. |
| EXE-006 | Bei O=100, H=105, L=97, Stop 98 und Ziel 104 erfolgt der Exit zu 98 samt Kennzeichnung der Mehrdeutigkeit. | Exit zu 98. Die Mehrdeutigkeit fehlt im ausgegebenen Trade. |
| EXE-009 | Absoluter Spread 0,2 Q führt bei Referenzen 100 und 110 zu 100,1 und 109,9. | Der relative Spread 0,001 ergibt 100,1 und 110. |

Ein Adapter müsste insbesondere Abschluss- und Kostenregeln eigenständig
abbilden und mit der Bibliotheksbuchhaltung abstimmen. Für R0 liegen
Ausführungsregeln und Kontoführung deshalb im selben Simulator. Die Entscheidung
gilt für den synthetischen Umfang. D-01 bleibt für den ersten realen Markt offen.
[Backtesting.py API-Dokumentation](https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html)
· [Geprüfte Version](https://pypi.org/project/backtesting/0.6.6/)

## D-03 · Ausführung

- **Instrument und Konto.** Das synthetische Instrument `TEST` hat
  Kontraktmultiplikator 1, Abrechnungswährung Q und Startkapital 1.000 Q.
  Standardmenge 1, Preisraster 0,1 Q und Mengenraster 1. Fallkonfigurationen
  dürfen Raster, Mindestmenge und Risikobudget ausdrücklich setzen. Short
  erfordert ausdrückliche Freigabe und vollständige Besicherung des
  Einstiegsnominals. Die Eingangsprüfung verlangt auch für Long ausreichendes
  Kapital für Nominal und Einstiegsgebühr. Im Profil gelten vollständige
  Ausführungen ohne Hebel, Funding oder Teilausführungen.
- **Entscheidung und Marktorder.** Die Strategie entscheidet an Kerzenschlüssen
  und erhält erst bei vollständigem Warmup ein unveränderliches, chronologisches
  Fenster. Jede Kerze darin muss geschlossen sein und `available_at <=`
  Entscheidungszeit erfüllen. Ein Fenster über eine noch unverfügbare Kerze
  wird nicht freigegeben. Eine verspätete Kerze kann ab dem ersten passenden
  Entscheidungszeitpunkt berücksichtigt werden. Eine Marktorder wird zur ersten
  folgenden Kerzeneröffnung ab dem Entscheidungszeitpunkt ausgeführt. Kerzenschluss
  und folgende Eröffnung dürfen denselben UTC-Zeitstempel haben. Ohne folgende
  Eröffnung bleibt die Order als unausgeführt im Bericht.
- **Position und Stop.** `HOLD` lässt den Zustand unverändert. `EXIT` schließt
  die bestehende Position vollständig. Ein Einstieg setzt eine freie Position
  voraus. Beim Einstieg werden Kapital und Lage der Schutzpreise am
  tatsächlichen Ausführungspreis erneut geprüft. Aktive Stop-Market-Orders
  lösen beim Erreichen des Stop-Referenzpreises aus. Liegt die Eröffnung
  bereits jenseits des Stops, ist die Eröffnung die Preisreferenz. Aktive
  Schutzorders greifen bei einer Eröffnung vor einer geplanten Markt-Exit-Order.
- **Ziel und Reihenfolge.** Ein Ziel ist eine Limitorder und wird zum Limit
  oder bei einer günstigeren ausführbaren Eröffnung zu diesem besseren Preis
  gefüllt. Ein neuer Einstieg kann sein Schutzpaar unmittelbar nach dem Fill
  aktivieren. Ist die Reihenfolge zweier Auslösungen innerhalb derselben
  OHLC-Kerze unbekannt, erhält der Stop Vorrang und die Ausführung wird als
  intrabar-mehrdeutig gekennzeichnet. Eindeutige Auslösungen zur Eröffnung werden
  vor der Intrabar-Prüfung verarbeitet. Stop und Ziel sind gekoppelt. Die erste
  Ausführung hebt die andere Order auf.
- **Kosten und Risiko.** Der feste volle Spread wird als halber Spread je Seite
  angewandt. Feste Slippage je Einheit und Seite verschlechtert den Preis
  zusätzlich. Käuferpreise werden zum nächsten zulässigen Tick aufgerundet,
  Verkäuferpreise abgerundet. Beim Ziel muss der so ausführbare Preis das
  Limit wahren. Gebühren sind ein Anteil des tatsächlichen Nominals je
  Ausführung. Spread und Slippage stecken bereits im Fillpreis. Für
  Stop-basierte Größe gilt das Budget einschließlich modellierter Kosten.
  Mengen werden auf das zulässige Raster abgerundet. Relative Stop-Abstände
  werden als Bruchteil des geplanten Einstiegspreises interpretiert und
  vom Einstieg weg auf das Preisraster gerundet. Stop-Risiko bleibt eine
  Planungsgröße. Kurslücken können den realisierten Verlust erhöhen.
- **Laufende Bewertung und Laufende.** Offene Positionen werden standardmäßig
  zum letzten verfügbaren Schlusskurs bewertet. Die explizite Endschließung
  erzeugt nach den regulären Ausführungen genau einen Fill zum letzten Close
  als Preisreferenz und berechnet die konfigurierten Kosten. Kontorendite und
  Drawdown folgen dem Equity-Verlauf einschließlich Startkapital.
- **Eingaben und Bericht.** Der Runner weist aktive Datenlücken und
  nicht unterstützte Profile vor einer erfolgreichen Auswertung ab.
  Signale, Ablehnungen, unausgeführte Orders, Fills, Kosten, Mehrdeutigkeiten,
  offene Position und Equity bleiben nachvollziehbar. Identische Eingaben und
  Profile erzeugen dieselben fachlichen Ergebnisse.

## Implementierte Schnittstellen

`market_trader.backtesting.run_backtest(dataset, strategy, lookback=1, profile=...)`
verarbeitet einen normalisierten Datensatz und ruft die Strategie mit einer
`Decision` aus Zeitpunkt, Kerzenfenster, Konto und Position auf. Die Strategie
liefert ein `Signal` mit Aktion, fester Menge oder `risk_fraction` und bei
Bedarf absoluten Stop-/Zielpreisen. `stop_distance` ist die relative Alternative
zum absoluten Stop. Jeder Aufruf startet mit einem frischen Konto.
Zustandsbehaftete Strategien werden pro Lauf neu instanziiert.

Risikobasierte Orders verwenden zum Entscheidungszeitpunkt das verfügbare Equity
und den letzten Close des Strategie-Fensters als Preisreferenz. Der Plan
speichert erwarteten Einstieg, Stop, Größe, geplantes Stop-Risiko und Budget.
Die Größe bleibt am nächsten Open fest. Dort werden Schutzpreise und Kapital
erneut geprüft. Bei festen Mengen mit absoluten Schutzpreisen ist deren Lage
am tatsächlichen Einstieg maßgeblich.

Das Ergebnis enthält Fills mit Preisreferenz und Gebühr, Trades, Kontostände,
Rendite, Drawdown, offene Position, Ablehnungen und eine gegebenenfalls
unausgeführte Order. `accounts` beginnt mit dem Startkonto und enthält einen
Stand je Kerzenschluss. Intrabar-Fills tragen die Identität der Kerze und die
Phase. Bei expliziter Endschließung enthält der letzte Kontostand den
abgerechneten Fill. Ist der finale Close verspätet, erfolgt die Endschließung
zum Zeitpunkt des Laufendes auf Basis des letzten verfügbaren Close.

`market-trader backtest --config configs/backtest-example.toml` liest die
synthetische CSV über dieselbe Datenpipeline wie der Datenimport und gibt einen
versionierten JSON-Bericht aus. Die interne Registry wählt die Strategie über
`strategy.name`, validiert ihre Parameter und erzeugt pro Lauf eine neue
Instanz. Der Bericht enthält Datensatzkennung,
Quellfingerabdruck, Strategieparameter, Ausführungsprofil, Python- und
Paketversion sowie einen Hash des installierten Python-Quellcodes.
Dezimalwerte werden als Zeichenketten gespeichert. Die [Testdokumentation](TESTING.md)
enthält den Beispielaufruf und seine erwarteten Werte.

Die Backtest-Konfiguration enthält genau eine Datenquelle. `data_config`
verweist auf den CSV-Datenimport und `dataset_artifact` auf ein gespeichertes
Datensatzartefakt. Relative Pfade beziehen sich auf die Backtest-Konfiguration.
Beim Artefakt prüft die CLI die Integrität, übernimmt den gespeicherten
Kalender und die Herkunft und gibt `dataset_artifact_id` im Bericht aus.

## Registrierte Strategien

Der [Strategieüberblick](STRATEGIES.md) erklärt die Signale und zeigt einen
Ridge-Trade aus dem interaktiven Bericht.

| Name | Parameter und Warmup | Signal |
| --- | --- | --- |
| `single_long` | `risk_fraction`, `stop_distance`, eine verfügbare Kerze. | Ein Long-Einstiegssignal beim ersten Aufruf. |
| `sma_crossover` | Ganze Zahlen `0 < fast_period < slow_period`, `slow_period + 1` verfügbare Kerzen und entweder `quantity` oder `risk_fraction` mit `stop_distance`. | Long-Einstieg bei einem Wechsel des schnellen Durchschnitts von unterhalb oder gleich zu oberhalb des langsamen. Ein Wechsel von oberhalb oder gleich zu unterhalb erzeugt bei einer Long-Position ein Ausstiegssignal. |
| `two_legged_pullback` | `risk_fraction`, `movement_threshold`, `bounce_threshold`, `stop_buffer`, `reward_multiple` und vier verfügbare Kerzen. | Long-Einstieg nach einem bestätigten zweiten Tief bei freier Position. Stop und Ziel kommen als absolute Preise in das Signal. |
| `ridge_long` | `model_report`, `entry_threshold`, `quantity` und der im Modell gespeicherte Lookback. | Long-Einstieg oberhalb der festgelegten Renditeschwelle. Der Ausstieg wird nach Ablauf des gespeicherten Horizonts ausgelöst. |

Alle registrierten Strategien handeln Long. `risk_fraction` liegt in
`(0, 1]`. Der relative `stop_distance` von `single_long` und `sma_crossover`
liegt in `(0, 1)`. Der SMA-Vergleich verwendet
ausschließlich Schlusskurse aus dem aktuellen Strategie-Fenster. Die
Konfiguration in [backtest-sma-example.toml](configs/backtest-sma-example.toml)
verwendet dieselben Daten und Ausführungsregeln wie das ursprüngliche Beispiel.
Mit `quantity` verwendet SMA eine feste Menge und steigt beim Down-Cross
oder bei konfigurierter Endschließung aus.

### Eingefrorenes Ridge-Modell

```toml
[strategy]
name = "ridge_long"
model_report = "ridge-validation.json"
entry_threshold = 0.01
quantity = 2
```

`model_report` verweist relativ zur Backtest-TOML oder absolut auf den
gespeicherten Ridge-Validierungsbericht. Beim Laden werden Prüfsumme,
Merkmalsreihenfolge, Zieldefinition und Modellparameter geprüft.
Instrument und Kerzenintervall müssen mit dem Auswertungsdatensatz
übereinstimmen. Seine erste Kerze muss nach dem letzten verfügbaren
Trainingsziel beginnen. Der Backtest-Bericht enthält `model_report_id`
und `horizon_bars`.

Die Vorhersage verwendet dieselbe Merkmalsfunktion wie die
Datenaufbereitung und die gespeicherte Skalierung. Bei freier Position
erzeugt eine Prognose strikt oberhalb von `entry_threshold` ein Long-Signal
mit der festen Menge `quantity`. Die Schwelle ist ein nichtnegativer
Renditeanteil. Spread, Slippage, Preisraster und Gebühren werden bei den
Ausführungen durch R0 berücksichtigt.

Bei expliziten Sitzungen müssen der Lookback und der erwartete Ausstiegs-Open
in derselben Sitzung liegen. Der Ausstieg wird beim ersten verfügbaren
Entscheidungsschluss am oder nach dem Ende des Horizonts ab Einstiegs-Open
eingereicht. R0 füllt ihn zum nächsten Kerzen-Open. Verspätete Kerzen können
diesen Zeitpunkt verschieben. Das Renditeziel des Trainings verwendet
den Schlusskurs der letzten Zielkerze.

### Zweibeiniger Pullback

Die letzten vier verfügbaren Kerzen werden in zeitlicher Reihenfolge ausgewertet.

1. Das erste Tief liegt unter dem Tief der Startkerze. Die Strecke vom Hoch
   der Startkerze zum ersten Tief überschreitet `movement_threshold` relativ
   zum Starthoch.
2. Die Erholungskerze schließt höher als die erste Tiefkerze, hält deren Tief
   und bleibt mit ihrem Hoch unter dem Starthoch. Die Strecke vom ersten Tief
   zum Erholungshoch überschreitet `bounce_threshold` relativ zum ersten Tief.
3. Die vierte Kerze setzt ein tieferes Tief. Ihr Schluss liegt relativ zum
   Schluss der Startkerze um mehr als `movement_threshold` tiefer.

Das Signal entsteht am Schluss der vierten Kerze. Der Stop liegt bei ihrem
Tief abzüglich `stop_buffer` und wird auf das Preisraster abgerundet.
Das Ziel liegt um `reward_multiple` mal die Strecke vom gerundeten Stop
zum Signal-Schlusskurs höher und wird aufgerundet. Die
[Beispielkonfiguration](configs/backtest-pullback-example.toml) setzt
0,3 % Mindestbewegung, 0,2 % Mindesterholung, 0,2 % Stoppuffer und ein
Ziel von 2R vor Kosten. Sie legt den Preistick auf 0,25 Q fest.
Der tatsächliche Einstieg, die Größe und beide Schutzorders folgen den
R0-Regeln. Python-Aufrufer übergeben bei einem abweichenden Raster
`configure_strategy(..., price_tick=profile.price_tick)`.
