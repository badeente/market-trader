"""Attach transparent AP1 protocol/validation summaries to a validated report."""

from decimal import Decimal, localcontext
from html import escape
import json


def enrich_reference_report(html: bytes, *, summary: dict) -> bytes:
    """Preserve the interactive ledger view; never label validation as final testing."""
    if summary["test_metrics_computed"] is not False:
        raise ValueError("AP1 reserved test metrics must not be computed")
    document = html.decode("utf-8")
    if "<main>" not in document:
        raise ValueError("reference report template has no main content")
    with localcontext() as context:
        context.prec = 28
        forecast_rows = []
        for key, label in (("ridge", "Ridge"), ("zero_return", "Nullvorhersage")):
            metrics = summary["validation_forecast"][key]
            cells = [escape(label)]
            for metric in ("mae", "rmse"):
                value = Decimal(str(metrics[metric]))
                if not value.is_finite() or value < 0:
                    raise ValueError("reference forecast errors must be finite and nonnegative")
                cells.append(f"{value * 10000:.3f}")
            forecast_rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
        trade_rows = []
        for key, label in (("ridge", "Ridge > 0"), ("sma_timed", "SMA 4 > SMA 8")):
            metrics = summary["trading"][key]
            cells = [label, metrics["trade_count"], metrics["reference_points"],
                     metrics["modeled_cost_points"], metrics["net_points"], metrics["max_drawdown_points"]]
            trade_rows.append("<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in cells) + "</tr>")
        ridge_months = summary["trading"]["ridge"]["months"]
        sma_months = summary["trading"]["sma_timed"]["months"]
        monthly_rows = []
        for ridge, sma in zip(ridge_months, sma_months, strict=True):
            if ridge["month"] != sma["month"] or not ridge["month"].startswith("2024-"):
                raise ValueError("reference month summaries must align in 2024")
            cells = [ridge["month"], ridge["trade_count"], ridge["net_points"], sma["trade_count"], sma["net_points"]]
            monthly_rows.append("<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in cells) + "</tr>")
    provenance = escape(json.dumps({key: value for key, value in summary.items() if key not in
                                   ("trading", "validation_forecast")}, ensure_ascii=False, sort_keys=True, indent=2))
    panel = (
        '<style>#ap1-protocol h2{margin-top:1.2rem}#ap1-protocol p{margin-top:.6rem}'
        '#ap1-protocol table{width:100%;border-collapse:collapse;margin-top:.7rem;font-variant-numeric:tabular-nums}'
        '#ap1-protocol th,#ap1-protocol td{text-align:left;padding:.4rem;border-bottom:1px solid var(--border)}'
        '#ap1-protocol .scroll{overflow-x:auto}</style>'
        '<section class="panel" id="ap1-protocol"><h1>AP1 · Validierung 2024</h1>'
        '<p><strong>2025 bleibt reserviert. Keine finalen Testmetriken berechnet.</strong></p>'
        '<p>Trainingszeitraum 2021–2023; Auswertung 2024. 15-Minuten-Bid-Kerzen von Dukascopy, '
        'acht abgeschlossene Kerzen Lookback, 30 Minuten Open-to-Open-Ziel. '
        'Forschungsfenster Montag–Freitag 09:00–17:30 Europe/Berlin; keine Overnight-Positionen.</p>'
        f'<p>Trainingsbeispiele: {escape(str(summary["training_rows"]))}; '
        f'Validierungsbeispiele: {escape(str(summary["validation_rows"]))}. '
        'Die überlappenden Fenster sind keine unabhängigen Marktbeobachtungen.</p>'
        '<h2>Vorhersagequalität – nur Validierung</h2>'
        '<p>MAE und RMSE in Basispunkten der Bruttorendite; niedriger ist besser. '
        'Bessere Vorhersagefehler bedeuten nicht automatisch höhere Nettogewinne.</p>'
        '<div class="scroll"><table><thead><tr><th>Modell</th><th>MAE (bp)</th><th>RMSE (bp)</th></tr></thead><tbody>'
        + ''.join(forecast_rows) + '</tbody></table></div>'
        '<h2>Handelsergebnis – Punkte je fester Einheit</h2>'
        '<p>Maximal eine Long-Position, identischer Zeitausstieg und Kosten für beide Regeln. '
        'Keine Stops, Gewinnziele, Hebel oder Parameteroptimierung.</p>'
        '<p><strong>Keine bestätigten Brokerkonditionen:</strong> synthetischer voller Spread 2,0 Punkte, '
        'Slippage 0,1 Punkt je Ausführung, separate Kommission 0. Preisraster 0,001, '
        'eine Einheit; Startmittel 100000 synthetische Q. Keine Euro-Gewinn- oder Kontogrößenempfehlung. '
        'Die Prozentkurve unten hängt von dieser festen Buchungsgröße ab.</p>'
        '<div class="scroll"><table><thead><tr><th>Regel</th><th>Trades</th><th>Referenzbewegung</th>'
        '<th>Modellierte Kosten</th><th>Netto</th><th>Max. Drawdown (Punkte)</th></tr></thead><tbody>'
        + ''.join(trade_rows) + '</tbody></table></div>'
        '<h2>Monatliche Nettopunkte 2024</h2><div class="scroll"><table><thead><tr><th>Monat</th>'
        '<th>Ridge Trades</th><th>Ridge Netto</th><th>SMA Trades</th><th>SMA Netto</th></tr></thead><tbody>'
        + ''.join(monthly_rows) + '</tbody></table></div>'
        '<p>Datenfilter: Nur vollständige beobachtete Forschungstage. Partielle und ganz fehlende '
        'Werktage werden protokolliert, nicht aufgefüllt und nicht ohne Quelle zu Feiertagen erklärt. '
        'Dieser Filter kann Auswahlverzerrungen verursachen und ist kein offizieller Produktkalender.</p>'
        '<details><summary>Protokoll, Datenidentität und Ausschlussmanifest</summary><pre>'
        + provenance + '</pre></details></section>'
    )
    return document.replace("<main>", "<main>" + panel, 1).replace(
        "<title>Market Trader · Backtest report</title>", "<title>AP1 · Validierung 2024</title>", 1
    ).encode("utf-8")
