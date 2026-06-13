import dash
from dash import dcc, html, dash_table
import plotly.graph_objects as go
import plotly.express as px
import requests, pandas as pd

app = dash.Dash(__name__, title="VSM Optimization Dashboard")
API = "http://localhost:5000/api"

def fetch(endpoint):
    try:
        return requests.get(f"{API}/{endpoint}").json()
    except:
        return []

def layout():
    kpis    = fetch("metrics/efficiency")
    vsm     = fetch("vsm/summary")
    bns     = fetch("vsm/bottlenecks")
    recs    = fetch("optimize/recommendations")

    df_vsm  = pd.DataFrame(vsm)  if vsm  else pd.DataFrame()
    df_bns  = pd.DataFrame(bns)  if bns  else pd.DataFrame()

    # ── KPI Cards ───────────────────────────────────
    kpi_cards = html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
        html.Div(style={"background":c,"borderRadius":"8px","padding":"16px",
                         "minWidth":"160px","color":"white"}, children=[
            html.P(label, style={"margin":"0","fontSize":"13px","opacity":"0.85"}),
            html.H2(str(val), style={"margin":"4px 0 0 0","fontSize":"26px"})
        ])
        for label, val, c in [
            ("Overall Efficiency", f'{kpis.get("overall_avg_efficiency_pct","—")}%', "#1B4F72"),
            ("Avg Lead Time",      f'{kpis.get("avg_lead_time_min","—")} min',      "#17A589"),
            ("Avg Utilization",    f'{round(kpis.get("avg_machine_utilization",0)*100,1)}%', "#E67E22"),
            ("Bottleneck Procs",   kpis.get("bottleneck_process_count","—"),          "#C0392B"),
            ("Avg Defect Rate",    f'{round(kpis.get("avg_defect_rate",0)*100,2)}%', "#8E44AD"),
        ]
    ])

    # ── Bottleneck Bar Chart ─────────────────────────
    fig_bn = go.Figure()
    if not df_bns.empty:
        colors = ["#C0392B" if s>0.8 else "#E67E22" if s>0.55 else "#17A589"
                  for s in df_bns["score"]]
        fig_bn.add_trace(go.Bar(x=df_bns["Process_Name"], y=df_bns["score"],
                                marker_color=colors, name="Bottleneck Score"))
    fig_bn.update_layout(title="Bottleneck Score by Process",
                         yaxis_title="Score (0–1)", xaxis_tickangle=-30,
                         plot_bgcolor="#F8F9FA", paper_bgcolor="white")

    # ── Machine Utilization Chart ────────────────────
    fig_util = px.bar(df_vsm, x="Process_Name", y="avg_utilization",
                      color="avg_utilization",
                      color_continuous_scale=["#2ECC71","#F39C12","#E74C3C"],
                      title="Average Machine Utilization per Process") if not df_vsm.empty else go.Figure()
    fig_util.update_layout(xaxis_tickangle=-30, plot_bgcolor="#F8F9FA")

    # ── Lead Time vs Cycle Time ──────────────────────
    fig_lt = go.Figure()
    if not df_vsm.empty:
        fig_lt.add_trace(go.Bar(name="Cycle Time",   x=df_vsm["Process_Name"], y=df_vsm["avg_cycle_time"]))
        fig_lt.add_trace(go.Bar(name="Waiting Time", x=df_vsm["Process_Name"], y=df_vsm["avg_waiting_time"]))
    fig_lt.update_layout(barmode="stack", title="Cycle Time vs Waiting Time Stack",
                         xaxis_tickangle=-30, plot_bgcolor="#F8F9FA")

    # ── Recommendations Table ────────────────────────
    rec_rows = []
    for r in recs[:10]:
        for rec in r.get("recommendations",[]):
            rec_rows.append({"Process":r["process"],"Priority":r["priority"],"Recommendation":rec})
    df_recs = pd.DataFrame(rec_rows)

    return html.Div(style={"fontFamily":"Arial","padding":"24px","background":"#ECF0F1"}, children=[
        html.H1("VSM-ML Production Optimization Dashboard",
                style={"color":"#1B3A5C","borderBottom":"3px solid #2E6DA4","paddingBottom":"8px"}),
        html.P("Real-time bottleneck detection · Lean waste identification · Optimization recommendations",
               style={"color":"#555","marginBottom":"24px"}),
        kpi_cards,
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px","marginTop":"24px"}, children=[
            dcc.Graph(figure=fig_bn),
            dcc.Graph(figure=fig_util),
            dcc.Graph(figure=fig_lt, style={"gridColumn":"span 2"}),
        ]),
        html.H2("Optimization Recommendations", style={"color":"#1B3A5C","marginTop":"24px"}),
        dash_table.DataTable(
            data=df_recs.to_dict("records"),
            columns=[{"name":c,"id":c} for c in df_recs.columns] if not df_recs.empty else [],
            style_cell={"textAlign":"left","padding":"10px","fontFamily":"Arial","fontSize":"13px"},
            style_header={"backgroundColor":"#1B3A5C","color":"white","fontWeight":"bold"},
            style_data_conditional=[
                {"if":{"filter_query":"{Priority} = HIGH"},  "backgroundColor":"#FADBD8","color":"#922B21"},
                {"if":{"filter_query":"{Priority} = MEDIUM"},"backgroundColor":"#FDEBD0","color":"#784212"},
            ],
            page_size=15,
        ),
        dcc.Interval(id="refresh", interval=60000),
    ])

app.layout = layout

if __name__ == "__main__":
    print("Dashboard: http://localhost:8050")
    app.run(debug=True, port=8050)