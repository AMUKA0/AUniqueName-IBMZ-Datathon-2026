import pandas as pd
df = pd.read_csv('data/processed/lsoa_risk_replay_2022-12-09.csv')
df['alone_66plus_hh'] = df.pct_alone_66plus / 100 * df.n_households
df['expected_at_risk'] = df.p_trigger_7d * df.alone_66plus_hh
total = df.alone_66plus_hh.sum()
act = df[df.priority == '1 - Act now']
print(f"Act now: {len(act)} neighbourhoods ({len(act)/len(df):.0%}), ~{act.alone_66plus_hh.sum():,.0f} older people living alone ({act.alone_66plus_hh.sum()/total:.0%} of GM)")
for name, order in [('vulnerability-ranked', 'risk_score'), ('people-ranked', 'expected_at_risk')]:
    top = act.sort_values(order, ascending=False).head(20)
    s = top.alone_66plus_hh.sum()
    print(f"Top 20 {name}: ~{s:,.0f} people | {s/total:.1%} of GM from {20/len(df):.1%} of areas | lift {s/total/(20/len(df)):.1f}x")
