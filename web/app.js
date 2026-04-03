const output = document.getElementById("output");
const runBtn = document.getElementById("runBtn");

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(",").map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const cols = line.split(",");
    const row = {};
    headers.forEach((h, i) => {
      row[h] = (cols[i] || "").trim();
    });
    return row;
  });
}

function toNum(v, fallback) {
  const n = Number(v);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

function kelly(prob, odds) {
  const b = odds - 1;
  if (b <= 0) return 0;
  const edge = prob * (b + 1) - 1;
  return Math.max(0, edge / b);
}

function runBacktest(matches, opts) {
  const history = new Map();
  const pointsMap = { H: [3, 0], D: [1, 1], A: [0, 3] };

  const enriched = matches.map((m) => {
    const home = m.HomeTeam;
    const away = m.AwayTeam;
    const homeHist = (history.get(home) || []).slice(-opts.lookback);
    const awayHist = (history.get(away) || []).slice(-opts.lookback);

    const homeForm = homeHist.length ? homeHist.reduce((a, b) => a + b, 0) / (3 * homeHist.length) : 0.5;
    const awayForm = awayHist.length ? awayHist.reduce((a, b) => a + b, 0) / (3 * awayHist.length) : 0.5;

    const ftr = m.FTR;
    const points = pointsMap[ftr] || [0, 0];
    history.set(home, [...(history.get(home) || []), points[0]]);
    history.set(away, [...(history.get(away) || []), points[1]]);

    return {
      ...m,
      homeForm,
      awayForm,
      oddsH: toNum(m.B365H, 2.5),
      oddsD: toNum(m.B365D, 3.1),
      oddsA: toNum(m.B365A, 2.9),
    };
  });

  const split = Math.max(1, Math.floor(enriched.length * opts.trainFraction));
  const test = enriched.slice(split);

  let bankroll = opts.initialBankroll;
  let totalStake = 0;
  let totalProfit = 0;
  let bets = 0;

  for (const m of test) {
    const x = m.homeForm - m.awayForm;
    const pHome = Math.min(0.8, Math.max(0.1, 0.45 + x * 0.35));
    const pAway = Math.min(0.8, Math.max(0.1, 0.35 - x * 0.25));
    const pDraw = Math.max(0.05, 1 - pHome - pAway);
    const norm = pHome + pAway + pDraw;
    const p = { H: pHome / norm, D: pDraw / norm, A: pAway / norm };

    const odds = { H: m.oddsH, D: m.oddsD, A: m.oddsA };
    const edges = {
      H: p.H * odds.H - 1,
      D: p.D * odds.D - 1,
      A: p.A * odds.A - 1,
    };

    const side = Object.keys(edges).reduce((best, key) => (edges[key] > edges[best] ? key : best), "H");
    if (edges[side] < opts.minEdge) continue;

    const frac = Math.min(kelly(p[side], odds[side]), opts.maxKelly);
    const stake = bankroll * frac;
    if (stake <= 0) continue;

    bets += 1;
    totalStake += stake;
    const won = m.FTR === side;
    const profit = won ? stake * (odds[side] - 1) : -stake;
    totalProfit += profit;
    bankroll += profit;
  }

  return {
    matches: enriched.length,
    test_matches: test.length,
    bets_placed: bets,
    roi: totalStake ? totalProfit / totalStake : 0,
    final_bankroll: bankroll,
    note: "Browser version uses heuristic probabilities for mobile demo deployment.",
  };
}

runBtn.addEventListener("click", async () => {
  const file = document.getElementById("csvFile").files[0];
  if (!file) {
    output.textContent = "Please choose a CSV file first.";
    return;
  }

  const text = await file.text();
  const rows = parseCsv(text);
  const required = ["HomeTeam", "AwayTeam", "FTR"];
  const missing = required.filter((col) => !(col in rows[0]));
  if (missing.length) {
    output.textContent = `Missing required columns: ${missing.join(", ")}`;
    return;
  }

  const result = runBacktest(rows, {
    lookback: Number(document.getElementById("lookback").value || 5),
    trainFraction: Number(document.getElementById("trainFraction").value || 0.8),
    minEdge: Number(document.getElementById("minEdge").value || 0.02),
    maxKelly: Number(document.getElementById("maxKelly").value || 0.1),
    initialBankroll: Number(document.getElementById("bankroll").value || 1000),
  });

  output.textContent = JSON.stringify(result, null, 2);
});
