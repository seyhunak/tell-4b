"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./globals.css";

type Option = { label: string; key: string; description: string };
type Ticket = {
  id: string; category: string; tag: string; state: string;
  question: string;   options: Option[]; expected: string; adversarial?: string; variant?: string;
  facts: unknown;
};
type Result = {
  label: string | null; status: string; latencyMs: number; clientMs?: number; prompt?: string;
};

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// Adaptive latency formatting: sub-ms engine times used to render as "0.0 ms".
function fmtMs(ms: number): string {
  if (!isFinite(ms)) return "—";
  if (ms < 1) return `${(ms * 1000).toFixed(0)} µs`;
  if (ms < 10) return `${ms.toFixed(2)} ms`;
  return `${ms.toFixed(1)} ms`;
}
function pct(n: number): string {
  return `${(100 * n).toFixed(1)}%`;
}

export default function Page() {
  const [tickets, setTickets] = useState<Ticket[]>([]);
  const [results, setResults] = useState<Record<string, Result>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(220);
  const [filter, setFilter] = useState<"all" | "correct" | "wrong" | "adversarial" | "variations">("all");
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const [runWindow, setRunWindow] = useState<{ start: number; end: number | null } | null>(null);
  const [copied, setCopied] = useState(false);
  const stopRef = useRef(false);

  useEffect(() => {
    fetch("/api/tickets").then((r) => r.json()).then((d) => setTickets(d.tickets));
  }, []);

  const classifyOne = useCallback(async (t: Ticket) => {
    setBusy(t.id);
    const t0 = performance.now();
    const res = await fetch("/api/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(t),
    }).then((r) => r.json());
    res.clientMs = performance.now() - t0;
    setResults((m) => ({ ...m, [t.id]: res }));
    setBusy(null);
    return res as Result;
  }, []);

  const runAll = useCallback(async () => {
    stopRef.current = false;
    setRunning(true);
    setRunWindow((w) => w ?? { start: performance.now(), end: null });
    for (const t of tickets) {
      if (stopRef.current) break;
      if (results[t.id]) continue;
      await classifyOne(t);
      if (speed > 0) await sleep(speed);
    }
    setRunWindow((w) => (w ? { ...w, end: performance.now() } : w));
    setRunning(false);
  }, [tickets, results, classifyOne, speed]);

  const reset = () => { stopRef.current = true; setResults({}); setRunning(false); setBusy(null); setRunWindow(null); };

  const stats = useMemo(() => {
    const done = tickets.filter((t) => results[t.id]).length;
    let correct = 0, invalid = 0;
    const lats: number[] = [];
    const nets: number[] = [];
    const per: Record<string, { n: number; c: number }> = {};
    for (const t of tickets) {
      const r = results[t.id];
      if (!r) continue;
      if (r.label == null) invalid++;
      else if (r.label === t.expected) correct++;
      lats.push(r.latencyMs ?? 0);
      if (r.clientMs != null) nets.push(Math.max(0, r.clientMs - (r.latencyMs ?? 0)));
      const k = t.expected;
      per[k] = per[k] || { n: 0, c: 0 };
      per[k].n++;
      if (r.label === t.expected) per[k].c++;
    }
    lats.sort((a, b) => a - b);
    const q = (p: number) => (lats.length ? lats[Math.min(lats.length - 1, Math.floor(p * lats.length))] : 0);
    const avg = lats.length ? lats.reduce((a, b) => a + b, 0) / lats.length : 0;
    const netAvg = nets.length ? nets.reduce((a, b) => a + b, 0) / nets.length : 0;
    const secs = runWindow ? ((runWindow.end ?? performance.now()) - runWindow.start) / 1000 : 0;
    return {
      done, correct, invalid,
      acc: done ? correct / done : 0,
      inv: done ? invalid / done : 0,
      avg, p50: q(0.5), p95: q(0.95), max: lats.length ? lats[lats.length - 1] : 0,
      netAvg,
      secs: runWindow?.end ? secs : 0,
      tps: runWindow?.end && secs > 0 ? done / secs : 0,
      per,
    };
  }, [results, tickets, runWindow, running]);

  const visible = tickets.filter((t) => {
    if (filter === "adversarial") return !!t.adversarial;
    if (filter === "variations") return !!t.variant;
    const r = results[t.id];
    if (filter === "correct") return r && r.label === t.expected;
    if (filter === "wrong") return r && r.label !== t.expected;
    return true;
  });

  return (
    <div className="wrap">
      <header className="top">
        <div>
          <div className="eyebrow">TELL-4B · SPECIALIST DECISION MODEL</div>
          <h1>Live classification demo</h1>
          <p className="sub">
            50 preloaded tickets — invoice / PO matching, reconciliation triage,
            duplicate detection, tolerance handling, plus adversarial probes. Each ticket is sent to the server in real time and
            answered with exactly one label.
          </p>
        </div>
        <div className="mode"><span className="dot" />demo engine · Tell contract</div>
      </header>

      <div className="stats">
        <div className="stat"><div className="k">CLASSIFIED</div><div className="v">{stats.done}<span style={{fontSize:15,color:"#8b94a7"}}>/50</span></div><div className="sub2">{stats.done === 50 ? "complete" : running ? "running…" : "awaiting run"}</div></div>
        <div className="stat"><div className="k">ACCURACY</div><div className="v">{pct(stats.acc)}</div><div className="sub2">{stats.correct} correct of {stats.done}</div></div>
        <div className="stat"><div className="k">INVALID</div><div className="v">{pct(stats.inv)}</div><div className="sub2">{stats.invalid} abstentions</div></div>
        <div className="stat"><div className="k">ENGINE LATENCY</div><div className="v">{stats.done ? fmtMs(stats.avg) : "—"}</div><div className="sub2">p50 {fmtMs(stats.p50)} · p95 {fmtMs(stats.p95)} · max {fmtMs(stats.max)}</div></div>
        <div className="stat"><div className="k">ROUND-TRIP</div><div className="v">{stats.done ? fmtMs(stats.netAvg) : "—"}</div><div className="sub2">HTTP overhead ex-engine</div></div>
        <div className="stat"><div className="k">THROUGHPUT</div><div className="v">{stats.tps ? stats.tps.toFixed(1) : "—"}{stats.tps ? <span style={{fontSize:14}}> /s</span> : null}</div><div className="sub2">{stats.secs ? `${stats.secs.toFixed(1)}s wall for ${stats.done}` : "full run wall time"}</div></div>
      </div>

      {stats.done > 0 && (
        <div className="perclass">
          <div className="k">PER-CLASS ACCURACY</div>
          {Object.keys(stats.per).sort().map((k) => {
            const p = stats.per[k];
            const a = p.n ? p.c / p.n : 0;
            return (
              <div className="bar-row" key={k}>
                <span className="bar-label">{k}</span>
                <div className="bar"><div style={{ width: `${100 * a}%` }} /></div>
                <span className="bar-num">{p.c}/{p.n} · {pct(a)}</span>
              </div>
            );
          })}
        </div>
      )}

      <div className="controls">
        <button className="primary" onClick={runAll} disabled={running || tickets.length===0 || stats.done===50}>
          {running ? "Classifying…" : stats.done>0 ? "Continue" : "Classify all 50"}
        </button>
        <button className="ghost" onClick={() => { stopRef.current = true; setRunning(false); }} disabled={!running}>Stop</button>
        <button className="ghost" onClick={reset} disabled={stats.done===0 && !running}>Reset</button>
        <button className="ghost" onClick={() => {
          const payload = {
            generated: new Date().toISOString(),
            summary: { done: stats.done, correct: stats.correct, invalid: stats.invalid,
              accuracy: +stats.acc.toFixed(4), invalidRate: +stats.inv.toFixed(4),
              engineAvgMs: +stats.avg.toFixed(4), engineP95Ms: +stats.p95.toFixed(4),
              roundTripAvgMs: +stats.netAvg.toFixed(3), throughputPerSec: +stats.tps.toFixed(2) },
            perClass: Object.fromEntries(Object.entries(stats.per).map(([k, p]) => [k, { n: p.n, correct: p.c, accuracy: +(p.n ? p.c / p.n : 0).toFixed(4) }])),
            tickets: tickets.filter((t) => results[t.id]).map((t) => ({
              id: t.id, category: t.category, tag: t.variant ?? t.adversarial ?? t.tag,
              expected: t.expected, got: results[t.id].label, status: results[t.id].status,
              engineMs: +results[t.id].latencyMs.toFixed(3),
            })),
          };
          navigator.clipboard.writeText(JSON.stringify(payload, null, 2)).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 1600);
          });
        }} disabled={stats.done===0}>{copied ? "Copied" : "Copy results JSON"}</button>
        <label className="speed">pace
          <input type="range" min={0} max={900} step={20} value={speed} onChange={(e)=>setSpeed(+e.target.value)} />
          {speed}ms
        </label>
      </div>

      <div className="filters">
        {(["all","correct","wrong","adversarial","variations"] as const).map((f) => (
          <button key={f} className={"chip"+(filter===f?" on":"")} onClick={()=>setFilter(f)}>{f}</button>
        ))}
      </div>

      <div className="progress"><div style={{width:`${(stats.done/50)*100}%`}} /></div>
      <div className="proglabel">{stats.done} of 50 classified · {stats.correct} correct · {stats.invalid} invalid</div>

      <div className="grid">
        {visible.map((t) => {
          const r = results[t.id];
          const cls = !r ? "pending" : r.label === t.expected ? "ok" : "wrong";
          return (
            <div key={t.id} className={`card ${cls}${open[t.id]?" open":""}`}>
              <div className="row1">
                <span className="tid">{t.id} · {t.category}</span>
                <span className={`tag${t.adversarial?" adv":""}`}>{t.variant ?? t.adversarial ?? t.tag}</span>
              </div>
              <div className="state">{t.state}</div>
              <div className="out">
                <span className={`badge${busy===t.id?" busy":""}`}>{busy===t.id ? "…" : r?.label ?? "·"}</span>
                {r && <span className="meta">expected {t.expected} · {r.status} · {fmtMs(r.latencyMs)} engine</span>}
                {!r && <span className="meta">awaiting classification</span>}
              </div>
              {r?.prompt && (
                <>
                  <button className="toggle" onClick={()=>setOpen(o=>({...o,[t.id]:!o[t.id]}))}>
                    {open[t.id] ? "hide prompt" : "show prompt"}
                  </button>
                  {open[t.id] && <div className="prompt">{r.prompt}</div>}
                </>
              )}
            </div>
          );
        })}
      </div>

      <footer>
        Demo engine: deterministic mirror of Tell-4B&apos;s decision contract (same prompt
        template, same label validation as <code>src/tell/</code>). It reads only structured
        evidence — injections inside STATE are treated as data, never instructions.
        Production inference with real LoRA weights: <code>python scripts/inference.py</code> in
        the parent project. Starter data is synthetic smoke-test material, not production signal.
      </footer>
    </div>
  );
}
