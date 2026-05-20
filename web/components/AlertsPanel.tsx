"use client";

import { useEffect, useState } from "react";
import { type AlertItem, fetchAnomalies, recomputeFraud } from "@/lib/api";

type Props = {
  month?: string;
  onAlertClick?: (a: AlertItem) => void;
};

const KIND_COLOR: Record<string, string> = {
  duplicate_charge:         "bg-red-100   text-red-800   border-red-300",
  card_testing:             "bg-red-200   text-red-900   border-red-400",
  new_merchant_high_amount: "bg-orange-100 text-orange-800 border-orange-300",
  geo_mismatch:             "bg-amber-100 text-amber-800 border-amber-300",
  velocity:                 "bg-rose-100  text-rose-800  border-rose-300",
  round_fx:                 "bg-yellow-100 text-yellow-800 border-yellow-300",
};

export default function AlertsPanel({ month, onAlertClick }: Props) {
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState<string | null>(null);

  async function load() {
    setLoading(true); setError(null);
    try {
      const data = await fetchAnomalies(month);
      setAlerts(data.alerts);
    } catch (e) {
      setError((e as Error).message);
    } finally { setLoading(false); }
  }

  useEffect(() => { load(); }, [month]);

  async function onRecompute() {
    setLoading(true); setError(null);
    try {
      await recomputeFraud({ skipGds: false });
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally { setLoading(false); }
  }

  return (
    <div className="flex flex-col h-full text-sm">
      <header className="flex items-center justify-between p-2 border-b">
        <h3 className="font-medium">
          Alerts {month ? <span className="text-gray-500">· {month}</span> : null}
          <span className="ml-2 text-gray-400">{alerts.length}</span>
        </h3>
        <button
          className="px-2 py-1 text-xs border rounded hover:bg-gray-50"
          onClick={onRecompute}
          disabled={loading}
        >
          {loading ? "Working…" : "Recompute"}
        </button>
      </header>

      {error && <div className="p-2 text-red-600">{error}</div>}

      <ul className="overflow-y-auto flex-1 divide-y">
        {alerts.length === 0 && !loading && (
          <li className="p-3 text-gray-500">No alerts.</li>
        )}
        {alerts.map((a) => (
          <li
            key={a.alert_id}
            className="p-3 cursor-pointer hover:bg-gray-50"
            onClick={() => onAlertClick?.(a)}
          >
            <div className="flex items-center justify-between">
              <span className={`text-xs px-2 py-0.5 border rounded ${KIND_COLOR[a.kind] ?? ""}`}>
                {a.kind}
              </span>
              <span className="text-xs text-gray-500">
                score {a.fraud_score.toFixed(2)}
              </span>
            </div>
            <div className="mt-1 font-medium">
              {a.merchant} <span className="text-gray-500">· £{Math.abs(a.amount).toFixed(2)}</span>
            </div>
            <div className="text-xs text-gray-600">
              {a.date} · {a.location ?? "—"}
            </div>
            <div className="mt-1 text-xs italic text-gray-700">{a.rationale}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}
