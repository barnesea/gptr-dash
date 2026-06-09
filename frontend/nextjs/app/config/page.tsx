"use client";

import { useEffect, useMemo, useState } from "react";

type RuntimeOption = {
  key: string;
  category: string;
  default: unknown;
  type: string;
  value: unknown;
  override: string;
  source: string;
  sensitive: boolean;
};

type RuntimeConfig = {
  path: string;
  options: RuntimeOption[];
  overrides: Record<string, string>;
};

const stringifyValue = (value: unknown) => {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
};

const maskValue = (value: string) => {
  if (!value) return "";
  if (value.length <= 8) return "****";
  return `${value.slice(0, 4)}****${value.slice(-4)}`;
};

export default function ConfigPage() {
  const [config, setConfig] = useState<RuntimeConfig | null>(null);
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("All");
  const [showSensitive, setShowSensitive] = useState(false);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState("");

  const loadConfig = async () => {
    const response = await fetch("/api/config", { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Failed to load config");
    setConfig(data);
    setDraft(data.overrides || {});
  };

  useEffect(() => {
    loadConfig().catch((error) => setStatus(error.message));
  }, []);

  const categories = useMemo(() => {
    const names = new Set(config?.options.map((option) => option.category) || []);
    return ["All", ...Array.from(names).sort()];
  }, [config]);

  const filteredOptions = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return (config?.options || []).filter((option) => {
      const categoryMatch = category === "All" || option.category === category;
      const queryMatch =
        !needle ||
        option.key.toLowerCase().includes(needle) ||
        option.type.toLowerCase().includes(needle) ||
        option.source.toLowerCase().includes(needle);
      return categoryMatch && queryMatch;
    });
  }, [config, query, category]);

  const updateDraft = (key: string, value: string) => {
    setDraft((current) => ({ ...current, [key]: value }));
  };

  const save = async (persist: boolean) => {
    setSaving(true);
    setStatus("");
    try {
      const response = await fetch("/api/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ overrides: draft, persist }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Failed to save config");
      setConfig(data);
      setDraft(data.overrides || {});
      setStatus(persist ? "Saved defaults" : "Applied runtime overrides");
    } catch (error: any) {
      setStatus(error.message || "Failed to save config");
    } finally {
      setSaving(false);
    }
  };

  const reset = async () => {
    setSaving(true);
    setStatus("");
    try {
      const response = await fetch("/api/config", { method: "DELETE" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Failed to reset config");
      setConfig(data);
      setDraft({});
      setStatus("Reset overrides");
    } catch (error: any) {
      setStatus(error.message || "Failed to reset config");
    } finally {
      setSaving(false);
    }
  };

  return (
    <main className="min-h-screen bg-[#0C111F] px-4 py-6 text-slate-100 md:px-8">
      <div className="mx-auto flex max-w-7xl flex-col gap-5">
        <header className="flex flex-col gap-4 border-b border-slate-800 pb-5 md:flex-row md:items-end md:justify-between">
          <div>
            <a href="/" className="text-sm text-teal-300 hover:text-teal-200">
              GPT Researcher
            </a>
            <h1 className="mt-2 text-2xl font-semibold tracking-normal text-white md:text-3xl">
              Runtime Configuration
            </h1>
            <p className="mt-1 max-w-3xl text-sm text-slate-400">
              Active values are read from the backend process. Saved defaults are written to {config?.path || "data/runtime-config.json"}.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => save(false)}
              disabled={saving || !config}
              className="h-10 rounded-md border border-slate-700 px-4 text-sm text-slate-100 hover:border-teal-400 disabled:opacity-50"
            >
              Apply Runtime
            </button>
            <button
              type="button"
              onClick={() => save(true)}
              disabled={saving || !config}
              className="h-10 rounded-md bg-teal-500 px-4 text-sm font-medium text-slate-950 hover:bg-teal-400 disabled:opacity-50"
            >
              Save Defaults
            </button>
            <button
              type="button"
              onClick={reset}
              disabled={saving || !config}
              className="h-10 rounded-md border border-red-500/50 px-4 text-sm text-red-200 hover:border-red-400 disabled:opacity-50"
            >
              Reset
            </button>
          </div>
        </header>

        <section className="grid gap-3 border-b border-slate-800 pb-4 md:grid-cols-[1fr_220px_auto]">
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search options"
            className="h-10 rounded-md border border-slate-700 bg-slate-950 px-3 text-sm text-white outline-none focus:border-teal-400"
          />
          <select
            value={category}
            onChange={(event) => setCategory(event.target.value)}
            className="h-10 rounded-md border border-slate-700 bg-slate-950 px-3 text-sm text-white outline-none focus:border-teal-400"
          >
            {categories.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
          <label className="flex h-10 items-center gap-2 rounded-md border border-slate-700 px-3 text-sm text-slate-300">
            <input
              type="checkbox"
              checked={showSensitive}
              onChange={(event) => setShowSensitive(event.target.checked)}
              className="h-4 w-4 accent-teal-500"
            />
            Show secrets
          </label>
        </section>

        {status && (
          <div className="rounded-md border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-200">
            {status}
          </div>
        )}

        <section className="overflow-hidden rounded-md border border-slate-800">
          <div className="max-h-[calc(100vh-260px)] overflow-auto">
            <table className="w-full min-w-[1080px] border-collapse text-left text-sm">
              <thead className="sticky top-0 z-10 bg-slate-950 text-xs uppercase text-slate-400">
                <tr>
                  <th className="w-[260px] border-b border-slate-800 px-3 py-3">Option</th>
                  <th className="w-[130px] border-b border-slate-800 px-3 py-3">Type</th>
                  <th className="border-b border-slate-800 px-3 py-3">Current</th>
                  <th className="border-b border-slate-800 px-3 py-3">Default</th>
                  <th className="w-[320px] border-b border-slate-800 px-3 py-3">Override</th>
                </tr>
              </thead>
              <tbody>
                {filteredOptions.map((option) => {
                  const currentValue = stringifyValue(option.value);
                  const defaultValue = stringifyValue(option.default);
                  const displayCurrent = option.sensitive && !showSensitive ? maskValue(currentValue) : currentValue;
                  const displayDefault = option.sensitive && !showSensitive ? maskValue(defaultValue) : defaultValue;

                  return (
                    <tr key={option.key} className="border-b border-slate-900 bg-slate-950/60">
                      <td className="align-top px-3 py-3">
                        <div className="font-mono text-xs text-slate-100">{option.key}</div>
                        <div className="mt-1 flex gap-2 text-xs text-slate-500">
                          <span>{option.category}</span>
                          <span>-</span>
                          <span>{option.source}</span>
                        </div>
                      </td>
                      <td className="align-top px-3 py-3 font-mono text-xs text-slate-400">{option.type}</td>
                      <td className="max-w-[280px] align-top px-3 py-3">
                        <pre className="whitespace-pre-wrap break-words font-mono text-xs text-slate-200">{displayCurrent}</pre>
                      </td>
                      <td className="max-w-[280px] align-top px-3 py-3">
                        <pre className="whitespace-pre-wrap break-words font-mono text-xs text-slate-500">{displayDefault}</pre>
                      </td>
                      <td className="align-top px-3 py-3">
                        <textarea
                          value={draft[option.key] ?? ""}
                          onChange={(event) => updateDraft(option.key, event.target.value)}
                          placeholder="empty clears override"
                          spellCheck={false}
                          className="min-h-10 w-full resize-y rounded-md border border-slate-800 bg-slate-900 px-2 py-2 font-mono text-xs text-white outline-none focus:border-teal-400"
                        />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </main>
  );
}
