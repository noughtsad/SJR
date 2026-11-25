"use client";

import { useEffect, useState } from "react";
import Chat from "@/components/Chat";
import ForYou from "@/components/ForYou";
import Search from "@/components/Search";

const TABS = ["Search", "For you", "Career agent"] as const;
type Tab = (typeof TABS)[number];

export default function Home() {
  const [tab, setTab] = useState<Tab>("Search");
  const [user, setUser] = useState("demo");

  // remember the handle per browser (a convenience only; there is no auth in this demo)
  useEffect(() => {
    try {
      const saved = localStorage.getItem("sjr-user");
      if (saved) setUser(saved);
    } catch {}
  }, []);
  const changeUser = (u: string) => {
    setUser(u);
    try {
      localStorage.setItem("sjr-user", u);
    } catch {}
  };

  return (
    <main>
      <h1>Job search &amp; career agent</h1>
      <p className="sub">
        Hybrid search over a live jobs index, resume-based recommendations, and an agent that only cites real postings.
      </p>
      <div className="row" style={{ marginBottom: 12 }}>
        <label className="check" htmlFor="user">
          User handle
        </label>
        <input id="user" type="text" value={user} onChange={(e) => changeUser(e.target.value.trim() || "demo")}
          style={{ maxWidth: 180, flex: "none" }} />
      </div>
      <div className="tabs" role="tablist">
        {TABS.map((t) => (
          <button key={t} role="tab" className="tab" aria-selected={tab === t} onClick={() => setTab(t)}>
            {t}
          </button>
        ))}
      </div>
      {tab === "Search" && <Search user={user} />}
      {tab === "For you" && <ForYou user={user} />}
      {tab === "Career agent" && <Chat user={user} />}
    </main>
  );
}
