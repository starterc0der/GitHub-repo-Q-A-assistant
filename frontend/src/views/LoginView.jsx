import { useState } from "react";
import { login, signup } from "../api.js";

const FEATURES = [
  { num: "01", title: "Any source, plain English", desc: "Repos, PDFs, docs, or pasted text — ask like you would a teammate." },
  { num: "02", title: "Cited, not guessed", desc: "Every answer links to the exact file and line range it came from." },
  { num: "03", title: "Organized by space", desc: "Group any mix of sources into spaces and share them with your team." },
];

export function LoginView({ onAuthed }) {
  const [mode, setMode] = useState("login"); // "login" | "signup"
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const isSignup = mode === "signup";

  async function submit(e) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const user = isSignup ? await signup({ email, name, password }) : await login({ email, password });
      onAuthed(user);
    } catch (err) {
      setError(err.message.replace(/^\/auth\/\w+ failed \(\d+\): /, ""));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rag-landing">
      <div className="rag-landing__hero">
        <span className="rag-landing__dotgrid" />
        <span className="rag-landing__ring" />

        <div className="rag-landing__brand">
          <span className="rag-spaces__brand-mark">S</span>
          <div>
            <div className="rag-landing__title">Sift</div>
            <div className="rag-landing__tagline">Q&amp;A over your knowledge</div>
          </div>
        </div>

        <div>
          <div className="rag-landing__headline">Ask your repos, docs,<br />and PDFs a question.</div>
          <div className="rag-landing__sub">
            Point Sift at a codebase, a PDF, or pasted text and get answers with citations back to the exact source — not a guess.
          </div>
        </div>

        <div className="rag-landing__features">
          {FEATURES.map((f) => (
            <div className="rag-landing__feature" key={f.title}>
              <span className="rag-landing__feature-num">{f.num}</span>
              <div>
                <div className="rag-landing__feature-title">{f.title}</div>
                <div className="rag-landing__feature-desc">{f.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="rag-landing__form-wrap">
        <form className="rag-landing-card" onSubmit={submit}>
          <div className="rag-landing-card__tabs">
            <button
              type="button"
              className={`rag-landing-card__tab${!isSignup ? " rag-landing-card__tab--active" : ""}`}
              onClick={() => { setMode("login"); setError(null); }}
            >
              Log in
            </button>
            <button
              type="button"
              className={`rag-landing-card__tab${isSignup ? " rag-landing-card__tab--active" : ""}`}
              onClick={() => { setMode("signup"); setError(null); }}
            >
              Create account
            </button>
          </div>

          <div className="rag-landing-card__heading">{isSignup ? "Create your account" : "Welcome back"}</div>
          <div className="rag-landing-card__subheading">
            {isSignup ? "Start asking questions about any codebase in minutes." : "Log in to keep exploring your repos."}
          </div>

          {error && <p className="rag-error" style={{ marginBottom: 4 }}>{error}</p>}

          <label className="rag-input-card__label">Email</label>
          <input
            className="rag-input" type="email" placeholder="you@company.com" value={email} required autoFocus
            onChange={(e) => setEmail(e.target.value)} disabled={busy}
          />

          {isSignup && (
            <>
              <label className="rag-input-card__label">Name</label>
              <input
                className="rag-input" placeholder="Ada Lovelace" value={name} required
                onChange={(e) => setName(e.target.value)} disabled={busy}
              />
            </>
          )}

          <label className="rag-input-card__label">Password</label>
          <div className="rag-password-field">
            <input
              className="rag-input" type={showPassword ? "text" : "password"} placeholder="At least 8 characters"
              value={password} required minLength={8} onChange={(e) => setPassword(e.target.value)} disabled={busy}
            />
            <button
              type="button" className="rag-password-field__toggle rag-password-field__toggle--text" onClick={() => setShowPassword((s) => !s)}
              disabled={busy} aria-label={showPassword ? "Hide password" : "Show password"}
            >
              {showPassword ? "Hide" : "Show"}
            </button>
          </div>

          <button type="submit" className="rag-btn rag-landing-card__submit" disabled={busy}>
            {busy ? "Please wait…" : isSignup ? "Create account" : "Log in"}
          </button>
        </form>
      </div>
    </div>
  );
}
