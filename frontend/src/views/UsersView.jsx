import { useEffect, useState } from "react";
import { assignSpace, listSpaces, listUsers, logout, unassignSpace, updateUserRole } from "../api.js";
import { AVATAR_BG, AVATAR_INK, timeAgo } from "../components/RagAtoms.jsx";

function RoleModal({ user, currentUser, onClose, onChanged }) {
  const [role, setRole] = useState(user.role);
  const [busy, setBusy] = useState(false);
  const isSelf = user.id === currentUser.id;

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    try {
      await updateUserRole(user.id, role);
      onChanged();
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rag-modal-backdrop" onClick={onClose}>
      <form className="rag-modal-card" onClick={(e) => e.stopPropagation()} onSubmit={submit}>
        <h2>Change role — {user.name}</h2>
        <label className="rag-input-card__label">Role</label>
        <select className="rag-input" value={role} onChange={(e) => setRole(e.target.value)} disabled={busy}>
          <option value="user">User</option>
          <option value="admin">Admin</option>
        </select>
        {isSelf && role !== "admin" && (
          <p className="rag-error">You're changing your own role — you'll lose admin access immediately.</p>
        )}
        <div className="rag-modal-card__actions">
          <button type="button" className="rag-btn rag-btn--ghost" onClick={onClose}>Cancel</button>
          <button type="submit" className="rag-btn" disabled={busy || role === user.role}>
            {busy ? "Saving…" : "Save"}
          </button>
        </div>
      </form>
    </div>
  );
}

function AssignSpaceModal({ user, allSpaces, onClose, onChanged }) {
  const assignedIds = new Set(user.spaces.map((s) => s.id));
  const available = allSpaces.filter((s) => !assignedIds.has(s.id));
  const [spaceId, setSpaceId] = useState(available[0]?.id || "");
  const [busy, setBusy] = useState(false);

  async function submit(e) {
    e.preventDefault();
    if (!spaceId) return;
    setBusy(true);
    try {
      await assignSpace(user.id, spaceId);
      onChanged();
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rag-modal-backdrop" onClick={onClose}>
      <form className="rag-modal-card" onClick={(e) => e.stopPropagation()} onSubmit={submit}>
        <h2>Assign a space — {user.name}</h2>
        {available.length === 0 ? (
          <p className="rag-hint">Already assigned to every space.</p>
        ) : (
          <>
            <label className="rag-input-card__label">Space</label>
            <select className="rag-input" value={spaceId} onChange={(e) => setSpaceId(e.target.value)} disabled={busy}>
              {available.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
            </select>
          </>
        )}
        <div className="rag-modal-card__actions">
          <button type="button" className="rag-btn rag-btn--ghost" onClick={onClose}>Cancel</button>
          <button type="submit" className="rag-btn" disabled={busy || available.length === 0}>
            {busy ? "Assigning…" : "Assign"}
          </button>
        </div>
      </form>
    </div>
  );
}

const FILTERS = [
  { key: "all", label: "All" },
  { key: "admin", label: "Admins" },
  { key: "user", label: "Members" },
];

function UserRow({ user, currentUser, allSpaces, onOpenUser, onChanged }) {
  const [roleModal, setRoleModal] = useState(false);
  const [assignModal, setAssignModal] = useState(false);

  async function removeSpace(e, spaceId) {
    e.stopPropagation();
    await unassignSpace(user.id, spaceId);
    onChanged();
  }

  return (
    <div className="rag-users-table__row" onClick={() => onOpenUser(user.id)}>
      <div className="rag-users-table__name">
        <span className="rag-avatar" style={{ background: AVATAR_BG.accent, color: AVATAR_INK.accent }}>
          {user.name.trim().charAt(0).toUpperCase() || "?"}
        </span>
        <span className="rag-users-table__name-body">
          <span className="rag-users-table__name-text">{user.name}</span>
          <span className="rag-dim">{user.email}</span>
        </span>
      </div>
      <button
        type="button"
        className={`rag-users-table__role${user.role === "admin" ? " rag-users-table__role--admin" : ""}`}
        onClick={(e) => { e.stopPropagation(); setRoleModal(true); }}
      >
        {user.role}
      </button>
      <div className="rag-users__spaces" onClick={(e) => e.stopPropagation()}>
        {user.spaces.map((s) => (
          <span key={s.id} className="rag-users__space-chip">
            {s.name}
            <button type="button" className="rag-users__space-chip-x" onClick={(e) => removeSpace(e, s.id)}>×</button>
          </span>
        ))}
        <button type="button" className="rag-users__assign-btn" onClick={() => setAssignModal(true)}>+ Add</button>
      </div>
      <span className="rag-mono rag-dim">{new Date(user.created_at).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })}</span>
      <span className="rag-mono rag-dim">{user.last_active ? timeAgo(user.last_active) : "—"}</span>
      <span className="rag-dim">›</span>

      {roleModal && (
        <RoleModal
          user={user} currentUser={currentUser} onClose={() => setRoleModal(false)}
          onChanged={() => { setRoleModal(false); onChanged(); }}
        />
      )}
      {assignModal && (
        <AssignSpaceModal
          user={user} allSpaces={allSpaces} onClose={() => setAssignModal(false)}
          onChanged={() => { setAssignModal(false); onChanged(); }}
        />
      )}
    </div>
  );
}

export function UsersView({ currentUser, onBack, onOpenUser, onLoggedOut }) {
  const [users, setUsers] = useState(null);
  const [allSpaces, setAllSpaces] = useState([]);
  const [search, setSearch] = useState("");
  const [roleFilter, setRoleFilter] = useState("all");
  const [error, setError] = useState(null);

  function refresh() {
    setError(null);
    Promise.all([listUsers(), listSpaces()])
      .then(([u, s]) => { setUsers(u.users); setAllSpaces(s.spaces); })
      .catch((err) => setError(err.message));
  }

  useEffect(refresh, []);

  const filtered = users?.filter((u) => {
    if (roleFilter !== "all" && u.role !== roleFilter) return false;
    const q = search.trim().toLowerCase();
    return !q || u.name.toLowerCase().includes(q) || u.email.toLowerCase().includes(q);
  });

  const admins = users?.filter((u) => u.role === "admin").length || 0;
  const total = users?.length || 0;
  const spacesInUse = users ? new Set(users.flatMap((u) => u.spaces.map((s) => s.id))).size : 0;

  return (
    <div className="rag-spaces-shell">
      <aside className="rag-spaces__sidebar">
        <button className="rag-space-sidebar__back" onClick={onBack}>
          <span className="rag-space-sidebar__back-icon">←</span>Back to spaces
        </button>
        <div className="rag-spaces__brand">
          <span className="rag-spaces__brand-mark">S</span>
          <div><h1>Users</h1><p>Roles &amp; space access</p></div>
        </div>
        <div>
          <div className="rag-spaces__nav-label">Workspace</div>
          <div className="rag-spaces__nav-list">
            <button className="rag-spaces__nav-item" onClick={onBack}>
              <span className="rag-spaces__nav-dot" />
              All spaces
            </button>
            <button className="rag-spaces__nav-item rag-spaces__nav-item--active">
              <span className="rag-spaces__nav-dot" />
              Users
            </button>
          </div>
        </div>
        <div className="rag-spaces__directory">
          <div className="rag-spaces__nav-label">Directory</div>
          <div className="rag-user-sidebar__row"><span>Total users</span><span className="rag-mono">{total}</span></div>
          <div className="rag-user-sidebar__row"><span>Admins</span><span className="rag-mono">{admins}</span></div>
        </div>
        <div className="rag-spaces__footer">
          <div className="rag-spaces__footer-identity">
            <span className="rag-spaces__footer-avatar">{currentUser?.name?.trim().charAt(0).toUpperCase() || "?"}</span>
            <span className="rag-spaces__footer-name">{currentUser?.name}</span>
          </div>
          <button className="rag-spaces__logout-btn" onClick={() => logout().finally(onLoggedOut)}>
            Log out
          </button>
        </div>
      </aside>

      <div className="rag-spaces__main">
        <div className="rag-spaces__container">
          <div className="rag-spaces__header">
            <div>
              <h1>Users</h1>
              <p>Manage roles and which spaces each person can access.</p>
            </div>
            <div className="rag-users-toolbar">
              <input
                className="rag-input rag-users__search" placeholder="Search users…"
                value={search} onChange={(e) => setSearch(e.target.value)}
              />
              <div className="rag-range-toggle">
                {FILTERS.map((f) => (
                  <button
                    key={f.key}
                    className={`rag-range-toggle__btn${roleFilter === f.key ? " rag-range-toggle__btn--active" : ""}`}
                    onClick={() => setRoleFilter(f.key)}
                  >
                    {f.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {error ? (
            <p className="rag-error">Could not reach the backend: {error}</p>
          ) : users === null ? (
            <p className="rag-hint">Loading…</p>
          ) : (
            <>
              <div className="rag-insight-cards">
                <div className="rag-insight-card">
                  <span className="rag-insight-card__label">Total users</span>
                  <span className="rag-insight-card__value">{total}</span>
                </div>
                <div className="rag-insight-card">
                  <span className="rag-insight-card__label">Admins</span>
                  <span className="rag-insight-card__value" style={{ color: "var(--accent-ink)" }}>{admins}</span>
                </div>
                <div className="rag-insight-card">
                  <span className="rag-insight-card__label">Members</span>
                  <span className="rag-insight-card__value">{total - admins}</span>
                </div>
                <div className="rag-insight-card">
                  <span className="rag-insight-card__label">Spaces in use</span>
                  <span className="rag-insight-card__value">{spacesInUse}/{allSpaces.length}</span>
                </div>
              </div>

              <div className="rag-users-table">
                <div className="rag-users-table__head">
                  <span>Name</span><span>Role</span><span>Spaces</span><span>Joined</span><span>Last active</span><span />
                </div>
                {filtered.length === 0 ? (
                  <p className="rag-hint" style={{ padding: 32, textAlign: "center" }}>No users match your search.</p>
                ) : (
                  filtered.map((u) => (
                    <UserRow
                      key={u.id} user={u} currentUser={currentUser} allSpaces={allSpaces}
                      onOpenUser={onOpenUser} onChanged={refresh}
                    />
                  ))
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
