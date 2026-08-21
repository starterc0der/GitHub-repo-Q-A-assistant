import { useEffect, useState } from "react";
import { me } from "./api.js";
import { LoginView } from "./views/LoginView.jsx";
import { SpaceView } from "./views/SpaceView.jsx";
import { SpacesView } from "./views/SpacesView.jsx";
import { UserInsightsView } from "./views/UserInsightsView.jsx";
import { UsersView } from "./views/UsersView.jsx";
import "./App.css";

// Screens switched by plain state rather than a router — nothing here needs a
// shareable URL or browser back-button support at this scale.
export default function App() {
  const [currentUser, setCurrentUser] = useState(undefined); // undefined = still checking
  const [screen, setScreen] = useState("spaces"); // "spaces" | "space" | "users" | "user-detail"
  const [spaceId, setSpaceId] = useState(null);
  const [userDetailId, setUserDetailId] = useState(null);

  useEffect(() => {
    me()
      .then(setCurrentUser)
      .catch(() => setCurrentUser(null));
  }, []);

  if (currentUser === undefined) return null; // avoid a login-screen flash while checking
  if (currentUser === null) return <LoginView onAuthed={setCurrentUser} />;

  function openSpace(id) {
    setSpaceId(id);
    setScreen("space");
  }

  function openUsers() {
    setScreen("users");
  }

  function openUserDetail(id) {
    setUserDetailId(id);
    setScreen("user-detail");
  }

  if (screen === "space") {
    return <SpaceView spaceId={spaceId} currentUser={currentUser} onBack={() => setScreen("spaces")} />;
  }
  if (screen === "users") {
    return (
      <UsersView
        currentUser={currentUser}
        onBack={() => setScreen("spaces")}
        onOpenUser={openUserDetail}
        onLoggedOut={() => setCurrentUser(null)}
      />
    );
  }
  if (screen === "user-detail") {
    return <UserInsightsView userId={userDetailId} onBack={() => setScreen("users")} />;
  }
  return (
    <SpacesView currentUser={currentUser} onOpen={openSpace} onOpenUsers={openUsers} onLoggedOut={() => setCurrentUser(null)} />
  );
}
