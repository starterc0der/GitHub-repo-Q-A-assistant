import { useState } from "react";
import { SpaceView } from "./views/SpaceView.jsx";
import { SpacesView } from "./views/SpacesView.jsx";
import "./App.css";

// Two screens, switched by plain state rather than a router — nothing here needs a
// shareable URL or browser back-button support at this scale.
export default function App() {
  const [spaceId, setSpaceId] = useState(null);

  return spaceId ? (
    <SpaceView spaceId={spaceId} onBack={() => setSpaceId(null)} />
  ) : (
    <SpacesView onOpen={setSpaceId} />
  );
}
