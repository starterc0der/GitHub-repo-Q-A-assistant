import { useRef, useState } from "react";

const STEP_MS = 650;

// Drives a step-by-step reveal animation over data that has already been fetched in
// full — each stage is "shown" in sequence rather than actually re-fetched per stage.
export function useStageRunner(stageIds) {
  const [statuses, setStatuses] = useState({});
  const [visible, setVisible] = useState({});
  const [running, setRunning] = useState(false);
  const cancelRef = useRef(0);

  function run(onStage) {
    cancelRef.current += 1;
    const token = cancelRef.current;
    const allPending = {};
    const noneVisible = {};
    stageIds.forEach((id) => {
      allPending[id] = "pending";
      noneVisible[id] = false;
    });
    setStatuses(allPending);
    setVisible(noneVisible);
    setRunning(true);

    function step(i) {
      if (cancelRef.current !== token) return;
      if (i >= stageIds.length) {
        setRunning(false);
        return;
      }
      const id = stageIds[i];
      setStatuses((s) => ({ ...s, [id]: "active" }));
      setVisible((v) => ({ ...v, [id]: true }));
      onStage?.(id);
      setTimeout(() => {
        if (cancelRef.current !== token) return;
        setStatuses((s) => ({ ...s, [id]: "done" }));
        setTimeout(() => step(i + 1), 150);
      }, STEP_MS);
    }
    step(0);
  }

  return { statuses, visible, running, run };
}
