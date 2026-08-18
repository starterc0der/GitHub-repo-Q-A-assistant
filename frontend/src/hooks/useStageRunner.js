import { useRef, useState } from "react";

// Reveals all stages of data that has already been fetched in full, immediately —
// no per-stage delay. `onStage` still fires (last id only) so callers that use it to
// set an initial "current" selection keep working.
export function useStageRunner(stageIds) {
  const [statuses, setStatuses] = useState({});
  const [visible, setVisible] = useState({});
  const cancelRef = useRef(0);

  function run(onStage) {
    cancelRef.current += 1;
    const allDone = {};
    const allVisible = {};
    stageIds.forEach((id) => {
      allDone[id] = "done";
      allVisible[id] = true;
    });
    setStatuses(allDone);
    setVisible(allVisible);
    stageIds.forEach((id) => onStage?.(id));
  }

  return { statuses, visible, run };
}
