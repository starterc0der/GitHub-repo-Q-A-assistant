import { Component } from "react";

// Without a boundary, one bad point or a stale trace shape unmounts the entire app and
// leaves a blank white page with the reason only visible in devtools. This is a debugging
// tool — a broken stage should say what broke and leave every other stage readable.
export class RagErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error("Stage render failed:", error, info?.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="rag-boundary">
        <p className="rag-boundary__title">{this.props.label || "This view"} failed to render.</p>
        <pre className="rag-boundary__msg">{String(this.state.error?.message || this.state.error)}</pre>
        <p className="rag-boundary__hint">
          If you just updated the app, a hard reload (Ctrl/Cmd+Shift+R) clears stale state
          left behind by hot-reloading. Full stack trace is in the browser console.
        </p>
        <button className="rag-btn" onClick={() => this.setState({ error: null })}>
          Retry
        </button>
      </div>
    );
  }
}
